import json

import pytest

from app.agents.agent import ask
from app.agents.config import AgentConfig
from app.agents.schemas import AgentStatus, StopReason
from app.tools.registry import reference

from .conftest import answer, call, reply, search_args

B = "b" * 64


def test_search_to_grounded_answer_with_real_retrieval(index, llm, client):
    record = index.get_record(f"{B}:39")
    assert record is not None
    citation = reference(record).model_dump(mode="json")
    client.responses.create.side_effect = [
        reply(
            calls=[
                {"type": "reasoning", "id": "reason_1", "summary": []},
                call(arguments=search_args()),
            ]
        ),
        reply(
            answer=answer(
                sources=[citation],
                answer_type="document",
                text="Quantity: -2.5; source: boq.xlsx, Structural, row 38.",
            )
        ),
    ]
    result = ask("What is the quantity for item 2.3?", index, llm)
    assert result.sources[0].row_start == 38
    assert result.sources[0].record_id == record.id
    assert "-2.5" in result.answer
    assert client.responses.create.call_count == 2
    continuation = client.responses.create.call_args.kwargs
    assert continuation["previous_response_id"] == "resp_first"
    assert continuation["tool_choice"] == "auto"
    output = continuation["input"][1]
    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call_1"
    data = json.loads(output["output"])["data"]
    assert data["results"][0]["record"]["boq"]["quantity"] == -2.5
    assert data["results"][0]["citation"] == citation


@pytest.mark.parametrize("name", ["get_record", "list_documents"])
def test_model_can_select_other_registered_tools(index, llm, client, name):
    arguments = {"record_id": f"{B}:39"} if name == "get_record" else {}

    def final(**kwargs):
        output = json.loads(kwargs["input"][1]["output"])["data"]
        citation = (
            output["citation"]
            if name == "get_record"
            else output["documents"][0]["citation"]
        )
        return reply(
            answer=answer(
                sources=[citation],
                answer_type="document",
                text="See the cited document.",
            )
        )

    first = reply(calls=[call(name, arguments)])
    client.responses.create.side_effect = lambda **kwargs: (
        final(**kwargs) if kwargs["previous_response_id"] else first
    )
    assert ask("Find document evidence", index, llm).sources


def test_multiple_calls_have_matching_ids_and_context(index, llm, client):
    client.responses.create.side_effect = [
        reply(
            calls=[
                call("search_documents", search_args(query="unmatched_xyz"), "search"),
                call("get_record", {"record_id": "missing"}, "get"),
                call("list_documents", {}, "list"),
            ]
        ),
        reply(
            answer=answer(
                answer_type="insufficient", text="The requested item was not found."
            )
        ),
    ]
    result = ask("Find missing data", index, llm)
    assert result.answer_type == "insufficient"
    outputs = client.responses.create.call_args.kwargs["input"][1:]
    assert [item["call_id"] for item in outputs] == ["search", "get", "list"]
    assert all(json.loads(item["output"])["ok"] for item in outputs)


def test_direct_general_answer_needs_one_request(index, llm, client):
    client.responses.create.return_value = reply(answer=answer())
    result = ask("What is a BOQ?", index, llm)
    assert result.answer_type == "general"
    assert not result.sources
    client.responses.create.assert_called_once()


@pytest.mark.parametrize(
    "response",
    [
        reply(status="incomplete"),
        reply(status="failed"),
        reply(refusal=True),
        reply(raw=""),
        reply(raw="not json"),
        reply(raw='{"answer":"missing fields"}'),
    ],
)
def test_invalid_or_refused_model_responses_are_controlled(
    index, llm, client, response
):
    client.responses.create.return_value = response
    result = ask("Question", index, llm)
    assert result.status == AgentStatus.FAILED
    assert result.stop_reason == StopReason.INVALID_RESPONSE
    assert client.responses.create.call_count == 1


@pytest.mark.parametrize(
    "modification",
    [
        {"record_id": "fabricated"},
        {"filename": "invented.xlsx"},
        {"row_start": 100},
        {"page": 99},
        {"sheet": "Other"},
        {"record_id": f"{B}:40"},
    ],
)
def test_citations_must_match_evidence_actually_supplied(
    index, llm, client, modification
):
    record = index.get_record(f"{B}:39")
    assert record is not None
    citation = {**reference(record).model_dump(mode="json"), **modification}
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer(answer_type="document", sources=[citation])),
    ]
    result = ask("Find item 2.3", index, llm)
    assert result.status == AgentStatus.FAILED
    assert "not supplied" in result.answer


def test_direct_answer_cannot_invent_citations(index, llm, client):
    record = index.get_record(f"{B}:39")
    assert record is not None
    client.responses.create.return_value = reply(
        answer=answer(
            answer_type="document", sources=[reference(record).model_dump(mode="json")]
        )
    )
    result = ask("Find item 2.3", index, llm)
    assert result.status == AgentStatus.FAILED
    assert "not supplied" in result.answer


def test_document_answer_requires_sources(index, llm, client):
    client.responses.create.return_value = reply(answer=answer(answer_type="document"))
    result = ask("Find a quantity", index, llm)
    assert result.status == AgentStatus.FAILED
    assert "require supporting" in result.answer


def test_evidence_cannot_be_labeled_general_to_omit_sources(index, llm, client):
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer()),
    ]
    result = ask("Find item 2.3", index, llm)
    assert result.status == AgentStatus.FAILED
    assert "cite its sources" in result.answer


def test_partial_evidence_warning_is_attached_even_if_model_omits_it(
    index, llm, client
):
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args(query="excav", limit=2))]),
        reply(
            answer=answer(
                answer_type="insufficient",
                text="I cannot establish a total from these records.",
            )
        ),
    ]
    result = ask("Total excavation?", index, llm)
    assert any("incomplete" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    "bad_call", [call("unknown", {}), {**call(), "arguments": "{"}]
)
def test_bad_tool_call_outputs_are_returned_to_model(index, llm, client, bad_call):
    client.responses.create.side_effect = [
        reply(calls=[bad_call]),
        reply(answer=answer(answer_type="insufficient", text="Retrieval failed.")),
    ]
    result = ask("Find evidence", index, llm)
    output = json.loads(client.responses.create.call_args.kwargs["input"][1]["output"])
    assert not output["ok"]
    assert result.warnings


def test_second_round_continues_to_final_answer(index, llm, client):
    client.responses.create.side_effect = [
        reply(calls=[call("list_documents", call_id="first")]),
        reply(calls=[call("list_documents", call_id="second")]),
        reply(
            answer=answer(answer_type="insufficient", text="Please specify a document.")
        ),
    ]
    result = ask("Plan many steps", index, llm)
    assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert result.tool_calls == 2
    assert client.responses.create.call_count == 3


@pytest.mark.parametrize(
    "calls",
    [
        [call("list_documents", call_id=str(number)) for number in range(9)],
        [call("list_documents"), call("list_documents")],
    ],
)
def test_tool_batch_is_bounded(index, llm, client, calls):
    client.responses.create.return_value = reply(calls=calls)
    result = ask("Question", index, llm, config=AgentConfig(max_tool_calls=8))
    assert result.stop_reason in {
        StopReason.MAX_TOOL_CALLS,
        StopReason.INVALID_RESPONSE,
    }
    assert result.tool_calls == 0
    assert client.responses.create.call_count == 1


def test_final_incomplete_response_is_not_accepted(index, llm, client):
    client.responses.create.side_effect = [
        reply(calls=[call("list_documents")]),
        reply(status="incomplete"),
    ]
    result = ask("List files", index, llm)
    assert result.status == AgentStatus.FAILED
    assert "incomplete" in result.answer


@pytest.mark.parametrize("question", ["", "   ", "x" * 16001])
def test_invalid_question_does_not_contact_provider(index, llm, client, question):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ask(question, index, llm)
    client.responses.create.assert_not_called()
