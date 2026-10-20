import json
import logging

import pytest

from app.agents.agent import ask, run_agent
from app.agents.config import AgentConfig
from app.agents.schemas import AgentStatus, StopReason
from app.tools.registry import reference

from .conftest import answer, call, reply, search_args

B = "b" * 64


def cited(index, record_id=f"{B}:39"):
    record = index.get_record(record_id)
    assert record is not None
    return reference(record).model_dump(mode="json")


def test_observations_drive_multidocument_search_then_full_record(index, llm, client):
    pdf = next(
        entry.record
        for entry in index.entries.values()
        if entry.record.source.format == "pdf" and entry.record.kind == "paragraph"
    )
    responses = [
        reply(calls=[call(arguments=search_args(), call_id="boq")]),
        reply(calls=[call("get_record", {"record_id": pdf.id}, "spec")]),
        reply(calls=[call("get_record", {"record_id": f"{B}:39"}, "full_boq")]),
        reply(
            answer=answer(
                answer_type="document",
                sources=[cited(index), cited(index, pdf.id)],
                text="The BOQ and specification provide different kinds of evidence; compare the cited records.",
            )
        ),
    ]
    requests = []

    def respond(**kwargs):
        requests.append(kwargs)
        if len(requests) > 1:
            observed = json.loads(kwargs["input"][1]["output"])["data"]
            if len(requests) == 2:
                assert observed["results"][0]["record"]["boq"]["quantity"] == -2.5
            if len(requests) == 3:
                assert observed["record"]["id"] == pdf.id
                assert observed["record"]["source"]["page"] == pdf.source.page
        response = responses[len(requests) - 1]
        response.id = f"response_{len(requests)}"
        return response

    client.responses.create.side_effect = respond
    state = run_agent("Compare the BOQ with the specification", index, llm)
    assert state.status == AgentStatus.COMPLETED
    assert state.iterations == 4
    assert state.tool_calls == 3
    assert [request["previous_response_id"] for request in requests] == [
        None,
        "response_1",
        "response_2",
        "response_3",
    ]
    assert [request["input"][1]["call_id"] for request in requests[1:]] == [
        "boq",
        "spec",
        "full_boq",
    ]
    assert len({request["instructions"] for request in requests}) == 1
    assert all(request["tool_choice"] == "auto" for request in requests)
    assert len(state.sources) == 2
    assert set(state.evidence) == {f"{B}:39", pdf.id}
    assert state.evidence[f"{B}:39"]["boq"]["quantity"] == -2.5
    assert state.input_tokens == 400
    assert state.output_tokens == 200


@pytest.mark.parametrize(
    "bad",
    [
        call("unknown"),
        {**call(), "arguments": "{"},
        call(arguments=search_args(filters={"kind": "invalid"})),
    ],
)
def test_invalid_tool_call_can_be_corrected_in_next_round(index, llm, client, bad):
    client.responses.create.side_effect = [
        reply(calls=[bad]),
        reply(calls=[call(arguments=search_args(), call_id="corrected")]),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
    ]
    state = run_agent("Find item 2.3", index, llm)
    assert state.status == AgentStatus.COMPLETED
    assert [outcome.success for outcome in state.outcomes] == [False, True]
    assert state.tool_calls == 2
    assert state.iterations == 3
    second = client.responses.create.call_args_list[1].kwargs
    assert not json.loads(second["input"][1]["output"])["ok"]
    assert state.final_response is not None
    assert state.final_response.warnings


def test_empty_search_can_be_reformulated(index, llm, client):
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args(query="nonexistent_zxy"))]),
        reply(calls=[call(arguments=search_args(), call_id="alternative")]),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
    ]
    state = run_agent("Find an item", index, llm)
    assert state.status == AgentStatus.COMPLETED
    assert [outcome.returned for outcome in state.outcomes] == [0, 1]


def test_insufficient_evidence_is_a_valid_distinct_terminal_status(index, llm, client):
    client.responses.create.return_value = reply(
        answer=answer(
            answer_type="insufficient",
            text="The collection does not establish a reliable quantity.",
        )
    )
    result = ask("Find missing quantity", index, llm)
    assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert result.stop_reason == StopReason.INSUFFICIENT_EVIDENCE
    assert result.iterations == 1


def test_debug_events_contain_metadata_without_question_evidence_or_reasoning(
    index, llm, client, caplog
):
    client.responses.create.side_effect = [
        reply(
            calls=[
                {
                    "type": "reasoning",
                    "id": "hidden_reason",
                    "summary": [{"type": "summary_text", "text": "PRIVATE REASONING"}],
                },
                call(arguments=search_args(query="PRIVATE QUESTION")),
            ]
        ),
        reply(answer=answer(answer_type="insufficient", text="PRIVATE FINAL ANSWER")),
    ]
    with caplog.at_level(logging.INFO):
        state = run_agent("PRIVATE QUESTION", index, llm)
    assert "PRIVATE" not in caplog.text
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "app.agents.events"
    ]
    assert {event["event"] for event in events} >= {
        "agent_started",
        "llm_requested",
        "tool_finished",
        "agent_stopped",
    }
    last = events[-1]
    assert last["session_id"] == state.session_id
    assert last["execution_id"] == state.execution_id
    assert last["iterations"] == 2
    assert last["tool_calls"] == 1
    tool = next(event for event in events if event["event"] == "tool_finished")
    assert tool["arguments"]["query_chars"] == len("PRIVATE QUESTION")
    assert tool["returned"] == 0


def test_duplicate_records_are_references_but_full_inspection_is_new_evidence(
    index, llm, client
):
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(calls=[call("get_record", {"record_id": f"{B}:39"}, "full")]),
        reply(calls=[call("get_record", {"record_id": f"{B}:39"}, "repeat")]),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
    ]
    state = run_agent("Inspect the complete item", index, llm)
    assert state.status == AgentStatus.COMPLETED
    full = json.loads(
        client.responses.create.call_args_list[2].kwargs["input"][1]["output"]
    )["data"]["record"]
    repeated = json.loads(
        client.responses.create.call_args_list[3].kwargs["input"][1]["output"]
    )["data"]["record"]
    assert full["boq"]["quantity"] == -2.5
    assert "metadata" in full
    assert repeated["already_supplied"]
    assert repeated["record_id"] == f"{B}:39"
    assert "boq" not in repeated
    assert len(state.sources) == 1
    assert state.evidence[f"{B}:39"]["boq"]["quantity"] == -2.5


def test_partial_exhaustive_search_preserves_completeness_flags(index, llm, client):
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args(query="excav", exhaustive=True))]),
        reply(
            answer=answer(
                answer_type="insufficient",
                text="A reliable total cannot be established from this incomplete result.",
            )
        ),
    ]
    result = ask(
        "Total excavation", index, llm, config=AgentConfig(max_tool_output_bytes=7000)
    )
    assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert any("incomplete" in warning for warning in result.warnings)
    data = json.loads(client.responses.create.call_args.kwargs["input"][1]["output"])[
        "data"
    ]
    assert not data["complete"]
    assert data["has_more"]
    assert data["next_offset"] == data["returned"]


def test_conflicting_line_values_remain_separate_evidence(index, llm, client):
    ids = [f"{B}:39", f"{B}:40"]
    originals = [index.get_record(record_id) for record_id in ids]
    assert all(record is not None for record in originals)
    client.responses.create.side_effect = [
        reply(
            calls=[
                call("get_record", {"record_id": record_id}, record_id)
                for record_id in ids
            ]
        ),
        reply(
            answer=answer(
                answer_type="document",
                sources=[cited(index, record_id) for record_id in ids],
                text="These are distinct BOQ entries; their quantities must not be silently merged or treated as a verified total.",
            )
        ),
    ]
    state = run_agent("Compare these two line items", index, llm)
    assert state.status == AgentStatus.COMPLETED
    outputs = client.responses.create.call_args.kwargs["input"][1:]
    assert len(outputs) == 2
    for record_id, original, item in zip(ids, originals, outputs, strict=True):
        assert original is not None
        assert original.boq is not None
        sent = json.loads(item["output"])["data"]["record"]
        assert sent["boq"] == original.boq.model_dump(mode="json")
        assert state.evidence[record_id]["boq"] == sent["boq"]
    assert len(state.sources) == 2
