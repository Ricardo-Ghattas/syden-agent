import json

import httpx2
import pytest
from openai import OpenAI

from app.agents.agent import run_agent
from app.agents.config import AgentConfig
from app.agents.llm import LLM, Settings
from app.agents.schemas import AgentStatus, StopReason
from app.agents.state import ConversationSession
from app.tools.registry import encode, reference

from .conftest import answer, call, reply, search_args


@pytest.mark.parametrize(
    "base_url,stored",
    [
        ("https://openrouter.ai/api/v1", False),
        ("https://api.openai.com/v1", True),
        ("https://openrouter.ai.other.example/v1", True),
    ],
)
def test_openrouter_is_selected_from_endpoint(monkeypatch, base_url, stored):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-unused")
    monkeypatch.setenv("OPENAI_MODEL", "minimax/minimax-m3.1-flash:free")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    assert Settings.from_env().store_responses is stored


def test_openrouter_replays_tools_and_followups_without_remote_state(index):
    requests = []
    citation = reference(index.get_record("b" * 64 + ":39")).model_dump(mode="json")

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.host == "openrouter.ai"
        assert request.url.path == "/api/v1/responses"
        assert body["store"] is False
        assert body.get("previous_response_id") is None
        assert body["reasoning"] == {"effort": "low"}
        assert '"answer_type"' in body["input"][0]["content"]
        assert "without Markdown fences" in body["input"][0]["content"]
        search_tool = next(
            tool for tool in body["tools"] if tool["name"] == "search_documents"
        )
        assert search_tool["strict"] is False
        assert search_tool["parameters"]["required"] == ["query"]
        if len(requests) == 1:
            response = reply(
                calls=[
                    {
                        "type": "reasoning",
                        "id": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "PRIVATE REASONING"}
                        ],
                    },
                    call(arguments=search_args(), call_id="search"),
                ]
            )
        elif len(requests) == 2:
            outputs = [
                item
                for item in body["input"]
                if item.get("type") == "function_call_output"
            ]
            assert (
                json.loads(outputs[0]["output"])["data"]["results"][0]["record"]["boq"][
                    "quantity"
                ]
                == -2.5
            )
            response = reply(
                calls=[call("get_record", {"record_id": "b" * 64 + ":39"}, "full")]
            )
        else:
            assert "PRIVATE REASONING" not in json.dumps(body)
            assert (
                len(
                    [
                        item
                        for item in body["input"]
                        if item.get("type") == "function_call_output"
                    ]
                )
                == 2
            )
            response = reply(
                answer=answer(
                    answer_type="document",
                    sources=[citation],
                    text="The cited item supplies the requested value.",
                )
            )
        return httpx2.Response(200, json=response.model_dump(mode="json"))

    session = ConversationSession()
    with OpenAI(
        api_key="offline-unused",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
    ) as sdk:
        llm = LLM(
            Settings(
                api_key="offline-unused",
                model="minimax/minimax-m3.1-flash:free",
                store_responses=False,
                reasoning_effort="low",
            ),
            sdk,
        )
        first = run_agent("Item 2.3 quantity?", index, llm, session=session)
        second = run_agent("And its rate?", index, llm, session=session)
    assert first.status == second.status == AgentStatus.COMPLETED
    assert first.tool_calls == 2
    assert second.tool_calls == 0
    assert [
        item["content"] for item in requests[-1]["input"] if item.get("role") == "user"
    ] == ["Item 2.3 quantity?", "And its rate?"]
    assert session.context_bytes == len(encode(session.response_input).encode())
    assert "PRIVATE REASONING" not in encode(session.response_input)
    session.clear()
    assert not session.response_input
    assert not ConversationSession().response_input


def test_stateless_context_limit_applies_to_replayed_history(index, llm, client):
    llm.settings = llm.settings.model_copy(update={"store_responses": False})
    session = ConversationSession(
        response_input=[{"role": "user", "content": "x" * 4096}]
    )
    state = run_agent(
        "New question",
        index,
        llm,
        session=session,
        config=AgentConfig(max_session_context_bytes=4096),
    )
    assert state.stop_reason == StopReason.CONTEXT_LIMIT
    client.responses.create.assert_not_called()


def test_failed_stateless_turn_preserves_last_accepted_history(index, llm, client):
    llm.settings = llm.settings.model_copy(update={"store_responses": False})
    session = ConversationSession()
    client.responses.create.side_effect = [reply(answer=answer()), reply(raw="invalid")]
    run_agent("Accepted", index, llm, session=session)
    before = encode(session.response_input)
    failed = run_agent("Fails", index, llm, session=session)
    assert failed.status == AgentStatus.FAILED
    assert encode(session.response_input) == before
    assert len(session.history) == 1


@pytest.mark.parametrize("corrected", [True, False])
def test_stateless_citation_correction_is_bounded_and_still_validated(
    index, llm, client, corrected
):
    llm.settings = llm.settings.model_copy(update={"store_responses": False})
    good = reference(index.get_record("b" * 64 + ":39")).model_dump(mode="json")
    bad = {**good, "page": 999}
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer(answer_type="document", sources=[bad])),
        reply(
            answer=answer(answer_type="document", sources=[good if corrected else bad])
        ),
    ]
    session = ConversationSession()
    state = run_agent("Find this item", index, llm, session=session)
    assert state.status == (AgentStatus.COMPLETED if corrected else AgentStatus.FAILED)
    assert state.iterations == 3
    assert state.tool_calls == 1
    assert client.responses.create.call_args.kwargs["tools"] == []
    assert len(session.history) == (1 if corrected else 0)
    assert client.responses.create.call_count == 3
