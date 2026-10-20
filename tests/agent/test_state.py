import json

import pytest
from pydantic import ValidationError

from app.agents.agent import run_agent
from app.agents.config import AgentConfig
from app.agents.schemas import AgentStatus, StopReason
from app.agents.state import AgentState, ConversationSession

from .conftest import answer, call, reply, search_args
from .test_orchestrator import cited


def test_initial_state_and_configuration_do_not_share_mutable_defaults():
    first = AgentState(session_id="first", question="One")
    second = AgentState(session_id="second", question="Two")
    assert first.status == AgentStatus.INITIALIZED
    assert first.iterations == first.tool_calls == 0
    first.examined_record_ids.add("example")
    first.evidence["example"] = {"value": 1}
    assert not second.examined_record_ids
    assert not second.evidence
    sessions = [ConversationSession(), ConversationSession()]
    sessions[0].evidence["example"] = {"value": 1}
    assert sessions[0].session_id != sessions[1].session_id
    assert not sessions[1].evidence
    with pytest.raises(ValidationError):
        AgentConfig().max_iterations = 99


def test_state_transitions_are_ordered_and_terminal_states_cannot_restart(
    index, llm, client, monkeypatch
):
    transitions = []
    original = AgentState.transition

    def transition(self, status):
        transitions.append(status)
        original(self, status)

    monkeypatch.setattr(AgentState, "transition", transition)
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
    ]
    state = run_agent("Find an item", index, llm)
    assert transitions == [
        AgentStatus.RUNNING,
        AgentStatus.AWAITING_TOOL_RESULTS,
        AgentStatus.RUNNING,
        AgentStatus.COMPLETED,
    ]
    with pytest.raises(ValueError, match="finished"):
        state.transition(AgentStatus.RUNNING)
    fresh = AgentState(session_id="new", question="Question")
    with pytest.raises(ValueError, match="requires a running"):
        fresh.transition(AgentStatus.AWAITING_TOOL_RESULTS)


def test_follow_up_reuses_response_context_and_known_sources_without_extra_search(
    index, llm, client
):
    session = ConversationSession()
    first_final = reply(
        answer=answer(
            answer_type="document",
            sources=[cited(index)],
            text="Item 2.3 quantity is -2.5, boq.xlsx, Structural, row 38.",
        )
    )
    first_final.id = "accepted_first_answer"
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        first_final,
        reply(
            answer=answer(
                answer_type="document",
                sources=[cited(index)],
                text="The unit rate is recorded in the same cited BOQ item.",
            )
        ),
    ]
    first = run_agent("Find item 2.3 quantity", index, llm, session=session)
    before_bytes = session.context_bytes
    second = run_agent("And what is its unit rate?", index, llm, session=session)
    assert first.status == second.status == AgentStatus.COMPLETED
    assert second.iterations == 1
    assert second.tool_calls == 0
    request = client.responses.create.call_args.kwargs
    assert request["previous_response_id"] == "accepted_first_answer"
    assert request["input"][1] == {
        "role": "user",
        "content": "And what is its unit rate?",
    }
    assert len(session.history) == 2
    assert len(second.history) == 1
    assert second.history[0].question == "Find item 2.3 quantity"
    assert second.sources == first.sources
    assert session.context_bytes > before_bytes
    assert session.evidence["b" * 64 + ":39"]["boq"]["quantity"] == -2.5
    assert first.final_response is not None
    first.final_response.answer = "mutated externally"
    first.evidence["b" * 64 + ":39"]["boq"]["quantity"] = 999
    assert session.history[0].response.answer != "mutated externally"
    assert session.evidence["b" * 64 + ":39"]["boq"]["quantity"] == -2.5


def test_general_follow_up_is_allowed_in_a_document_conversation(index, llm, client):
    session = ConversationSession()
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
        reply(answer=answer(text="BOQ means bill of quantities.")),
    ]
    run_agent("Find item 2.3", index, llm, session=session)
    state = run_agent("What does BOQ stand for generally?", index, llm, session=session)
    assert state.status == AgentStatus.COMPLETED
    assert state.final_response is not None
    assert state.final_response.answer_type == "general"
    assert not state.final_response.sources


def test_other_sessions_and_cleared_sessions_cannot_cite_remembered_evidence(
    index, llm, client
):
    first = ConversationSession()
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
    ]
    run_agent("First session item", index, llm, session=first)
    second = ConversationSession()
    client.responses.create.side_effect = None
    client.responses.create.return_value = reply(
        answer=answer(answer_type="document", sources=[cited(index)])
    )
    failed = run_agent("What about its rate?", index, llm, session=second)
    assert failed.stop_reason == StopReason.INVALID_RESPONSE
    assert client.responses.create.call_args.kwargs["previous_response_id"] is None
    assert not second.sources
    old_id = first.session_id
    first.clear()
    assert first.session_id == old_id
    assert not first.history and not first.sources and not first.evidence
    assert not first.representation_hashes and not first.warnings
    assert first.context_bytes == 0
    assert first.previous_response_id is None
    cleared = run_agent("What about its rate?", index, llm, session=first)
    assert cleared.status == AgentStatus.FAILED
    assert client.responses.create.call_args.kwargs["previous_response_id"] is None


def test_failed_run_does_not_replace_last_accepted_session_context(index, llm, client):
    session = ConversationSession()
    accepted = reply(answer=answer())
    accepted.id = "accepted"
    client.responses.create.side_effect = [
        accepted,
        reply(calls=[call("list_documents")]),
        reply(raw="invalid answer"),
        reply(answer=answer()),
    ]
    run_agent("First", index, llm, session=session)
    failed = run_agent("Second fails", index, llm, session=session)
    assert failed.status == AgentStatus.FAILED
    assert session.previous_response_id == "accepted"
    assert len(session.history) == 1
    assert not session.sources
    third = run_agent("Third", index, llm, session=session)
    assert third.status == AgentStatus.COMPLETED
    assert (
        client.responses.create.call_args.kwargs["previous_response_id"] == "accepted"
    )
    assert [turn.question for turn in session.history] == ["First", "Third"]


def test_follow_up_preserves_incomplete_evidence_warning(index, llm, client):
    session = ConversationSession()
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args(query="excav", limit=1))]),
        reply(answer=answer(answer_type="insufficient")),
        reply(answer=answer(answer_type="insufficient")),
    ]
    run_agent("Total excavation", index, llm, session=session)
    second = run_agent("Can that establish the total?", index, llm, session=session)
    assert second.final_response is not None
    assert any("incomplete" in warning for warning in second.final_response.warnings)


def test_remembered_record_representation_can_be_referenced_in_later_turn(
    index, llm, client
):
    session = ConversationSession()
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
    ]
    run_agent("Find item", index, llm, session=session)
    state = run_agent("Recheck that item", index, llm, session=session)
    assert state.status == AgentStatus.COMPLETED
    output = json.loads(client.responses.create.call_args.kwargs["input"][1]["output"])
    record = output["data"]["results"][0]["record"]
    assert record["already_supplied"]
    assert record["representation_sha256"] in session.representation_hashes
