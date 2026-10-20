from threading import Event

from app.agents.agent import run_agent
from app.agents.schemas import StopReason
from app.agents.state import ConversationSession

from .conftest import answer, call, reply, search_args


def test_cancelled_before_request_never_calls_provider(index, llm, client):
    cancelled = Event()
    cancelled.set()
    state = run_agent("Question", index, llm, cancelled=cancelled)
    assert state.stop_reason == StopReason.CANCELLED
    client.responses.create.assert_not_called()


def test_cancelled_inflight_answer_never_changes_memory(index, llm, client):
    cancelled = Event()
    session = ConversationSession()
    before = session.model_dump_json()

    def respond(**kwargs):
        cancelled.set()
        return reply(answer=answer(answer_type="general", text="Unaccepted answer"))

    client.responses.create.side_effect = respond
    state = run_agent("Question", index, llm, session=session, cancelled=cancelled)
    assert state.stop_reason == StopReason.CANCELLED
    assert session.model_dump_json() == before


def test_activity_reports_real_tool_actions_and_no_private_payload(index, llm, client):
    events = []
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(answer=answer(answer_type="insufficient", text="No complete answer")),
    ]
    run_agent("PRIVATE QUESTION", index, llm, on_activity=events.append)
    assert events == [
        "llm_requested",
        "response_received",
        "search_documents",
        "llm_requested",
        "response_received",
        "preparing_answer",
    ]
