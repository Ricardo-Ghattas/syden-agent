from time import monotonic

from app.agents.agent import run_agent
from app.agents.config import AgentConfig
from app.agents.schemas import AgentError, AgentStatus, StopReason
from app.agents.state import ConversationSession
from app.tools.registry import ToolRegistry

from .conftest import answer, call, reply, search_args


def test_last_iteration_does_not_execute_tools_with_no_request_budget_left(
    index, llm, client
):
    client.responses.create.return_value = reply(calls=[call("list_documents")])
    state = run_agent(
        "Find documents", index, llm, config=AgentConfig(max_iterations=1)
    )
    assert state.status == AgentStatus.LIMIT_REACHED
    assert state.stop_reason == StopReason.MAX_ITERATIONS
    assert state.iterations == 1
    assert state.tool_calls == 0


def test_iteration_limit_keeps_evidence_and_does_not_commit_unfinished_session(
    index, llm, client
):
    session = ConversationSession()
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(calls=[call("list_documents")]),
    ]
    state = run_agent(
        "Find evidence",
        index,
        llm,
        session=session,
        config=AgentConfig(max_iterations=2),
    )
    assert state.stop_reason == StopReason.MAX_ITERATIONS
    assert state.tool_calls == 1
    assert state.sources
    assert state.evidence
    assert state.final_response is not None
    assert state.final_response.answer_type == "insufficient"
    assert session.previous_response_id is None
    assert session.history == []
    assert client.responses.create.call_count == 2


def test_global_tool_budget_applies_across_rounds(index, llm, client):
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        reply(
            calls=[
                call("list_documents"),
                call("get_record", {"record_id": "missing"}, "other"),
            ]
        ),
    ]
    state = run_agent(
        "Many operations", index, llm, config=AgentConfig(max_tool_calls=2)
    )
    assert state.stop_reason == StopReason.MAX_TOOL_CALLS
    assert state.tool_calls == 1
    assert len(state.outcomes) == 1


def test_equivalent_repeated_queries_stop_after_configured_no_progress_count(
    index, llm, client
):
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args(query=value), call_id=str(number))])
        for number, value in enumerate(["2.3", "  2.3  ", "2.3"])
    ]
    state = run_agent(
        "Repeat item lookup", index, llm, config=AgentConfig(repeated_call_limit=2)
    )
    assert state.stop_reason == StopReason.REPEATED_CALLS
    assert state.tool_calls == 3
    assert state.iterations == 3
    assert len({outcome.signature for outcome in state.outcomes}) == 1


def test_changed_pagination_is_not_a_repeated_call(index, llm, client):
    client.responses.create.side_effect = [
        reply(
            calls=[
                call(
                    arguments=search_args(query="excav", offset=offset, limit=1),
                    call_id=str(offset),
                )
            ]
        )
        for offset in [0, 1, 2]
    ] + [
        reply(
            answer=answer(
                answer_type="insufficient",
                text="Only three of the matching rows were inspected.",
            )
        )
    ]
    state = run_agent(
        "Inspect separate pages", index, llm, config=AgentConfig(repeated_call_limit=1)
    )
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert state.tool_calls == 3
    assert len(state.examined_record_ids) == 3


def test_changed_record_contents_reset_repetition_detector(index, llm, client):
    count = 0

    def respond(**kwargs):
        nonlocal count
        count += 1
        if count <= 3:
            record = index.entries["b" * 64 + ":39"].record
            assert record.boq is not None
            record.boq.quantity = count
            return reply(calls=[call(arguments=search_args(), call_id=str(count))])
        return reply(
            answer=answer(
                answer_type="insufficient",
                text="The observed values changed; reconcile the source.",
            )
        )

    client.responses.create.side_effect = respond
    state = run_agent(
        "Check updated data", index, llm, config=AgentConfig(repeated_call_limit=1)
    )
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert state.tool_calls == 3
    assert state.evidence["b" * 64 + ":39"]["boq"]["quantity"] == 3


def test_tool_failure_recovery_is_bounded(index, llm, client):
    client.responses.create.side_effect = [
        reply(calls=[call(name, call_id=str(number))])
        for number, name in enumerate(["bad_a", "bad_b", "bad_c"])
    ]
    state = run_agent(
        "Try invalid tools", index, llm, config=AgentConfig(max_tool_failures=2)
    )
    assert state.stop_reason == StopReason.TOOL_FAILURES
    assert state.tool_calls == 2
    assert not any(outcome.success for outcome in state.outcomes)


def test_nonretryable_api_error_stops_immediately(index, llm, client):
    client.responses.create.side_effect = AgentError("Authentication failed")
    state = run_agent("Question", index, llm)
    assert state.status == AgentStatus.FAILED
    assert state.stop_reason == StopReason.API_ERROR
    assert state.iterations == 1


def test_retryable_api_error_recovers_and_counts_requests(
    index, llm, client, monkeypatch
):
    monkeypatch.setattr("app.agents.agent.sleep", lambda delay: None)
    client.responses.create.side_effect = [
        AgentError("Temporary issue", retryable=True),
        reply(answer=answer()),
    ]
    state = run_agent("General explanation", index, llm)
    assert state.status == AgentStatus.COMPLETED
    assert state.iterations == 2
    assert state.tool_calls == 0
    options = client.with_options.call_args_list
    assert all(item.kwargs["max_retries"] == 0 for item in options)
    assert all(0 < item.kwargs["timeout"] <= llm.settings.timeout for item in options)


def test_api_recovery_exhaustion_does_not_loop(index, llm, client, monkeypatch):
    monkeypatch.setattr("app.agents.agent.sleep", lambda delay: None)
    client.responses.create.side_effect = AgentError("Temporary issue", retryable=True)
    state = run_agent("Question", index, llm)
    assert state.status == AgentStatus.FAILED
    assert state.iterations == 2  # One configured retry, each counted as an iteration.
    assert client.responses.create.call_count == 2


def test_recovery_counts_against_iteration_budget(index, llm, client):
    client.responses.create.side_effect = AgentError("Temporary issue", retryable=True)
    state = run_agent("Question", index, llm, config=AgentConfig(max_iterations=1))
    assert state.status == AgentStatus.FAILED
    assert state.iterations == 1
    assert client.responses.create.call_count == 1


def test_api_timeout_rejects_late_answer(index, llm, client, monkeypatch):
    now = [monotonic()]
    monkeypatch.setattr("app.agents.agent.monotonic", lambda: now[0])

    def respond(**kwargs):
        now[0] += 2
        return reply(answer=answer())

    client.responses.create.side_effect = respond
    state = run_agent("Slow answer", index, llm, config=AgentConfig(timeout_seconds=1))
    assert state.stop_reason == StopReason.TIMEOUT
    assert state.status == AgentStatus.LIMIT_REACHED
    assert state.iterations == 1
    assert state.tool_calls == 0


def test_timeout_between_tools_stops_the_rest_of_the_batch(
    index, llm, client, monkeypatch
):
    now = [monotonic()]
    monkeypatch.setattr("app.agents.agent.monotonic", lambda: now[0])
    original = ToolRegistry.execute

    def execute(self, name, arguments):
        output = original(self, name, arguments)
        now[0] += 2
        return output

    monkeypatch.setattr(ToolRegistry, "execute", execute)
    client.responses.create.return_value = reply(
        calls=[call(arguments=search_args()), call("list_documents", call_id="other")]
    )
    state = run_agent("Slow tools", index, llm, config=AgentConfig(timeout_seconds=1))
    assert state.stop_reason == StopReason.TIMEOUT
    assert state.tool_calls == 1
    assert state.evidence
    assert client.responses.create.call_count == 1


def test_keyboard_interrupt_is_a_controlled_stop(index, llm, client):
    client.responses.create.side_effect = KeyboardInterrupt()
    state = run_agent("Question", index, llm)
    assert state.status == AgentStatus.CANCELLED
    assert state.stop_reason == StopReason.CANCELLED


def test_session_turn_limit_makes_no_api_request(index, llm, client):
    session = ConversationSession()
    config = AgentConfig(max_session_turns=1)
    client.responses.create.return_value = reply(answer=answer())
    run_agent("First", index, llm, session=session, config=config)
    result = run_agent("Second", index, llm, session=session, config=config)
    assert result.stop_reason == StopReason.SESSION_LIMIT
    assert result.iterations == 0
    assert client.responses.create.call_count == 1


def test_context_limit_keeps_latest_question_and_does_not_call_provider(
    index, llm, client
):
    session = ConversationSession(context_bytes=4090)
    state = run_agent(
        "Latest question",
        index,
        llm,
        session=session,
        config=AgentConfig(max_session_context_bytes=4096),
    )
    assert state.question == "Latest question"
    assert state.stop_reason == StopReason.CONTEXT_LIMIT
    client.responses.create.assert_not_called()


def test_api_continuation_retry_does_not_execute_tools_again(
    index, llm, client, monkeypatch
):
    from .test_orchestrator import cited

    monkeypatch.setattr("app.agents.agent.sleep", lambda duration: None)
    executed = []
    original = ToolRegistry.execute

    def execute(self, name, arguments):
        executed.append(name)
        return original(self, name, arguments)

    monkeypatch.setattr(ToolRegistry, "execute", execute)
    client.responses.create.side_effect = [
        reply(calls=[call(arguments=search_args())]),
        AgentError("Temporary continuation failure", retryable=True),
        reply(answer=answer(answer_type="document", sources=[cited(index)])),
    ]
    state = run_agent("Find item 2.3", index, llm)
    assert state.status == AgentStatus.COMPLETED
    assert state.iterations == 3
    assert state.tool_calls == 1
    assert executed == ["search_documents"]
    failed, retried = [
        item.kwargs for item in client.responses.create.call_args_list[1:]
    ]
    assert failed["previous_response_id"] == retried["previous_response_id"]
    assert failed["input"] == retried["input"]
