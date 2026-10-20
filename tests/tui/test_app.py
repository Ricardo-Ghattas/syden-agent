from threading import Event
from unittest.mock import MagicMock

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static

from app.agents.config import AgentConfig
from app.agents.llm import LLM, Settings
from app.agents.schemas import AgentResponse, AgentStatus, StopReason
from app.agents.state import ConversationTurn
from app.tui.app import SydenApp
from app.tui.widgets import AnswerView, Composer


@pytest.fixture
def app(tmp_path):
    llm = LLM(Settings(api_key="test-unused", model="test-model"), MagicMock())
    return SydenApp(llm, AgentConfig(), tmp_path)


async def idle(app, pilot):
    for _ in range(100):
        await pilot.pause(0.01)
        if not app.busy and not app.loading:
            return
    raise AssertionError("Worker did not finish")


def completed(text="**820.75 m³** — raft concrete."):
    return AgentResponse(answer=text, answer_type="general", sources=[], warnings=[])


def notices(app):
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    console = Console(file=stream, width=100, color_system=None)
    for widget in app.query(Static):
        console.print(widget.content)
    return stream.getvalue()


@pytest.mark.asyncio
async def test_submit_existing_agent_and_preserve_history(app, monkeypatch):
    calls = []

    def ask(question, index, llm, *, session, config, cancelled, on_activity):
        calls.append((question, len(session.history)))
        on_activity("search_documents")
        on_activity("response_received")
        result = completed()
        session.history.append(ConversationTurn(question=question, response=result))
        return result

    monkeypatch.setattr("app.tui.app.ask", ask)
    async with app.run_test() as pilot:
        await idle(app, pilot)
        composer = app.query_one(Composer)
        composer.load_text("Quantity?")
        await pilot.press("enter")
        await idle(app, pilot)
        composer.load_text("What about the rate?")
        await pilot.press("enter")
        await idle(app, pilot)
        assert calls == [("Quantity?", 0), ("What about the rate?", 1)]
        assert len(app.query(AnswerView)) == 2
        assert len(app.session.history) == 2
        assert app.focused is composer
        assert app.connection == "Response received"
        assert app.query_one(AnswerView).has_class("answer-message")
        assert "Quantity?" in notices(app)
        assert "No documents imported yet" in notices(app)


@pytest.mark.asyncio
async def test_multiline_fallbacks_and_editing(app):
    async with app.run_test() as pilot:
        await idle(app, pilot)
        await pilot.press("a", "ctrl+o", "b", "f4", "c")
        assert app.query_one(Composer).text == "a\nb\nc"
        await pilot.press("ctrl+a", "backspace")
        assert app.query_one(Composer).text == "a\nbc"


@pytest.mark.asyncio
async def test_clear_and_new_shortcuts_reset_memory_and_display(app):
    async with app.run_test() as pilot:
        await idle(app, pilot)
        original = app.session.session_id
        app.session.history.append(
            ConversationTurn(question="old", response=completed())
        )
        app.latest = completed()
        app.add_notice("old transcript")
        await pilot.press("ctrl+l")
        assert app.session.session_id == original
        assert not app.session.history and app.latest is None
        assert "old transcript" not in notices(app)
        await pilot.press("ctrl+n")
        assert app.session.session_id != original
        app.query_one(Composer).load_text("/help")
        await pilot.press("enter")
        assert "Ctrl+O" in notices(app)
        await pilot.press("ctrl+q")
        assert app.return_value is None


@pytest.mark.asyncio
async def test_cancel_and_new_session_while_request_blocked(app, monkeypatch):
    started, release = Event(), Event()
    calls = []

    def ask(question, index, llm, *, session, **kwargs):
        calls.append(question)
        started.set()
        assert release.wait(5)
        # Deliberately emulate a client that ignores cancellation and mutates its copy.
        session.history.append(
            ConversationTurn(question=question, response=completed())
        )
        return completed()

    monkeypatch.setattr("app.tui.app.ask", ask)
    try:
        async with app.run_test() as pilot:
            await idle(app, pilot)
            app.query_one(Composer).load_text("Slow question")
            await pilot.press("enter")
            assert started.is_set() and app.busy
            app.query_one(Composer).load_text("Duplicate")
            await pilot.press("enter")
            assert calls == ["Slow question"]
            await pilot.press("escape")
            assert app.cancelled.is_set()
            old_id = app.session.session_id
            await pilot.press("ctrl+n")
            assert app.session.session_id != old_id
            assert not app.session.history
            # All keyboard interactions above completed while the request was blocked.
            release.set()
            await idle(app, pilot)
            assert not app.session.history
            assert not app.query(AnswerView)
            assert "Request cancelled" in notices(app)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_safe_exit_waits_for_active_call_without_freezing(app, monkeypatch):
    release = Event()

    def ask(*args, **kwargs):
        assert release.wait(5)
        return completed()

    monkeypatch.setattr("app.tui.app.ask", ask)
    try:
        async with app.run_test() as pilot:
            await idle(app, pilot)
            app.query_one(Composer).load_text("Wait")
            await pilot.press("enter", "ctrl+q")
            assert app.exit_requested and app.cancelled.is_set()
            assert "Stopping" in notices(app)
            release.set()
            await pilot.pause()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_unexpected_error_is_redacted_and_retry_possible(app, monkeypatch):
    monkeypatch.setattr(
        "app.tui.app.ask", MagicMock(side_effect=RuntimeError("SECRET payload"))
    )
    async with app.run_test() as pilot:
        await idle(app, pilot)
        app.query_one(Composer).load_text("Question")
        await pilot.press("enter")
        await idle(app, pilot)
        assert "The request failed" in notices(app)
        assert "SECRET" not in notices(app)
        assert not app.session.history
        assert app.focused is app.query_one(Composer)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(32, 18), (80, 24)])
async def test_long_markdown_sources_warnings_resize_and_commands(
    app, monkeypatch, size
):
    from app.agents.schemas import SourceReference

    sources = [
        SourceReference(
            document_id="a",
            record_id=str(row),
            filename="Example.xlsx",
            sheet="Concrete",
            page=None,
            row_start=row,
            row_end=row,
            cell_range=None,
        )
        for row in [8, 9, 12]
    ]
    result = AgentResponse(
        answer="| Item | Quantity |\n| :--- | ---: |\n| Raft | 820.75 m³ |\n| Slab | 210.00 m³ |\n\n"
        + ("Long explanation. " * 100),
        answer_type="document",
        sources=sources,
        warnings=["Incomplete evidence"],
        status=AgentStatus.INSUFFICIENT_EVIDENCE,
        stop_reason=StopReason.INSUFFICIENT_EVIDENCE,
    )
    monkeypatch.setattr("app.tui.app.ask", MagicMock(return_value=result))
    async with app.run_test(size=size) as pilot:
        await idle(app, pilot)
        app.query_one(Composer).load_text("Compare")
        await pilot.press("enter")
        await idle(app, pilot)
        assert app.query_one(AnswerView).response is result
        assert app.query_one("#conversation", VerticalScroll).max_scroll_y > 0
        await pilot.resize_terminal(40, 20)
        await pilot.pause()
        history = app.query_one("#conversation", VerticalScroll)
        assert history.scroll_y == history.max_scroll_y
        app.query_one(Composer).load_text("/sources")
        await pilot.press("enter")
        assert "Sources" in notices(app)
        assert "rows 8–9, 12" in notices(app)
        assert "Incomplete evidence" in notices(app)
        app.query_one(Composer).load_text("/clear")
        await pilot.press("enter")
        assert app.latest is None
        assert not app.query(AnswerView)
        app.query_one(Composer).load_text("/new")
        await pilot.press("enter")
        app.query_one(Composer).load_text("/exit")
        await pilot.press("enter")


@pytest.mark.asyncio
async def test_real_agent_with_mocked_sdk_keeps_server_context(app):
    from tests.agent.conftest import answer, reply

    client = app.llm.client
    client.with_options.return_value = client
    client.responses.create.side_effect = [
        reply(
            answer=answer(answer_type="general", text="**35 MPa** is a strength."),
        ).model_copy(update={"id": "first"}),
        reply(
            answer=answer(answer_type="general", text="A concise follow-up."),
        ).model_copy(update={"id": "second"}),
    ]
    async with app.run_test() as pilot:
        await idle(app, pilot)
        for question in ["Explain strength", "Explain further"]:
            app.query_one(Composer).load_text(question)
            await pilot.press("enter")
            await idle(app, pilot)
        assert len(app.session.history) == 2
        assert app.session.previous_response_id == "second"
        calls = client.responses.create.call_args_list
        assert calls[1].kwargs["previous_response_id"] == "first"
        assert app.query(AnswerView).last().response.answer == "A concise follow-up."


@pytest.mark.asyncio
async def test_invalid_input_and_unknown_command_do_not_call_agent(app, monkeypatch):
    agent = MagicMock()
    monkeypatch.setattr("app.tui.app.ask", agent)
    async with app.run_test() as pilot:
        await idle(app, pilot)
        app.query_one(Composer).load_text("x" * 16001)
        await pilot.press("enter")
        assert "16000 characters" in notices(app)
        app.query_one(Composer).load_text("/unknown")
        await pilot.press("enter")
        assert "Unknown command" in notices(app)
        agent.assert_not_called()


@pytest.mark.asyncio
async def test_index_failure_is_redacted_and_does_not_crash(app, monkeypatch):
    monkeypatch.setattr(
        "app.tui.app.DocumentIndex",
        MagicMock(side_effect=OSError("SECRET path payload")),
    )
    async with app.run_test() as pilot:
        await idle(app, pilot)
        app.query_one(Composer).load_text("Question")
        await pilot.press("enter")
        assert "Cannot load this collection" in notices(app)
        assert "SECRET" not in notices(app)
        assert not app.busy
