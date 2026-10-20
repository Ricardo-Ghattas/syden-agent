"""Responsive Textual shell; all reasoning remains in app.agents.agent.ask."""

from pathlib import Path
from threading import Event
from typing import ClassVar

from pydantic import ValidationError
from rich.console import Group
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Static

from app.agents.agent import ask
from app.agents.config import AgentConfig
from app.agents.llm import LLM
from app.agents.rendering import THEME, sources_renderable
from app.agents.schemas import AgentError, AgentRequest, AgentResponse, AgentStatus
from app.agents.state import ConversationSession
from app.retrieval import DocumentIndex

from .widgets import AnswerView, Composer

HELP = """Enter sends · Ctrl+O or F4 inserts a newline
Ctrl+N new session · Ctrl+L clear context and display · Esc cancel
Ctrl+Q exit · Tab focuses history; PageUp/PageDown scroll
Select text with the mouse; terminal Shift+selection is also available.
/help commands · /new new session · /clear clear context
/sources latest references · /exit quit
Memory is in-process only. /clear keeps the session ID; /new changes it.
Cancellation waits for any current API call, then discards the unfinished turn."""
ACTIVITIES = {
    "llm_requested": "Thinking…",
    "search_documents": "Searching documents…",
    "list_documents": "Finding documents…",
    "get_record": "Reading BOQ records…",
    "preparing_answer": "Preparing answer…",
    "response_received": "Processing response…",
}


class Activity(Message):
    def __init__(self, name: str):
        super().__init__()
        self.name = name


class Loaded(Message):
    def __init__(self, index: DocumentIndex | None):
        super().__init__()
        self.index = index


class Completed(Message):
    def __init__(
        self,
        session: ConversationSession,
        response: AgentResponse | None,
        error: str | None = None,
    ):
        super().__init__()
        self.session, self.response, self.error = session, response, error


class SydenApp(App):
    TITLE = "SYDEN"
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "theme.tcss"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit", "Exit", priority=True),
        Binding("ctrl+n", "new_session", "New", priority=True),
        Binding("ctrl+l", "clear_session", "Clear", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    def __init__(
        self, llm: LLM, config: AgentConfig, processed_dir: Path, *, debug: bool = False
    ):
        super().__init__()
        self.llm, self.config = llm, config
        self.processed_dir, self.show_debug = processed_dir, debug
        self.session = ConversationSession()
        self.index = None
        self.latest = None
        self.busy = False
        self.loading = True
        self.cancelled = Event()
        self.exit_requested = False
        self.connection = "Not checked"
        self.console.push_theme(THEME)

    def compose(self) -> ComposeResult:
        yield Static(id="identity", markup=False)
        yield Static(Text(str(self.processed_dir)), id="collection")
        yield VerticalScroll(id="conversation", can_focus=True)
        yield Static(Text("Loading documents…"), id="activity")
        yield Composer(id="composer", soft_wrap=True, show_line_numbers=False)
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "textual-dark"
        self.update_identity()
        self.query_one(Composer).focus()
        self.load_index(self.processed_dir)

    def update_identity(self) -> None:
        line = Text.assemble(
            ("SYDEN", "bold blue"), f"  {self.llm.settings.model}  ·  {self.connection}"
        )
        if self.show_debug:
            line.append(f"  ·  {self.session.session_id}", style="dim")
        self.query_one("#identity", Static).update(line)

    def set_activity(self, text: str) -> None:
        self.query_one("#activity", Static).update(Text(text))

    def add_notice(self, text: str) -> None:
        history = self.query_one("#conversation", VerticalScroll)
        history.mount(Static(Text(text), classes="notice"))
        self.call_after_refresh(history.anchor)

    @work(thread=True, exit_on_error=False)
    def load_index(self, path: Path) -> None:
        try:
            index = DocumentIndex(path)
        except Exception:  # noqa: BLE001 - UI boundary must redact unexpected error payloads.
            self.post_message(Loaded(None))
        else:
            self.post_message(Loaded(index))

    def on_loaded(self, message: Loaded) -> None:
        self.index = message.index
        self.loading = False
        if self.exit_requested:
            self.exit()
            return
        if self.index is None:
            self.add_notice(
                "Cannot load this collection. Check --processed-dir and file permissions, then restart SYDEN."
            )
        elif not self.index.entries:
            self.add_notice(
                "No documents loaded. Run syden ingest FILE, then reopen this collection. General questions are still available."
            )
        elif self.index.report.issues:
            self.add_notice(
                "Some document records could not be loaded. Evidence may be incomplete; check the processed collection."
            )
        self.add_notice("Ask about your documents. /help lists commands and shortcuts.")
        self.set_activity("Idle")

    @on(Composer.Submitted)
    async def submit(self) -> None:
        composer = self.query_one(Composer)
        question = composer.text.strip()
        if not question:
            return
        if question.startswith("/"):
            composer.clear()
            await self.command(question)
            return
        if self.busy or self.loading or self.exit_requested:
            self.set_activity(
                "Wait for the current operation; Esc requests cancellation."
            )
            return
        if self.index is None:
            self.add_notice(
                "No readable collection. Restart with --processed-dir PATH."
            )
            return
        try:
            AgentRequest(question=question)
        except ValidationError:
            self.add_notice("Use a question between 1 and 16000 characters.")
            return
        composer.clear()
        history = self.query_one("#conversation", VerticalScroll)
        await history.mount(
            Static(Text("You"), classes="role"),
            Static(Text(question), classes="message"),
        )
        self.call_after_refresh(history.anchor)
        self.busy = True
        self.cancelled = Event()
        self.set_activity("Thinking…")
        # The worker owns a deep copy. Only an accepted result is committed on the UI thread.
        self.execute(
            question, self.session.model_copy(deep=True), self.cancelled, self.index
        )

    async def command(self, command: str) -> None:
        if command == "/help":
            self.add_notice(HELP)
        elif command == "/sources":
            if self.latest and self.latest.sources:
                history = self.query_one("#conversation", VerticalScroll)
                await history.mount(
                    Static(
                        Group(
                            Text("Sources", style="bold blue"),
                            sources_renderable(self.latest.sources),
                        ),
                        classes="notice",
                    )
                )
                self.call_after_refresh(history.anchor)
            else:
                self.add_notice("No sources in the latest answer.")
        elif command == "/new":
            await self.action_new_session()
        elif command == "/clear":
            await self.action_clear_session()
        elif command == "/exit":
            self.action_quit()
        else:
            self.add_notice("Unknown command. Use /help.")

    @work(thread=True, exit_on_error=False)
    def execute(
        self,
        question: str,
        session: ConversationSession,
        cancelled: Event,
        index: DocumentIndex,
    ) -> None:
        try:
            response = ask(
                question,
                index,
                self.llm,
                session=session,
                config=self.config,
                cancelled=cancelled,
                on_activity=lambda name: self.post_message(Activity(name)),
            )
        except AgentError as error:
            self.post_message(Completed(session, None, str(error)))
        except Exception:  # noqa: BLE001 - UI boundary must redact unexpected error payloads.
            self.post_message(
                Completed(
                    session,
                    None,
                    "The request failed. Check provider configuration and collection access, then retry.",
                )
            )
        else:
            self.post_message(Completed(session, response))

    def on_activity(self, message: Activity) -> None:
        if self.show_debug:
            self.add_notice(f"Event: {message.name}")
        if message.name == "response_received":
            self.connection = "Response received"
            self.update_identity()
        if not self.cancelled.is_set():
            self.set_activity(ACTIVITIES.get(message.name, "Working…"))

    async def on_completed(self, message: Completed) -> None:
        self.busy = False
        if self.exit_requested:
            self.exit()
            return
        history = self.query_one("#conversation", VerticalScroll)
        if self.cancelled.is_set():
            self.add_notice(
                "Request cancelled. The unfinished turn was not added to conversation memory."
            )
        elif message.error:
            self.add_notice(message.error)
        elif message.response is not None:
            self.session = message.session
            self.latest = message.response
            await history.mount(
                Static(Text("SYDEN"), classes="role"),
                AnswerView(message.response, debug=self.show_debug),
            )
            if message.response.status in {
                AgentStatus.FAILED,
                AgentStatus.LIMIT_REACHED,
            }:
                self.add_notice(
                    "Try a narrower question or /clear to reset context. For API errors, check configuration or retry later."
                )
        self.call_after_refresh(history.anchor)
        self.set_activity("Idle")
        self.query_one(Composer).focus()

    def action_cancel(self) -> None:
        if self.busy and not self.cancelled.is_set():
            self.cancelled.set()
            self.set_activity("Cancelling… waiting for the current API/tool call.")
        else:
            self.query_one(Composer).focus()

    async def reset(self, *, new: bool) -> None:
        if self.exit_requested:
            return
        self.action_cancel()
        if new:
            self.session = ConversationSession()
        else:
            self.session.clear()
        self.latest = None
        await self.query_one("#conversation", VerticalScroll).remove_children()
        self.query_one(Composer).clear()
        self.add_notice(
            "New conversation."
            if new
            else "Conversation cleared; model context and displayed messages were removed."
        )
        self.update_identity()
        self.query_one(Composer).focus()

    async def action_new_session(self) -> None:
        await self.reset(new=True)

    async def action_clear_session(self) -> None:
        await self.reset(new=False)

    def action_quit(self) -> None:
        if self.busy or self.loading:
            self.exit_requested = True
            self.cancelled.set()
            self.set_activity("Stopping… waiting for the current call before closing.")
        else:
            self.exit()

    def on_unmount(self) -> None:
        self.cancelled.set()
