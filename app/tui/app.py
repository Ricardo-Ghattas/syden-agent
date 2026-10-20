"""Responsive Textual shell; all reasoning remains in app.agents.agent.ask."""

import shlex
from pathlib import Path
from threading import Event
from typing import ClassVar

from pydantic import ValidationError
from rich.console import Group
from rich.table import Table
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
from app.ingestion.importing import (
    MAX_IMPORT_FILES,
    ImportBatch,
    ImportItem,
    import_documents,
)
from app.retrieval import DocumentIndex

from .import_dialog import ImportDialog
from .widgets import AnswerView, Composer

HELP = """F3 or /add opens the file picker · /add "path with spaces.pdf" imports directly
/documents lists the active collection · Local importing makes no API requests
Enter sends · Ctrl+O or F4 inserts a newline
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


class ImportProgress(Message):
    def __init__(self, position: int, item: ImportItem):
        super().__init__()
        self.position, self.item = position, item


class Imported(Message):
    def __init__(self, batch: ImportBatch):
        super().__init__()
        self.batch = batch


class SydenApp(App):
    TITLE = "SYDEN"
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "theme.tcss"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("f3", "import_files", "Add files", priority=True),
        Binding("ctrl+q", "quit", "Exit", priority=True),
        Binding("ctrl+n", "new_session", "New", priority=True),
        Binding("ctrl+l", "clear_session", "Clear", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    def __init__(
        self,
        llm: LLM | None,
        config: AgentConfig,
        processed_dir: Path,
        *,
        debug: bool = False,
        data_dir: Path | None = None,
        startup_notice: str | None = None,
    ):
        super().__init__()
        self.llm, self.config = llm, config
        self.processed_dir, self.show_debug = processed_dir, debug
        self.data_dir = data_dir if data_dir is not None else processed_dir.parent
        self.startup_notice = startup_notice
        self.importing = False
        self.pending_import: list[Path] | None = None
        self.import_cancelled = Event()
        self.import_items: list[ImportItem] = []
        self.import_view: Static | None = None
        self.import_status: dict[str, str] = {}
        self.last_import_dir = Path.home()
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
        document_count = len(self.index.documents) if self.index is not None else 0
        collection = self.processed_dir.name or "Document collection"
        line = Text.assemble(
            ("SYDEN", "bold blue"),
            ("  /  ", "dim"),
            (collection, ""),
            (
                f"  ·  {document_count} document{'s' if document_count != 1 else ''}"
                "  ·  indexed"
                f"  ·  {self.connection}"
                if self.llm is not None
                else f"  ·  {document_count} document{'s' if document_count != 1 else ''}"
                "  ·  local import"
            ),
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

    async def on_loaded(self, message: Loaded) -> None:
        self.index = message.index
        self.loading = False
        self.update_identity()
        if self.exit_requested:
            self.exit()
            return
        if self.index is None:
            self.add_notice(
                "Cannot load this collection. Check --processed-dir and file permissions, then restart SYDEN."
            )
        elif not self.index.entries:
            self.add_notice(
                "No documents imported yet.\n\nPress F3 to add PDF or Excel files."
            )
        elif self.index.report.issues:
            self.add_notice(
                "Some document records could not be loaded. Evidence may be incomplete; check the processed collection."
            )
        if self.startup_notice:
            self.add_notice(self.startup_notice)
        self.add_notice(
            "Ask about your documents. F3 adds files; /help lists commands."
        )
        self.set_activity("Idle")
        await self.start_pending_import()

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
        if (
            self.busy
            or self.loading
            or self.importing
            or self.pending_import
            or self.exit_requested
        ):
            self.set_activity(
                "Wait for the current operation; Esc requests cancellation."
            )
            return
        if self.llm is None:
            self.add_notice(
                "Chat is not configured. Local import works with F3 or /add; configure the API and reopen SYDEN to ask questions."
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
            Static(Text("You"), classes="role user-role"),
            Static(Text(question), classes="message user-message"),
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
        if command == "/add" or command.startswith("/add "):
            try:
                paths = shlex.split(command)[1:]
            except ValueError:
                self.add_notice(
                    'Invalid path quoting. Use /add "path with spaces.pdf".'
                )
                return
            if paths:
                await self.request_import([Path(path) for path in paths])
            else:
                self.action_import_files()
        elif command == "/documents":
            self.show_documents()
        elif command == "/help":
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
                Static(Text("SYDEN"), classes="role agent-role"),
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
        await self.start_pending_import()

    def action_cancel(self) -> None:
        if isinstance(self.screen, ImportDialog):
            self.screen.dismiss(None)
            return
        if self.pending_import is not None:
            self.pending_import = None
            self.add_notice("Queued import cancelled.")
        if self.importing:
            self.import_cancelled.set()
            self.set_activity("Cancelling import… finishing the current file safely.")
            return
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
        self.import_view = None
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
        if self.busy or self.loading or self.importing:
            self.pending_import = None
            self.import_cancelled.set()
            self.exit_requested = True
            self.cancelled.set()
            self.set_activity("Stopping… waiting for the current call before closing.")
        else:
            self.exit()

    def on_unmount(self) -> None:
        self.cancelled.set()
        self.import_cancelled.set()

    def action_import_files(self) -> None:
        if isinstance(self.screen, ImportDialog):
            return
        if self.importing or self.pending_import or self.exit_requested:
            self.add_notice(
                "An import is already running or queued. Wait for it to finish."
            )
            return
        self.push_screen(ImportDialog(self.last_import_dir), self.request_import)

    async def request_import(self, paths: list[Path] | None) -> None:
        if not paths:
            self.query_one(Composer).focus()
            return
        if len(paths) > MAX_IMPORT_FILES:
            self.add_notice(f"Select at most {MAX_IMPORT_FILES} files per import.")
            return
        if self.importing or self.pending_import or self.exit_requested:
            self.add_notice(
                "An import is already running or queued. Wait for it to finish."
            )
            return
        self.pending_import = paths
        try:
            directory = paths[-1].expanduser().absolute().parent
            if directory.is_dir():
                self.last_import_dir = directory
        except (OSError, ValueError, RuntimeError):
            pass  # The import worker reports invalid paths per file.
        if self.busy or self.loading:
            self.add_notice(
                "Import queued; it will start after the current operation finishes."
            )
        else:
            await self.start_pending_import()
        self.query_one(Composer).focus()

    async def start_pending_import(self) -> None:
        if (
            self.pending_import is None
            or self.busy
            or self.loading
            or self.importing
            or self.exit_requested
        ):
            return
        paths, self.pending_import = self.pending_import, None
        self.importing = True
        self.import_cancelled = Event()
        self.import_items = [ImportItem(path) for path in paths]
        self.import_view = Static(classes="notice")
        await self.query_one("#conversation", VerticalScroll).mount(self.import_view)
        self.update_import_progress()
        self.import_files(paths, self.import_cancelled)

    def update_import_progress(self) -> None:
        table = Table.grid(padding=(0, 2), expand=False)
        table.add_column(overflow="fold")
        table.add_column(overflow="fold")
        for item in self.import_items:
            table.add_row(
                Text(item.path.name),
                Text(item.status + (f" — {item.message}" if item.message else "")),
            )
        if self.import_view is not None and self.import_view.is_mounted:
            self.import_view.update(
                Group(Text("Importing documents", style="bold blue"), table)
            )
        finished = sum(
            item.status not in {"Queued", "Processing"} for item in self.import_items
        )
        self.set_activity(f"Importing: {finished} of {len(self.import_items)} finished")
        self.call_after_refresh(self.query_one("#conversation", VerticalScroll).anchor)

    @work(thread=True, exit_on_error=False)
    def import_files(self, paths: list[Path], cancelled: Event) -> None:
        batch = import_documents(
            paths,
            self.data_dir,
            self.processed_dir,
            cancelled=cancelled,
            on_progress=lambda position, item: self.post_message(
                ImportProgress(position, item)
            ),
        )
        self.post_message(Imported(batch))

    def on_import_progress(self, message: ImportProgress) -> None:
        self.import_items[message.position] = message.item
        self.update_import_progress()

    def on_imported(self, message: Imported) -> None:
        batch = message.batch
        self.importing = False
        self.import_items = batch.items
        self.update_import_progress()
        if batch.index is not None:
            if self.index is not None and any(
                record_id not in self.index.entries
                or record_id not in batch.index.entries
                or self.index.entries[record_id].record
                != batch.index.entries[record_id].record
                for record_id in self.session.evidence
            ):
                self.session.clear()
                self.latest = None
                self.add_notice(
                    "Previously referenced evidence changed or disappeared. Model context was cleared; displayed history was retained."
                )
            self.index = batch.index
            self.update_identity()
            self.import_status.update(
                {
                    item.document_id: "Ready"
                    for item in batch.items
                    if item.document_id and item.status in {"Ready", "Duplicate"}
                }
            )
        if batch.error:
            self.add_notice(batch.error)
        if batch.index is not None and any(
            document.warnings for document in batch.index.list_documents()
        ):
            self.add_notice(
                "The collection includes extraction warnings; /documents shows the details."
            )
        counts = {
            status: sum(item.status == status for item in batch.items)
            for status in ["Ready", "Saved", "Duplicate", "Failed", "Cancelled"]
        }
        total = len(self.index.documents) if self.index is not None else 0
        self.add_notice(
            f"Import complete: {counts['Ready'] + counts['Saved']} imported, {counts['Duplicate']} duplicates skipped, {counts['Failed']} failed, {counts['Cancelled']} cancelled. {total} documents currently searchable."
        )
        self.set_activity("Idle")
        self.query_one(Composer).focus()
        if self.exit_requested:
            self.exit()

    def show_documents(self) -> None:
        if self.index is None or not self.index.documents:
            self.add_notice(
                "No documents imported yet.\nPress F3 to add PDF or Excel files."
            )
            return
        table = Table(box=None, padding=(0, 1), show_edge=False)
        for heading in ["Filename", "Type", "Records", "Status"]:
            table.add_column(heading, overflow="fold")
        warnings = []
        for document in self.index.list_documents():
            status = self.import_status.get(document.document_id, "Indexed")
            table.add_row(
                Text(document.filename),
                document.format.upper(),
                str(document.record_count),
                status,
            )
            if document.warnings:
                warnings.append(
                    Text(document.filename + ": " + ", ".join(document.warnings))
                )
        history = self.query_one("#conversation", VerticalScroll)
        history.mount(
            Static(
                Group(Text("Documents", style="bold blue"), table, *warnings),
                classes="notice",
            )
        )
        self.call_after_refresh(history.anchor)
