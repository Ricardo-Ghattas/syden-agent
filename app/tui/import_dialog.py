"""Local, lazy directory browsing with explicit multi-file selection."""

from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Static

from app.ingestion.importing import MAX_IMPORT_FILES, SUPPORTED_SUFFIXES


class DocumentTree(DirectoryTree):
    ICON_FILE = ""
    ICON_FOLDER = ""
    ICON_FOLDER_OPEN = ""
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("space", "select_cursor", "Select", show=False)
    ]

    def filter_paths(self, paths):
        for path in paths:
            try:
                if path.is_dir() or path.suffix.casefold() in SUPPORTED_SUFFIXES:
                    yield path
            except OSError:
                continue


class ImportDialog(ModalScreen[list[Path] | None]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("f2", "confirm", "Import selected", priority=True),
    ]

    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory
        self.selected: list[Path] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="import-dialog"):
            yield Static(Text("Add documents", style="bold blue"))
            yield Static(Text("Local import · No API calls"))
            yield Input(
                str(self.directory),
                placeholder="Directory or file path; Enter to browse",
                id="browse-path",
            )
            yield DocumentTree(self.directory, id="file-tree")
            with VerticalScroll(id="selected-files", can_focus=True):
                yield Static(Text("No files selected"), id="selected-names")
            yield Static(
                Text("Arrows browse · Space selects · F2 imports · Esc cancels")
            )
            with Horizontal(id="import-buttons"):
                yield Button("Import", id="confirm-import")
                yield Button("Cancel", id="cancel-import")

    def on_mount(self) -> None:
        self.query_one(DocumentTree).focus()

    def show_selection(self, error: str | None = None) -> None:
        text = error or (
            f"Selected ({len(self.selected)}):\n"
            + "\n".join(str(path) for path in self.selected)
            if self.selected
            else "No files selected"
        )
        self.query_one("#selected-names", Static).update(Text(text))

    def toggle(self, path: Path) -> None:
        try:
            path = path.expanduser().resolve(strict=True)
            if not path.is_file() or path.suffix.casefold() not in SUPPORTED_SUFFIXES:
                self.show_selection("Select a PDF or XLSX file.")
                return
        except (OSError, ValueError, RuntimeError):
            self.show_selection("Cannot access that file.")
            return
        if path in self.selected:
            self.selected.remove(path)
        elif len(self.selected) < MAX_IMPORT_FILES:
            self.selected.append(path)
        else:
            self.show_selection(f"Select at most {MAX_IMPORT_FILES} files per import.")
            return
        self.show_selection()

    @on(DirectoryTree.FileSelected)
    def select_file(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        self.toggle(event.path)

    @on(Input.Submitted, "#browse-path")
    def browse(self, event: Input.Submitted) -> None:
        try:
            path = Path(event.value).expanduser().resolve(strict=True)
            if path.is_dir():
                self.directory = path
                self.query_one(DocumentTree).path = path
                self.query_one(DocumentTree).focus()
            else:
                self.toggle(path)
        except (OSError, ValueError, RuntimeError):
            self.show_selection(
                "Cannot open that path; check the directory and permissions."
            )

    def action_confirm(self) -> None:
        if self.selected:
            self.dismiss(list(self.selected))
        else:
            self.show_selection("Select at least one PDF or XLSX file.")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-import":
            self.action_confirm()
        else:
            self.action_cancel()
