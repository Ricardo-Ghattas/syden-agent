from threading import Event
from unittest.mock import MagicMock

import pytest

from app.agents.config import AgentConfig
from app.agents.llm import LLM, Settings
from app.agents.state import ConversationTurn
from app.retrieval import search_documents
from app.tui.app import SydenApp
from app.tui.import_dialog import DocumentTree, ImportDialog
from app.tui.widgets import Composer
from tests.tui.test_app import completed, notices


@pytest.fixture
def app(tmp_path):
    return SydenApp(
        None, AgentConfig(), tmp_path / "active", data_dir=tmp_path / "storage"
    )


async def idle(app, pilot):
    for _ in range(200):
        await pilot.pause(0.01)
        if (
            not app.busy
            and not app.loading
            and not app.importing
            and app.pending_import is None
        ):
            return
    raise AssertionError("Operation did not finish")


@pytest.mark.asyncio
async def test_add_quoted_path_and_documents_without_llm(app, workbook, tmp_path):
    spaced = tmp_path / "Project BOQ.xlsx"
    spaced.write_bytes(workbook.read_bytes())
    async with app.run_test() as pilot:
        await idle(app, pilot)
        assert "Press F3" in notices(app)
        app.query_one(Composer).load_text(f'/add "{spaced}"')
        await pilot.press("enter")
        await idle(app, pilot)
        assert len(app.index.documents) == 1
        assert search_documents(app.index, "001").total
        assert list(app.processed_dir.glob("*.jsonl"))
        assert not (app.data_dir / "processed").exists()
        app.query_one(Composer).load_text("/documents")
        await pilot.press("enter")
        assert "Project BOQ.xlsx" in notices(app)
        assert "XLSX" in notices(app) and "Ready" in notices(app)
        app.query_one(Composer).load_text(f'/add "{spaced}"')
        await pilot.press("enter")
        await idle(app, pilot)
        assert "1 duplicates skipped" in notices(app)
        app.query_one(Composer).load_text("Quantity?")
        await pilot.press("enter")
        assert "Chat is not configured" in notices(app)


@pytest.mark.asyncio
async def test_picker_keyboard_multi_select_confirm_and_cancel(
    app, workbook, pdf, tmp_path
):
    app.last_import_dir = tmp_path
    (tmp_path / "ignore.txt").write_text("unsupported")
    async with app.run_test(size=(60, 25)) as pilot:
        await idle(app, pilot)
        await pilot.press("f3")
        assert isinstance(app.screen, ImportDialog)
        tree = app.screen.query_one(DocumentTree)
        await tree.reload()
        # Select actual file nodes by keyboard; folders remain navigable with arrows.
        for _ in range(20):
            await pilot.press("down")
            node = tree.cursor_node
            if node and node.data and node.data.path == workbook:
                break
        assert tree.cursor_node.data.path == workbook
        await pilot.press("space")
        assert workbook.resolve() in app.screen.selected
        tree.focus()
        for _ in range(20):
            await pilot.press("down")
            node = tree.cursor_node
            if node and node.data and node.data.path == pdf:
                break
        assert tree.cursor_node.data.path == pdf
        await pilot.press("enter")
        assert len(app.screen.selected) == 2
        await pilot.press("f2")
        await idle(app, pilot)
        assert len(app.index.documents) == 2
        app.query_one(Composer).load_text("/add")
        await pilot.press("enter")
        assert isinstance(app.screen, ImportDialog)
        await pilot.press("escape")
        assert not isinstance(app.screen, ImportDialog)
        assert app.focused is app.query_one(Composer)


@pytest.mark.asyncio
async def test_bad_quoting_and_missing_path_report_without_crash(app, tmp_path):
    async with app.run_test() as pilot:
        await idle(app, pilot)
        await app.command('/add "unterminated')
        assert "Invalid path quoting" in notices(app)
        await app.command(f'/add "{tmp_path / "absent.pdf"}"')
        await idle(app, pilot)
        assert "1 failed" in notices(app)


@pytest.mark.asyncio
async def test_import_waits_for_active_agent_and_preserves_conversation(
    app, workbook, monkeypatch
):
    started, release = Event(), Event()
    app.llm = LLM(Settings(api_key="test-unused", model="test-model"), MagicMock())

    def ask(question, index, llm, *, session, **kwargs):
        started.set()
        assert release.wait(5)
        assert not index.documents
        session.history.append(
            ConversationTurn(question=question, response=completed())
        )
        return completed()

    monkeypatch.setattr("app.tui.app.ask", ask)
    try:
        async with app.run_test() as pilot:
            await idle(app, pilot)
            app.query_one(Composer).load_text("Explain concrete")
            await pilot.press("enter")
            assert started.is_set()
            await app.command(f'/add "{workbook}"')
            assert app.pending_import and not app.importing
            assert not app.processed_dir.exists()
            release.set()
            await idle(app, pilot)
            assert len(app.index.documents) == 1
            assert len(app.session.history) == 1
            assert "Import queued" in notices(app)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_refresh_failure_keeps_previous_index(app, workbook, monkeypatch):
    from app.ingestion.importing import ImportBatch, ImportItem

    async with app.run_test() as pilot:
        await idle(app, pilot)
        previous = app.index
        monkeypatch.setattr(
            "app.tui.app.import_documents",
            MagicMock(
                return_value=ImportBatch(
                    items=[ImportItem(workbook, "Ready")],
                    error="Index refresh failed; previous index retained.",
                )
            ),
        )
        await app.request_import([workbook])
        await idle(app, pilot)
        assert app.index is previous
        assert "Index refresh failed" in notices(app)


@pytest.mark.asyncio
async def test_import_worker_responsive_cancellation_and_duplicate_prevention(
    app, workbook, pdf, monkeypatch
):
    from app.ingestion.importing import import_documents

    release, started = Event(), Event()
    calls = []

    def blocking(paths, data_dir, processed_dir, **kwargs):
        calls.append(paths)
        started.set()
        assert release.wait(5)
        return import_documents(paths, data_dir, processed_dir, **kwargs)

    monkeypatch.setattr("app.tui.app.import_documents", blocking)
    try:
        async with app.run_test() as pilot:
            await idle(app, pilot)
            await app.request_import([workbook, pdf])
            await pilot.pause()
            assert started.is_set()
            await app.request_import([workbook])
            assert len(calls) == 1
            await pilot.press("escape")
            assert app.import_cancelled.is_set()
            release.set()
            await idle(app, pilot)
            assert len(app.index.documents) == 0
            assert "2 cancelled" in notices(app)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_referenced_record_changes_clear_context_but_added_documents_do_not(
    app, workbook, pdf
):
    import json

    async with app.run_test() as pilot:
        await idle(app, pilot)
        await app.request_import([workbook])
        await idle(app, pilot)
        record = next(iter(app.index.entries.values())).record
        app.session.evidence[record.id] = record.model_dump(mode="json")
        app.session.history.append(
            ConversationTurn(question="Old fact", response=completed())
        )
        original_session_id = app.session.session_id
        await app.request_import([pdf])
        await idle(app, pilot)
        assert len(app.session.history) == 1
        output = next(
            path
            for path in app.processed_dir.glob("*.jsonl")
            if record.document_id in path.name
        )
        records = [json.loads(line) for line in output.read_text().splitlines()]
        records[0]["text"] = "Changed externally"
        output.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        await app.request_import([pdf])
        await idle(app, pilot)
        assert not app.session.history
        assert app.session.session_id == original_session_id
        assert "context was cleared" in notices(app)


@pytest.mark.asyncio
async def test_picker_directory_input_and_invalid_path(app, tmp_path, workbook):
    from textual.widgets import Input

    app.last_import_dir = tmp_path
    async with app.run_test() as pilot:
        await idle(app, pilot)
        await pilot.press("f3")
        dialog = app.screen
        field = dialog.query_one(Input)
        field.focus()
        field.value = str(tmp_path / "absent")
        await pilot.press("enter")
        assert "Cannot open that path" in str(
            dialog.query_one("#selected-names").content
        )
        field.value = str(workbook)
        await pilot.press("enter")
        assert dialog.selected == [workbook.resolve()]
        await pilot.press("escape")
        assert not app.index.documents


@pytest.mark.asyncio
async def test_narrow_picker_keeps_tree_visible_and_keyboard_controls_work(
    app, tmp_path
):
    app.last_import_dir = tmp_path
    async with app.run_test(size=(32, 18)) as pilot:
        await idle(app, pilot)
        await pilot.press("f3")
        tree = app.screen.query_one(DocumentTree)
        assert tree.size.height > 0
        await pilot.press("shift+tab")
        from textual.widgets import Input

        assert isinstance(app.focused, Input)
        await pilot.press("escape")
        assert app.focused is app.query_one(Composer)


@pytest.mark.asyncio
async def test_agent_can_finish_while_picker_is_open(app, tmp_path, monkeypatch):
    started, release = Event(), Event()
    app.llm = LLM(Settings(api_key="test-unused", model="test-model"), MagicMock())
    app.last_import_dir = tmp_path

    def ask(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return completed()

    monkeypatch.setattr("app.tui.app.ask", ask)
    try:
        async with app.run_test() as pilot:
            await idle(app, pilot)
            app.query_one(Composer).load_text("Question")
            await pilot.press("enter")
            assert started.is_set()
            await pilot.press("f3")
            assert isinstance(app.screen, ImportDialog)
            release.set()
            await idle(app, pilot)
            assert isinstance(app.screen, ImportDialog)
            assert app.focused is app.screen.query_one(DocumentTree)
            await pilot.press("escape")
    finally:
        release.set()


@pytest.mark.asyncio
async def test_unknown_home_path_reports_failure_without_crashing(app):
    async with app.run_test() as pilot:
        await idle(app, pilot)
        await app.command("/add ~syden-user-that-does-not-exist/spec.pdf")
        await idle(app, pilot)
        assert "1 failed" in notices(app)


@pytest.mark.asyncio
async def test_selected_file_list_is_keyboard_scrollable(app, workbook, tmp_path):
    from textual.containers import VerticalScroll

    app.last_import_dir = tmp_path
    async with app.run_test(size=(50, 26)) as pilot:
        await idle(app, pilot)
        await pilot.press("f3")
        dialog = app.screen
        for number in range(6):
            copied = tmp_path / f"document-{number}.xlsx"
            copied.write_bytes(workbook.read_bytes())
            dialog.toggle(copied)
        await pilot.pause()
        await pilot.press("tab")
        selected = dialog.query_one("#selected-files", VerticalScroll)
        assert app.focused is selected
        assert selected.max_scroll_y > 0
        await pilot.press("down")
        await pilot.pause()
        assert selected.scroll_y > 0
        await pilot.press("escape")
