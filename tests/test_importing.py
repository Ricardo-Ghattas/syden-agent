import fcntl
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

from app.ingestion.importing import MAX_IMPORT_FILES, import_documents
from app.ingestion.pipeline import process_document
from app.retrieval import DocumentIndex, search_documents


def test_batch_imports_pdf_and_excel_into_explicit_collection(pdf, workbook, tmp_path):
    data, processed = tmp_path / "storage", tmp_path / "custom-collection"
    before = {path: path.read_bytes() for path in [pdf, workbook]}
    events = []
    batch = import_documents(
        [pdf, workbook],
        data,
        processed,
        on_progress=lambda pos, item: events.append((pos, item.status)),
    )
    assert [item.status for item in batch.items] == ["Ready", "Ready"]
    assert batch.error is None and batch.index is not None
    assert len(batch.index.documents) == 2
    assert search_documents(batch.index, "35 MPa").total
    assert search_documents(batch.index, "001", {"kind": "boq_item"}).total
    assert not (data / "processed").exists()
    assert len(list((data / "raw").iterdir())) == 2
    assert not list(processed.glob("*.tmp"))
    assert events == [(0, "Processing"), (0, "Ready"), (1, "Processing"), (1, "Ready")]
    assert all(path.read_bytes() == content for path, content in before.items())


def test_duplicates_by_content_and_updated_same_name_keep_provenance(
    workbook, tmp_path, monkeypatch
):
    data, processed = tmp_path / "data", tmp_path / "processed"
    original = import_documents([workbook], data, processed)
    first_bytes = workbook.read_bytes()
    renamed = tmp_path / "renamed.xlsx"
    renamed.write_bytes(first_bytes)
    processor = MagicMock(wraps=process_document)
    monkeypatch.setattr("app.ingestion.importing.process_document", processor)
    duplicate = import_documents([renamed, workbook], data, processed)
    assert [item.status for item in duplicate.items] == ["Duplicate", "Duplicate"]
    processor.assert_not_called()
    from openpyxl import load_workbook

    book = load_workbook(workbook)
    book.active["B3"] = "Updated concrete description"
    book.save(workbook)
    book.close()
    updated = import_documents([workbook], data, processed)
    assert updated.items[0].status == "Ready"
    assert len(updated.index.documents) == 2
    assert set(original.index.documents) < set(updated.index.documents)
    assert any(path.read_bytes() == first_bytes for path in (data / "raw").iterdir())
    assert len(list(processed.glob("*.jsonl"))) == 2


def test_validation_and_partial_success_continue(workbook, tmp_path):
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not a document")
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a PDF")
    paths = [
        unsupported,
        tmp_path / "missing.pdf",
        tmp_path,
        Path("\x00.pdf"),
        corrupt,
        workbook,
    ]
    batch = import_documents(paths, tmp_path / "data", tmp_path / "processed")
    assert [item.status for item in batch.items] == ["Failed"] * 5 + ["Ready"]
    assert len(batch.index.documents) == 1
    assert not list((tmp_path / "processed").glob("*.tmp"))


def test_extractor_failure_does_not_leak_and_remaining_files_continue(
    pdf, workbook, tmp_path, monkeypatch
):
    def processor(path, data_dir, **kwargs):
        if path.suffix == ".pdf":
            raise RuntimeError("PRIVATE provider-like payload")
        return process_document(path, data_dir, **kwargs)

    monkeypatch.setattr("app.ingestion.importing.process_document", processor)
    batch = import_documents([pdf, workbook], tmp_path / "data", tmp_path / "processed")
    assert [item.status for item in batch.items] == ["Failed", "Ready"]
    assert "PRIVATE" not in str(batch.items)


def test_cancellation_finishes_current_file_and_skips_remaining(
    pdf, workbook, tmp_path
):
    cancelled = Event()

    def progress(pos, item):
        if item.status == "Ready":
            cancelled.set()

    batch = import_documents(
        [pdf, workbook],
        tmp_path / "data",
        tmp_path / "processed",
        cancelled=cancelled,
        on_progress=progress,
    )
    assert [item.status for item in batch.items] == ["Ready", "Cancelled"]
    assert len(batch.index.documents) == 1
    assert not list((tmp_path / "processed").glob("*.tmp"))


def test_refresh_failure_does_not_publish_invalid_index(workbook, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "bad.jsonl").write_text("not JSON\n")
    batch = import_documents([workbook], tmp_path / "data", processed)
    assert batch.items[0].status == "Saved"
    assert batch.index is None and "index validation failed" in batch.error
    assert len(list(processed.glob("*.jsonl"))) == 2


def test_collection_lock_prevents_overlapping_writers(workbook, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    with (processed / ".syden-import.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = import_documents([workbook], tmp_path / "data", processed)
        assert "Another import" in batch.error
        assert batch.items[0].status == "Failed"
        assert not list(processed.glob("*.jsonl"))
    assert import_documents([workbook], tmp_path / "data", processed).index is not None


def test_batch_bound_is_checked_before_creating_storage(workbook, tmp_path):
    target = tmp_path / "unused"
    batch = import_documents(
        [workbook] * (MAX_IMPORT_FILES + 1), target, target / "processed"
    )
    assert batch.error and not batch.items
    assert not target.exists()


def test_cancellation_during_pipeline_finishes_atomic_output(
    workbook, pdf, tmp_path, monkeypatch
):
    from threading import Thread

    started, release, cancelled = Event(), Event(), Event()
    result = []

    def processor(path, data_dir, **kwargs):
        started.set()
        assert release.wait(5)
        return process_document(path, data_dir, **kwargs)

    monkeypatch.setattr("app.ingestion.importing.process_document", processor)
    worker = Thread(
        target=lambda: result.append(
            import_documents(
                [workbook, pdf],
                tmp_path / "data",
                tmp_path / "processed",
                cancelled=cancelled,
            )
        )
    )
    worker.start()
    try:
        assert started.wait(5)
        cancelled.set()
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert [item.status for item in result[0].items] == ["Ready", "Cancelled"]
        assert len(result[0].index.documents) == 1
        assert not list((tmp_path / "processed").glob("*.tmp"))
    finally:
        release.set()
        worker.join(5)


def test_index_constructor_failure_keeps_saved_files_without_leaking(
    workbook, tmp_path, monkeypatch
):
    calls = []

    def build(path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("SECRET storage payload")
        return DocumentIndex(path)

    monkeypatch.setattr("app.ingestion.importing.DocumentIndex", build)
    batch = import_documents([workbook], tmp_path / "data", tmp_path / "processed")
    assert batch.index is None
    assert batch.items[0].status == "Saved"
    assert batch.items[0].output.is_file()
    assert "SECRET" not in batch.error
