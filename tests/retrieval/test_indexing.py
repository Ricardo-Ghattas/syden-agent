import json
from pathlib import Path

import pytest

from app.retrieval import DocumentIndex, search_documents
from app.retrieval.indexing import load_records

A = "a" * 64
B = "b" * 64


def test_load_both_formats_and_discover_existing_metadata(index):
    assert index.report.records == 52
    assert index.report.documents == 2
    assert not index.report.issues
    documents = index.list_documents()
    assert [(doc.filename, doc.format, doc.record_count) for doc in documents] == [
        ("specification.pdf", "pdf", 9),
        ("boq.xlsx", "xlsx", 43),
    ]
    assert documents[1].sheets == ["Civil", "Structural"]
    assert "1 Concrete" in documents[0].sections
    assert "Earthworks" in documents[1].sections


def test_invalid_lines_duplicates_and_unsupported_formats_are_reported(
    index, processed
):
    valid = index.get_record(f"{B}:39").model_dump(mode="json")
    unsupported = {
        **valid,
        "id": "unsupported",
        "source": {**valid["source"], "format": "docx"},
    }
    extra = index.get_record(f"{A}:3")
    extra.id = f"{A}:10"
    extra.order_index = 10
    extra.text = "Valid record after malformed lines"
    path = processed / "z-invalid.jsonl"
    path.write_bytes(
        b"bad json\n\xff\n[]\n"
        + json.dumps(unsupported).encode()
        + b"\n"
        + json.dumps(valid).encode()
        + b"\n"
        + extra.model_dump_json().encode()
        + b"\n"
    )
    index.reload()
    assert index.report.records == 53
    assert index.get_record(extra.id).text == extra.text
    assert [issue.code for issue in index.report.issues] == ["malformed_record"] * 4 + [
        "duplicate_id"
    ]
    assert [issue.line for issue in index.report.issues] == [1, 2, 3, 4, 5]
    response = search_documents(index, "2.3")
    assert response.results[0].record.boq is not None
    assert response.results[0].record.boq.item == "2.3"
    assert not response.index_complete
    assert len(response.index_issues) == 5


def test_duplicate_policy_is_deterministic_by_sorted_file_then_line(index, tmp_path):
    record = index.get_record(f"{B}:39")
    first = tmp_path / "a.jsonl"
    last = tmp_path / "z.jsonl"
    first.write_text(record.model_dump_json() + "\n")
    record.text = "A conflicting duplicate"
    last.write_text(record.model_dump_json() + "\n")
    loaded, issues = load_records([last, first])
    assert loaded[record.id].text != "A conflicting duplicate"
    assert issues[0].path == str(last)
    assert issues[0].record_id == record.id


def test_inconsistent_document_identity_is_rejected(index, processed):
    record = index.get_record(f"{B}:39")
    record.id = "new-id"
    record.source.filename = "other.xlsx"
    (processed / "z.jsonl").write_text(record.model_dump_json() + "\n")
    index.reload()
    assert index.report.records == 52
    assert index.report.issues[0].code == "inconsistent_document"


def test_missing_and_empty_inputs_are_distinguishable(tmp_path):
    index = DocumentIndex(tmp_path / "missing")
    assert index.report.issues[0].code == "missing_directory"
    assert not search_documents(index, "concrete").index_complete
    index = DocumentIndex(tmp_path)
    assert index.report.records == 0
    assert not index.report.issues
    (tmp_path / "empty.jsonl").write_text("\n \n")
    index.reload()
    assert index.report.files == 1
    assert index.report.records == 0
    assert search_documents(index, "concrete").total == 0
    loaded, issues = load_records([tmp_path / "gone.jsonl"])
    assert loaded == {}
    assert issues[0].code == "missing_file"


def test_read_errors_keep_other_documents(processed, monkeypatch):
    original = Path.open

    def denied(path, *args, **kwargs):
        if path.name == "boq.jsonl":
            raise PermissionError("access denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    index = DocumentIndex(processed)
    assert index.report.records == 9
    assert index.report.issues[0].code == "read_error"
    assert search_documents(index, "concrete").results


def test_reload_replaces_snapshot_and_removes_stale_records(index, processed):
    output = processed / "boq.jsonl"
    record = index.get_record(f"{B}:39")
    record.text = "Novel substructure specification"
    record.boq.description = "Novel substructure specification"
    output.write_text(record.model_dump_json() + "\n")
    assert search_documents(index, "novel").total == 0
    report = index.reload()
    assert report.records == 10
    assert search_documents(index, "novel").total == 1
    assert index.get_record(f"{B}:4") is None
    output.unlink()
    index.reload()
    assert len(index.list_documents()) == 1
    assert search_documents(index, "2.3").total == 0


def test_get_record_preserves_full_data_and_returns_independent_values(
    index, processed
):
    original = next(
        json.loads(line)
        for line in (processed / "boq.jsonl").read_text().splitlines()
        if json.loads(line)["id"] == f"{B}:39"
    )
    record = index.get_record(f"{B}:39")
    assert record.model_dump(mode="json") == original
    assert record.boq.quantity == -2.5
    assert record.boq.rate == "1234.567890"
    assert record.boq.amount == "-3086.419725"
    record.boq.rate = 0
    record.source.sheet = "Changed"
    record.cells.clear()
    assert index.get_record(f"{B}:39").model_dump(mode="json") == original
    assert index.get_record("absent") is None
    with pytest.raises(ValueError, match="non-empty"):
        index.get_record("")
    documents = index.list_documents()
    documents[0].sections.clear()
    assert index.list_documents()[0].sections
