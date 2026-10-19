import json
import subprocess
import sys

from app.ingestion.pipeline import process_document
from app.retrieval import DocumentIndex, search_documents


def test_existing_pdf_and_excel_pipeline_outputs_are_retrievable(
    pdf, workbook, tmp_path
):
    data_dir = tmp_path / "data"
    originals = {path: path.read_bytes() for path in (pdf, workbook)}
    outputs = [process_document(path, data_dir) for path in (pdf, workbook)]
    expected = {
        record["id"]: record
        for path in outputs
        for record in map(json.loads, path.read_text().splitlines())
    }
    index = DocumentIndex(data_dir / "processed")
    assert not index.report.issues
    assert set(index.entries) == set(expected)
    result = search_documents(index, "35 MPa", {"format": "pdf"}).results[0]
    assert result.record.source.page == 1
    assert result.record.model_dump(mode="json") == expected[result.record_id]
    item = search_documents(
        index, "001", {"kind": "boq_item", "format": "xlsx"}
    ).results[0]
    assert item.record.boq.item == "001"
    assert item.record.boq.quantity == 0
    assert item.record.boq.rate == 12.5
    assert item.record.boq.amount == "=D3*E3"
    assert item.record.source.sheet == "BOQ"
    assert item.record.source.row_start == 3
    assert item.record.model_dump(mode="json") == expected[item.record_id]
    for record_id, original in expected.items():
        assert index.get_record(record_id).model_dump(mode="json") == original
    for path, contents in originals.items():
        assert path.read_bytes() == contents


def test_retrieval_cli_reports_scores_full_sources_and_counts(processed):
    command = [sys.executable, "-m", "app.retrieval", "--processed-dir", str(processed)]
    result = subprocess.run(
        [
            *command,
            "search",
            "excavation",
            "--filters",
            '{"kind":"boq_item"}',
            "--limit",
            "10",
            "--offset",
            "10",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    response = json.loads(result.stdout)
    assert response["total"] == 35
    assert response["returned"] == 10
    assert response["has_more"]
    assert response["results"][0]["score"] > 0
    reference = response["results"][0]["record"]["source"]
    assert reference["filename"] == "boq.xlsx"
    assert reference["sheet"] == "Structural"
    assert reference["row_start"] is not None
    listing = subprocess.run(
        [*command, "list"], capture_output=True, text=True, check=True
    )
    assert len(json.loads(listing.stdout)["documents"]) == 2
    record_id = response["results"][0]["record_id"]
    record = subprocess.run(
        [*command, "get", record_id], capture_output=True, text=True, check=True
    )
    assert json.loads(record.stdout)["record"]["id"] == record_id
    missing = subprocess.run(
        [*command, "get", "missing"], capture_output=True, text=True, check=False
    )
    assert missing.returncode == 1
    assert json.loads(missing.stdout)["record"] is None
    invalid = subprocess.run(
        [*command, "search", "concrete", "--filters", "[]"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "validation error" in invalid.stderr.casefold()
