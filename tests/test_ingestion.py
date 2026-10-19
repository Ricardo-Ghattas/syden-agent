import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pymupdf
import pytest
from openpyxl import Workbook
from openpyxl.worksheet.table import Table
from pydantic import ValidationError

from app.ingestion.exporters.jsonl import export_jsonl
from app.ingestion.extractors.excel import (
    SpreadsheetConversionError,
    SpreadsheetConverter,
    _validate_ooxml,
)
from app.ingestion.extractors.pdf import read_pdf
from app.ingestion.pipeline import process_document
from app.ingestion.schemas import Record, Source
from app.ingestion.storage import archive_original
from app.ingestion.transformers.normalize import normalize


def read_records(output):
    return [
        Record.model_validate_json(line) for line in output.read_text().splitlines()
    ]


def test_excel_preserves_structure_values_and_provenance(workbook, tmp_path):
    output = process_document(workbook, tmp_path / "data")
    records = read_records(output)
    original = records[0]
    archived = tmp_path / "data" / original.source.raw_path
    assert archived.read_bytes() == workbook.read_bytes()
    assert original.source.filename == "boq.xlsx"
    assert original.kind == "document"
    assert [r.text for r in records if r.kind == "sheet"] == ["BOQ", "Notes"]
    assert any(
        r.kind == "heading" and r.text.startswith("Concrete works") for r in records
    )
    assert any(
        r.kind == "region" and r.metadata["kind"] == "excel_table" for r in records
    )
    items = [r for r in records if r.kind == "boq_item"]
    assert len(items) == 2
    assert items[0].boq is not None
    assert items[1].boq is not None
    assert items[0].boq.item == "001"
    assert items[0].boq.quantity == 0
    assert items[0].boq.rate == 12.5
    assert items[1].boq.quantity == -2.5
    assert items[0].source.sheet == "BOQ"
    assert items[0].source.row_start == 3
    assert items[0].source.cell_range == "A3:F3"
    formula = next(c for c in items[0].cells if c.coordinate == "F3")
    assert formula.formula == formula.raw_value == "=D3*E3"
    assert formula.cached_value is None
    assert any("missing_formula_cache:BOQ!F3" in w for w in original.warnings)
    assert (
        next(c for c in items[0].cells if c.coordinate == "E3").currency_or_unit
        == "USD"
    )
    assert all(c.hidden for c in items[1].cells)
    notes = next(r for r in records if r.source.sheet == "Notes" and r.kind == "row")
    assert notes.cells[0].value == "2026-10-19T00:00:00"
    assert notes.cells[1].value is False
    assert notes.cells[2].cross_sheet_references == ("BOQ",)
    identifiers = {r.id for r in records}
    assert len(identifiers) == len(records)
    assert all(r.parent_id in identifiers for r in records[1:])
    before = output.read_bytes()
    assert process_document(workbook, tmp_path / "data").read_bytes() == before


def test_pdf_preserves_headings_table_boq_and_pages(pdf, tmp_path):
    records = read_records(process_document(pdf, tmp_path / "data"))
    assert records[0].metadata["page_count"] == 2
    assert [(r.text, r.source.page) for r in records if r.kind == "heading"] == [
        ("1 Concrete", 1),
        ("2 Installation", 2),
    ]
    paragraph = next(r for r in records if "35 MPa" in r.text)
    assert paragraph.kind == "paragraph"
    assert paragraph.semantic_path == ["1 Concrete"]
    table = next(r for r in records if r.kind == "table")
    assert len(table.rows) == 3
    assert table.rows[1][0] == "001"
    assert table.source.page == 1
    assert table.source.bbox is not None
    assert len(table.table_cells) == 18
    assert table.table_cells[6].header_ids == ("r0c0",)
    items = [r for r in records if r.kind == "boq_item"]
    assert len(items) == 2
    assert items[0].boq is not None
    assert items[1].boq is not None
    assert items[0].boq.item == "001"
    assert items[0].boq.quantity == "0"
    assert items[1].boq.quantity == "-2.5"
    assert items[0].parent_id == table.id
    assert items[0].source.row_start == 2
    assert items[0].source.block_id == table.source.block_id


def test_empty_pdf_surfaces_ocr_warning(tmp_path):
    path = tmp_path / "empty.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(path)
    records = read_records(process_document(path, tmp_path / "data"))
    assert records[0].warnings == ["page_requires_ocr_or_has_no_text:1"]


def test_cli(pdf, tmp_path):
    result = subprocess.run(
        [sys.executable, "main.py", str(pdf), "--data-dir", str(tmp_path / "data")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert read_records(Path(result.stdout.strip()))


def test_same_name_inputs_do_not_overwrite(workbook, tmp_path):
    output = process_document(workbook, tmp_path / "data")
    before = output.read_bytes()
    original = read_records(output)[0]
    workbook.write_bytes(b"invalid changed workbook")
    with pytest.raises(SpreadsheetConversionError):
        process_document(workbook, tmp_path / "data")
    assert output.read_bytes() == before
    assert (tmp_path / "data" / original.source.raw_path).exists()
    assert len(list((tmp_path / "data" / "raw").glob("*.xlsx"))) == 2


def test_atomic_export_keeps_previous_output_on_validation_failure(pdf, tmp_path):
    output = process_document(pdf, tmp_path / "data")
    records = read_records(output)
    before = output.read_bytes()
    records[-1].order_index = 0
    with pytest.raises(ValidationError):
        export_jsonl(records, output)
    assert output.read_bytes() == before
    assert not list(output.parent.glob("*.tmp"))


def test_schema_rejects_invalid_provenance_and_records():
    source = Source(
        filename="a.pdf", raw_path="raw/a.pdf", sha256="a" * 64, format="pdf"
    )
    with pytest.raises(ValidationError):
        Source.model_validate({**source.model_dump(), "page": 0})
    with pytest.raises(ValidationError):
        Source.model_validate({**source.model_dump(), "row_start": 3, "row_end": 2})
    with pytest.raises(ValidationError):
        Record(
            id="1", document_id="a" * 64, order_index=1, kind="boq_item", source=source
        )
    with pytest.raises(ValidationError):
        Record(
            id="1", document_id="b" * 64, order_index=1, kind="document", source=source
        )


def test_rejects_unsupported_and_corrupt_inputs(tmp_path):
    path = tmp_path / "old.xls"
    path.write_bytes(b"not an excel workbook")
    with pytest.raises(ValueError, match="Supported inputs"):
        process_document(path, tmp_path / "data")
    path = tmp_path / "bad.xlsx"
    path.write_bytes(b"not a zip")
    with pytest.raises(SpreadsheetConversionError, match="ZIP"):
        process_document(path, tmp_path / "data")
    assert not list((tmp_path / "data" / "processed").glob("*"))


def test_rejects_unsafe_workbook_archive(tmp_path):
    path = tmp_path / "unsafe.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(SpreadsheetConversionError, match="unsafe ZIP path"):
        _validate_ooxml(path)


def test_jsonl_is_unicode_and_json(workbook, tmp_path):
    output = process_document(workbook, tmp_path / "data")
    assert "أساس" in output.read_text()
    assert all(
        isinstance(json.loads(line), dict) for line in output.read_text().splitlines()
    )


def test_excel_formula_cache_and_merged_cells(workbook, tmp_path):
    # openpyxl does not calculate formulas; supply an Excel-style cached value in the fixture.
    with ZipFile(workbook) as archive:
        contents = {entry.filename: archive.read(entry) for entry in archive.infolist()}
    contents["xl/worksheets/sheet1.xml"], count = re.subn(
        rb"<f>D3\*E3</f><v\s*(?:/>|></v>)",
        b"<f>D3*E3</f><v>0</v>",
        contents["xl/worksheets/sheet1.xml"],
    )
    assert count == 1
    with ZipFile(workbook, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    records = read_records(process_document(workbook, tmp_path / "data"))
    item = next(r for r in records if r.kind == "boq_item")
    cell = next(c for c in item.cells if c.coordinate == "F3")
    assert cell.value == cell.cached_value == 0
    assert cell.formula == "=D3*E3"
    assert item.boq is not None
    assert item.boq.amount == 0
    title = next(r for r in records if r.kind == "heading")
    assert title.cells[0].merged_range == "A1:F1"
    assert len(title.cells) == 6
    assert all(c.blank_kind == "empty_cell" for c in title.cells[1:])


def test_modified_archive_is_not_overwritten(workbook, tmp_path):
    output = process_document(workbook, tmp_path / "data")
    source = read_records(output)[0].source
    archive = tmp_path / "data" / source.raw_path
    archive.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Archived file has been modified"):
        process_document(workbook, tmp_path / "data")
    assert archive.read_bytes() == b"modified"
    assert not list(archive.parent.glob("tmp*"))


def test_pdf_merged_header_span(tmp_path):
    path = tmp_path / "merged.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        for y in [100, 130, 160, 190]:
            page.draw_line((40, y), (340, y))
        for x in [40, 240, 340]:
            page.draw_line((x, 100), (x, 190))
        page.draw_line((140, 130), (140, 190))
        page.insert_text((50, 120), "Quantities", fontsize=10)
        page.insert_text((250, 120), "Amount", fontsize=10)
        for y in [150, 180]:
            page.insert_text((50, y), "12.50", fontsize=10)
            page.insert_text((150, y), "-2.50", fontsize=10)
            page.insert_text((250, y), "10", fontsize=10)
        document.save(path)
    records = read_records(process_document(path, tmp_path / "data"))
    table = next(r for r in records if r.kind == "table")
    assert table.rows[1] == ["12.50", "-2.50", "10"]
    assert table.table_cells[0].column_start == 0
    assert table.table_cells[0].column_end == 2
    assert all(c.header_ids == ("r0c0",) for c in table.table_cells[2:4])


def test_plain_excel_table_preserves_literal_formula_text(tmp_path):
    path = tmp_path / "plain.xlsx"
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.append(["Name", "Value"])
    sheet.append(["Literal", "=not a formula"])
    sheet["B2"].data_type = "s"
    sheet.append(["Zero", 0])
    sheet.append(["False", False])
    sheet.add_table(Table(displayName="Values", ref="A1:B4"))
    book.save(path)
    book.close()
    records = read_records(process_document(path, tmp_path / "data"))
    region = next(r for r in records if r.kind == "region")
    assert region.metadata["header_rows"] == [1]
    rows = [r for r in records if r.kind == "row"]
    assert rows[1].cells[1].raw_value == "=not a formula"
    assert rows[1].cells[1].formula is None
    assert rows[1].cells[1].value_type == "text"
    assert rows[2].cells[1].value == 0
    assert rows[3].cells[1].value is False
    assert all(r.parent_id == region.id for r in rows)


@pytest.mark.parametrize("input_fixture", ["pdf", "workbook"])
def test_extraction_normalization_and_export_work_independently(
    input_fixture, request, tmp_path
):

    path = request.getfixturevalue(input_fixture)
    source = Source(
        filename=path.name,
        raw_path=f"raw/{path.name}",
        sha256="a" * 64,
        format=path.suffix[1:],
    )
    document = (
        read_pdf(path)
        if input_fixture == "pdf"
        else SpreadsheetConverter().convert(path)
    )
    path.unlink()

    records = normalize(document, source)
    output = export_jsonl(iter(records), tmp_path / "export" / "records.jsonl")
    restored = read_records(output)

    assert [r.model_dump(mode="json") for r in restored] == [
        r.model_dump(mode="json") for r in records
    ]
    assert records[0].text == source.filename
    assert all(
        r.source.filename == source.filename
        and r.source.raw_path == source.raw_path
        and r.source.sha256 == source.sha256
        for r in restored
    )
    assert any(r.kind == "boq_item" for r in restored)
    assert not (tmp_path / "raw").exists()


def test_archiving_cleans_temporary_snapshot_when_input_cannot_be_read(tmp_path):

    raw_dir = tmp_path / "raw"
    with pytest.raises(FileNotFoundError):
        archive_original(tmp_path / "missing.pdf", raw_dir)
    assert list(raw_dir.iterdir()) == []
