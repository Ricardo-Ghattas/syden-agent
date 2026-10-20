"""Normalize PDF and workbook evidence into ordered, validated records."""

import re
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal

from openpyxl.utils import get_column_letter
from pydantic_core import to_jsonable_python

from ..extractors.pdf_models import ConvertedDocument, DocumentBlockType
from ..extractors.spreadsheet_models import SemanticWorkbook
from ..schemas import BoqItem, Record, Source
from .heading_detector import HeadingDetector

_ALIASES = {
    "item": {"item", "item no", "item number", "code", "item code", "no"},
    "description": {"description", "item description", "work description"},
    "unit": {"unit", "units", "uom"},
    "quantity": {"quantity", "qty"},
    "rate": {"rate", "unit rate", "unit price", "price"},
    "amount": {"amount", "total", "total amount"},
}


def _boq_columns(values):
    columns = {}
    for index, value in enumerate(values):
        label = re.sub(r"[^\w\s]", "", str(value or "").casefold())
        label = " ".join(label.split())
        for field, aliases in _ALIASES.items():
            if label in aliases:
                columns[field] = index
    return (
        columns
        if "description" in columns and {"quantity", "rate", "amount"} & columns.keys()
        else {}
    )


def _boq(values, columns):
    data = {
        field: values[index] if index < len(values) else None
        for field, index in columns.items()
    }
    for field in ("quantity", "rate", "amount"):
        value = data.get(field)
        # shortcut: parse only unambiguous decimal notation; retain locale-specific numbers as source text.
        if isinstance(value, str) and re.fullmatch(
            r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", value.strip()
        ):
            data[field] = Decimal(value.strip().replace(",", ""))
    return BoqItem(**data)


# * Normalize a PDF or workbook into validated records with source metadata.
def normalize(
    document: ConvertedDocument | SemanticWorkbook, source: Source
) -> list[Record]:
    """Return validated records without reading files or modifying source evidence.

    Record IDs and parent links follow source order. Locations retain the supplied
    file identity and the extracted page, sheet, row, and cell references.
    """
    records = []

    def add(kind, *, location=source, **fields):
        if "metadata" in fields:
            fields["metadata"] = to_jsonable_python(fields["metadata"])
        record = Record(
            id=f"{source.sha256}:{len(records) + 1}",
            document_id=source.sha256,
            order_index=len(records) + 1,
            kind=kind,
            source=location,
            **fields,
        )
        records.append(record)
        return record

    def location(**updates):
        return Source.model_validate({**source.model_dump(), **updates})

    if isinstance(document, ConvertedDocument):
        root = add(
            "document",
            text=document.source_filename,
            metadata={
                "page_count": document.page_count,
                "extractor": "pymupdf",
                "bbox_coordinates": "bottom-left PDF points",
                "table_row_coordinates": "one-based within table",
            },
            warnings=list(document.warnings),
        )
        detected = {block.id for block in HeadingDetector().detect(document)}
        headings = []
        for block in document.blocks:
            loc = location(
                page=block.page_number,
                block_id=block.id,
                bbox=block.bounding_box.to_list() if block.bounding_box else None,
            )
            if block.id in detected:
                level = block.heading_level or min(
                    len(block.text.split()[0].split(".")), 6
                )
                while headings and headings[-1][0] >= level:
                    headings.pop()
                record = add(
                    "heading",
                    location=loc,
                    text=block.text,
                    heading_level=level,
                    parent_id=headings[-1][1] if headings else root.id,
                    semantic_path=[h[2] for h in headings] + [block.text],
                )
                headings.append((level, record.id, block.text))
                continue
            common = {
                "parent_id": headings[-1][1] if headings else root.id,
                "semantic_path": [h[2] for h in headings],
            }
            if block.block_type != DocumentBlockType.TABLE:
                add("paragraph", location=loc, text=block.text, **common)
                continue
            headers = list(block.table_columns)
            table = add(
                "table",
                location=loc,
                text=block.text,
                headers=headers,
                rows=[list(row) for row in block.table_rows],
                table_cells=[asdict(cell) for cell in block.table_cells],
                **common,
            )
            columns = _boq_columns(headers)
            # Some tables expose BOQ column names only in their first source row.
            if not columns and block.table_rows:
                columns = _boq_columns(block.table_rows[0])
            if columns:
                first_is_header = bool(_boq_columns(block.table_rows[0]))
                for index, row in enumerate(block.table_rows, 1):
                    if index == 1 and first_is_header:
                        continue
                    add(
                        "boq_item",
                        location=location(
                            **{**loc.model_dump(), "row_start": index, "row_end": index}
                        ),
                        parent_id=table.id,
                        semantic_path=common["semantic_path"],
                        text=" | ".join(row),
                        rows=[list(row)],
                        boq=_boq(row, columns),
                    )
        return records

    root = add(
        "document",
        text=document.source_filename,
        metadata={
            "sheet_count": len(document.sheets),
            "named_ranges": list(document.named_ranges),
            "extractor": "openpyxl",
            "extractor_version": document.converter_version,
            "formula_execution": False,
        },
        warnings=list(document.warnings),
    )
    for sheet in document.sheets:
        sheet_source = location(sheet=sheet.name)
        sheet_record = add(
            "sheet",
            location=sheet_source,
            parent_id=root.id,
            text=sheet.name,
            semantic_path=[sheet.name],
            metadata={
                "state": sheet.state,
                "merged_ranges": list(sheet.merged_ranges),
                "hidden_rows": list(sheet.hidden_rows),
                "hidden_columns": list(sheet.hidden_columns),
                "charts": [asdict(chart) for chart in sheet.charts],
            },
        )
        regions = []
        for region in sheet.regions:
            record = add(
                "region",
                location=location(
                    sheet=sheet.name,
                    cell_range=region.range,
                    row_start=region.min_row,
                    row_end=region.max_row,
                ),
                parent_id=sheet_record.id,
                semantic_path=[sheet.name, region.name],
                text=region.name,
                metadata=asdict(region),
                warnings=list(region.warnings),
            )
            regions.append((region, record.id))
        by_row = defaultdict(list)
        for cell in sheet.cells:
            by_row[cell.row].append(cell)
        header_columns = {}
        heading = None
        for row, cells in sorted(by_row.items()):
            cells.sort(key=lambda cell: cell.column)
            start, end = cells[0].column, cells[-1].column
            values_by_column = {
                cell.column: cell.formula
                if cell.formula and cell.value is None
                else cell.value
                for cell in cells
            }
            values = [values_by_column.get(column) for column in range(start, end + 1)]
            matching = next(
                (
                    (region, identifier)
                    for region, identifier in regions
                    if region.min_row <= row <= region.max_row
                    and region.min_column <= start <= region.max_column
                ),
                None,
            )
            parent = matching[1] if matching else sheet_record.id
            region_key = parent
            columns = _boq_columns(values)
            nonempty = [
                cell for cell in cells if cell.value is not None or cell.formula
            ]
            kind = "row"
            fields = {}
            if columns:
                header_columns[region_key] = {
                    field: start + index for field, index in columns.items()
                }
                fields["metadata"] = {"role": "header"}
            elif (
                len(nonempty) == 1
                and (nonempty[0].bold or nonempty[0].merged_range)
                and isinstance(nonempty[0].value, str)
            ):
                kind = "heading"
                fields["heading_level"] = 1
                heading = str(nonempty[0].value)
            elif region_key in header_columns and nonempty:
                columns = header_columns[region_key]
                kind = "boq_item"
                fields["boq"] = _boq(
                    [values_by_column.get(column) for column in columns.values()],
                    {field: index for index, field in enumerate(columns)},
                )
            add(
                kind,
                location=location(
                    sheet=sheet.name,
                    row_start=row,
                    row_end=row,
                    cell_range=f"{get_column_letter(start)}{row}:{get_column_letter(end)}{row}",
                ),
                parent_id=parent,
                semantic_path=[sheet.name] + ([heading] if heading else []),
                text=" | ".join(
                    "" if value is None else str(value) for value in values
                ),
                rows=[values],
                cells=[asdict(cell) for cell in cells],
                **fields,
            )
    return records
