"""Extract PDF text, tables, headings, and page locations with PyMuPDF."""

from pathlib import Path
from typing import Any, cast

import pymupdf
from pymupdf.table import TableHeader

from .pdf_models import (
    BoundingBox,
    ConvertedDocument,
    DocumentBlock,
    DocumentBlockType,
    ExtractedTableCell,
)


# * Extract text, tables, headings, and page locations from a PDF file.
def read_pdf(path: Path) -> ConvertedDocument:
    """Return ordered source blocks with page numbers and bounding boxes.

    Scanned or image-only pages produce warnings; OCR is not performed.
    """
    blocks = []
    warnings = []
    with pymupdf.open(path) as document:
        # * Check that the PDF is unencrypted and does not require a password.
        if not document.is_pdf or document.needs_pass:
            raise ValueError("Expected an unencrypted PDF.")
        # Bookmark levels provide explicit heading evidence before font-based inference.
        outline = {
            (page, " ".join(title.split())): min(level, 6)
            for level, title, page in document.get_toc()
        }
        # * Extract tables and text blocks from each page, preserving order and bounding boxes.
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            page_number = page_index + 1
            finder = page.find_tables(use_layout=False)
            if finder is None:
                raise ValueError(f"Table extraction failed on page {page_number}.")
            tables = finder.tables
            positioned = []
            # * Extract tables first, then text blocks, preserving order by top-left coordinates.
            for table in tables:
                header = cast(TableHeader, table.header)
                # Read by detected cell bounds: table.extract() can reorder decimal punctuation.
                rows = tuple(
                    tuple(
                        page.get_textbox(box).strip() if box else ""
                        for box in row.cells
                    )
                    for row in table.rows
                )
                cells = []
                column_edges = sorted(
                    {box[0] for box in table.cells} | {box[2] for box in table.cells}
                )
                for row_index, row in enumerate(table.rows):
                    for column_index, box in enumerate(row.cells):
                        if box is None:
                            continue
                        # PyMuPDF omits covered cells of a merged source cell.
                        row_end = row_index + 1
                        while (
                            row_end < len(table.rows)
                            and table.rows[row_end].bbox[3] <= box[3] + 0.1
                        ):
                            row_end += 1
                        column_end = sum(edge < box[2] - 0.1 for edge in column_edges)
                        text = rows[row_index][column_index]
                        is_header = (
                            not header.external
                            and row_index == 0
                            and bool(header.names)
                        )
                        cells.append(
                            ExtractedTableCell(
                                text,
                                row_index,
                                row_end,
                                column_index,
                                column_end,
                                column_header=is_header,
                                is_header=is_header,
                            )
                        )
                positioned.append(
                    (
                        table.bbox,
                        DocumentBlockType.TABLE,
                        "\n".join(" | ".join(row) for row in rows),
                        None,
                        rows,
                        tuple(cells),
                        tuple(value or "" for value in header.names or ()),
                    )
                )
            layout = cast(
                dict[str, Any],
                page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT, sort=True),
            )
            # * Extract text blocks, skipping those that overlap with detected tables.
            for block in layout["blocks"]:
                for line in block.get("lines", ()):
                    spans = line["spans"]
                    text = "".join(span["text"] for span in spans).strip()
                    if not text:
                        continue
                    box = pymupdf.Rect(line["bbox"])
                    if any(
                        pymupdf.Rect(table.bbox).contains(
                            box.tl + (box.br - box.tl) / 2
                        )
                        for table in tables
                    ):
                        continue
                    level = outline.get((page_number, " ".join(text.split())))
                    # shortcut: font-based heading inference is heuristic; use a layout backend for ambiguous PDFs.
                    if (
                        level is None
                        and len(text) <= 120
                        and max(s["size"] for s in spans) >= 14
                    ):
                        level = 1 if max(s["size"] for s in spans) >= 20 else 2
                    kind = (
                        DocumentBlockType.HEADING
                        if level
                        else DocumentBlockType.PARAGRAPH
                    )
                    positioned.append(
                        (
                            (box.x0, box.y0, box.x1, box.y1),
                            kind,
                            text,
                            level,
                            (),
                            (),
                            (),
                        )
                    )
            if not positioned:
                warnings.append(f"page_requires_ocr_or_has_no_text:{page_number}")
            if page.get_images():
                warnings.append(f"page_images_not_extracted:{page_number}")
            for box, kind, text, level, rows, cells, columns in sorted(
                positioned, key=lambda item: (item[0][1], item[0][0])
            ):
                if not text.strip():
                    continue
                blocks.append(
                    DocumentBlock(
                        id=f"p{page_number}-b{len(blocks) + 1}",
                        text=text,
                        order_index=len(blocks) + 1,
                        page_number=page_number,
                        block_type=kind,
                        is_heading=level is not None,
                        heading_level=level,
                        table_rows=rows,
                        table_columns=columns,
                        table_cells=cells,
                        # Exported bounding boxes use bottom-left coordinates; PyMuPDF uses top-left.
                        bounding_box=BoundingBox(
                            box[0],
                            page.rect.height - box[3],
                            box[2],
                            page.rect.height - box[1],
                        ),
                    )
                )
        return ConvertedDocument(
            path.name, len(document), tuple(blocks), tuple(warnings)
        )
