"""Resolve table-header relationships from source labels and cell spans."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extractors.pdf_models import ExtractedTableCell


def resolve_table_headers(
    cells: tuple[ExtractedTableCell, ...],
) -> tuple[ExtractedTableCell, ...]:
    """Resolve only source-marked headers whose grid direction is unambiguous.

    Unmarked first rows and leftmost text are not evidence of header roles.
    Coordinates are zero-based, with exclusive end bounds.
    """
    if not cells:
        return ()
    ordered = sorted(cells, key=lambda cell: (cell.row_start, cell.column_start))
    active: list[ExtractedTableCell] = []
    for cell in ordered:
        active = [other for other in active if other.row_end > cell.row_start]
        if (
            min(cell.row_start, cell.column_start) < 0
            or cell.row_end <= cell.row_start
            or cell.column_end <= cell.column_start
            or any(
                other.column_start < cell.column_end
                and cell.column_start < other.column_end
                for other in active
            )
        ):
            return tuple(
                replace(item, header_ids=(), header_status="unresolved")
                for item in cells
            )
        active.append(cell)

    # The first body cell ends the top header band. Blank corner cells do not.
    body_rows = [
        cell.row_start
        for cell in cells
        if cell.text.strip()
        and not (cell.is_header or cell.column_header or cell.row_header)
    ]
    band_end = min(body_rows) if body_rows else None
    min_column = min(cell.column_start for cell in cells)
    classified = []
    for cell in cells:
        column_header, row_header = cell.column_header, cell.row_header
        if (
            cell.is_header
            and not (column_header or row_header)
            and band_end is not None
        ):
            if cell.row_start < band_end and cell.row_end <= band_end:
                column_header = True
            else:
                left = sorted(
                    (
                        other
                        for other in cells
                        if other.row_start <= cell.row_start < other.row_end
                        and other.column_end <= cell.column_start
                    ),
                    key=lambda other: other.column_start,
                )
                edge = min_column
                for other in left:
                    if other.column_start != edge or not (
                        other.is_header or other.row_header
                    ):
                        break
                    edge = other.column_end
                row_header = edge == cell.column_start and any(
                    other.row_start < cell.row_end
                    and cell.row_start < other.row_end
                    and other.column_start >= cell.column_end
                    and not (other.is_header or other.column_header or other.row_header)
                    for other in cells
                )
        classified.append(
            replace(cell, column_header=column_header, row_header=row_header)
        )

    headers = [
        cell
        for cell in classified
        if cell.text.strip()
        and (cell.column_header or cell.row_header or cell.is_header)
    ]
    result = []
    # shortcut: O(cells * headers); index header intervals if very wide tables dominate.
    for cell in classified:
        parents = []
        ambiguous = cell.column_header and cell.row_header
        for header in headers:
            if header.id == cell.id:
                continue
            above = (
                header.row_end <= cell.row_start
                and header.column_start < cell.column_end
                and cell.column_start < header.column_end
            )
            left = (
                header.column_end <= cell.column_start
                and header.row_start < cell.row_end
                and cell.row_start < header.row_end
            )
            if header.column_header and header.row_header and (above or left):
                ambiguous = True
            if header.column_header and above:
                if (
                    header.column_start <= cell.column_start
                    and cell.column_end <= header.column_end
                ):
                    parents.append(header)
                else:
                    ambiguous = True
            elif header.row_header and left:
                if (
                    header.row_start <= cell.row_start
                    and cell.row_end <= header.row_end
                ):
                    parents.append(header)
                else:
                    ambiguous = True
            elif not (header.column_header or header.row_header) and (above or left):
                ambiguous = True
        known_role = cell.column_header or cell.row_header
        resolved = (
            not ambiguous
            and (known_role or bool(parents))
            and not (cell.is_header and not known_role)
        )
        parents.sort(
            key=lambda header: (
                not header.column_header,
                header.row_start,
                header.column_start,
            )
        )
        result.append(
            replace(
                cell,
                header_ids=tuple(header.id for header in parents) if resolved else (),
                header_status="resolved" if resolved else "unresolved",
            )
        )
    return tuple(result)


def table_cell_records(
    cells: tuple[ExtractedTableCell, ...],
) -> list[dict[str, object]]:
    return [
        {**asdict(cell), "id": cell.id, "header_ids": list(cell.header_ids)}
        for cell in cells
    ]
