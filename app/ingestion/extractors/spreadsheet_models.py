"""Represent workbook source values, formulas, regions, and sheet metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SourceType(StrEnum):
    XLSX = "xlsx"


@dataclass(frozen=True, slots=True)
class WorkbookCell:
    coordinate: str
    row: int
    column: int
    value: object
    formula: str | None = None
    cached_value: object = None
    value_type: str = "text"
    number_format: str | None = None
    currency_or_unit: str | None = None
    bold: bool = False
    italic: bool = False
    fill_color: str | None = None
    indent: int = 0
    hidden: bool = False
    comment: str | None = None
    merged_range: str | None = None
    cross_sheet_references: tuple[str, ...] = ()
    # Keep source values separate so display values cannot overwrite formulas or identifiers.
    raw_value: object = None
    blank_kind: str | None = None


@dataclass(frozen=True, slots=True)
class WorkbookRegion:
    id: str
    name: str
    range: str
    kind: str
    min_row: int
    max_row: int
    min_column: int
    max_column: int
    header_rows: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    semantic_role: str = "table"
    confidence: float = 0.5
    source: str = "inferred"
    title_rows: tuple[int, ...] = ()
    total_rows: tuple[int, ...] = ()
    subtotal_rows: tuple[int, ...] = ()
    note_rows: tuple[int, ...] = ()
    parameter_block: bool = False
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class WorkbookChart:
    id: str
    title: str | None
    chart_type: str
    anchor: str | None
    series: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class WorkbookSheet:
    name: str
    state: str
    cells: tuple[WorkbookCell, ...]
    regions: tuple[WorkbookRegion, ...]
    merged_ranges: tuple[str, ...]
    charts: tuple[WorkbookChart, ...]
    hidden_rows: tuple[int, ...]
    hidden_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticWorkbook:
    source_filename: str
    source_type: SourceType
    sheets: tuple[WorkbookSheet, ...]
    named_ranges: tuple[dict[str, object], ...] = ()
    warnings: tuple[str, ...] = ()
    converter_version: str = "unknown"
    source_metadata: dict[str, object] = field(default_factory=dict)
