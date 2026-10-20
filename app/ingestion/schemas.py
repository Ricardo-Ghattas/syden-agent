"""Validated JSONL contract shared by the two source adapters."""

from datetime import date, datetime, time
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Scalar = str | bool | int | float | Decimal | datetime | date | time | None
RecordKind = Literal[
    "document", "sheet", "region", "heading", "paragraph", "table", "row", "boq_item"
]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Source(Model):
    filename: str = Field(min_length=1)
    raw_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["pdf", "xlsx"]
    page: int | None = Field(default=None, ge=1)
    sheet: str | None = None
    row_start: int | None = Field(default=None, ge=1)
    row_end: int | None = Field(default=None, ge=1)
    cell_range: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    block_id: str | None = None

    @model_validator(mode="after")
    def consistent_location(self):
        if self.format == "pdf" and self.sheet is not None:
            raise ValueError("PDF source cannot have a sheet")
        if self.format == "xlsx" and (self.page is not None or self.bbox is not None):
            raise ValueError("Workbook source cannot have a PDF page or bounding box")
        if (self.row_start is None) != (self.row_end is None):
            raise ValueError("Row bounds must be supplied together")
        if (
            self.row_start is not None
            and self.row_end is not None
            and self.row_end < self.row_start
        ):
            raise ValueError("Row bounds are inverted")
        return self


class Cell(Model):
    coordinate: str
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    value: Scalar
    formula: str | None = None
    cached_value: Scalar = None
    value_type: str
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
    raw_value: Scalar = None
    blank_kind: str | None = None


class TableCell(Model):
    text: str
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=1)
    column_start: int = Field(ge=0)
    column_end: int = Field(ge=1)
    column_header: bool = False
    row_header: bool = False
    is_header: bool = False
    header_ids: tuple[str, ...] = ()
    header_status: Literal["resolved", "unresolved"]

    @model_validator(mode="after")
    def positive_span(self):
        if self.row_end <= self.row_start or self.column_end <= self.column_start:
            raise ValueError("Table cell bounds must have a positive span")
        return self


class BoqItem(Model):
    item: Scalar = None
    description: Scalar = None
    unit: Scalar = None
    quantity: Scalar = None
    rate: Scalar = None
    amount: Scalar = None


class Record(Model):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1)
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_index: int = Field(ge=1)
    kind: RecordKind
    source: Source
    parent_id: str | None = None
    semantic_path: list[str] = Field(default_factory=list)
    text: str = ""
    heading_level: int | None = Field(default=None, ge=1, le=6)
    headers: list[str] = Field(default_factory=list)
    rows: list[list[Scalar]] = Field(default_factory=list)
    cells: list[Cell] = Field(default_factory=list)
    table_cells: list[TableCell] = Field(default_factory=list)
    boq: BoqItem | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def record_contract(self):
        if self.document_id != self.source.sha256:
            raise ValueError("Document ID must match source SHA-256")
        if self.kind == "heading" and (self.heading_level is None or not self.text):
            raise ValueError("Heading requires a level and text")
        if self.kind == "boq_item" and self.boq is None:
            raise ValueError("BOQ record requires item data")
        if self.kind in {"row", "boq_item"} and self.source.row_start is None:
            raise ValueError("Row record requires source row bounds")
        return self
