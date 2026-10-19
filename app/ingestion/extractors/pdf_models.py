"""Represent PDF source blocks, table-cell spans, and page coordinates."""

from dataclasses import dataclass
from enum import StrEnum


class DocumentBlockType(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    EQUATION = "equation"
    QUOTE = "quote"


class DocumentBlockRole(StrEnum):
    BODY = "body"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    CAPTION = "caption"


class FigureType(StrEnum):
    CHART = "chart"
    DIAGRAM = "diagram"
    ILLUSTRATION = "illustration"
    MAP = "map"
    PHOTOGRAPH = "photograph"
    PLOT = "plot"
    SCREENSHOT = "screenshot"


@dataclass(frozen=True, slots=True)
class ExtractedTableCell:
    text: str
    row_start: int
    row_end: int
    column_start: int
    column_end: int
    column_header: bool = False
    row_header: bool = False
    is_header: bool = False
    header_ids: tuple[str, ...] = ()
    header_status: str = "unresolved"

    @property
    def id(self) -> str:
        """Coordinates identify a cell within its owning table."""
        return f"r{self.row_start}c{self.column_start}"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: float
    bottom: float
    right: float
    top: float

    def __post_init__(self) -> None:
        if self.right < self.left or self.top < self.bottom:
            raise ValueError("Bounding box coordinates are inverted.")

    def to_list(self) -> list[float]:
        return [self.left, self.bottom, self.right, self.top]


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    """A provider-independent piece of extracted source evidence."""

    id: str
    text: str
    order_index: int
    page_number: int | None = None
    is_heading: bool = False
    heading_level: int | None = None
    block_type: DocumentBlockType = DocumentBlockType.PARAGRAPH
    caption: str | None = None
    list_ordered: bool = False
    table_columns: tuple[str, ...] = ()
    table_rows: tuple[tuple[str, ...], ...] = ()
    table_cells: tuple[ExtractedTableCell, ...] = ()
    asset_bytes: bytes | None = None
    asset_extension: str | None = None
    figure_type: FigureType | None = None
    figure_number: str | None = None
    source_block_ids: tuple[str, ...] = ()
    generated_description: str | None = None
    generated_description_source: str | None = None
    alt_text: str | None = None
    bounding_box: BoundingBox | None = None
    role: DocumentBlockRole = DocumentBlockRole.BODY
    suppressed_from_chunks: bool = False
    list_marker: str | None = None
    list_numbering_style: str | None = None
    list_ordinal: str | None = None
    list_depth: int = 0
    list_parent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Document block ID cannot be empty.")
        if not self.text.strip():
            raise ValueError("Document block text cannot be empty.")
        if self.order_index < 1:
            raise ValueError("Document block order must be greater than zero.")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("Document block page must be greater than zero.")
        if self.heading_level is not None and not 1 <= self.heading_level <= 6:
            raise ValueError("Markdown heading level must be between 1 and 6.")
        if self.list_depth < 0:
            raise ValueError("List depth cannot be negative.")

        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "text", self.text.strip())
        if self.is_heading and self.block_type is DocumentBlockType.PARAGRAPH:
            object.__setattr__(self, "block_type", DocumentBlockType.HEADING)
        object.__setattr__(self, "table_columns", tuple(self.table_columns))
        object.__setattr__(
            self, "table_rows", tuple(tuple(row) for row in self.table_rows)
        )
        from ..transformers.table_headers import resolve_table_headers

        object.__setattr__(
            self, "table_cells", resolve_table_headers(tuple(self.table_cells))
        )
        object.__setattr__(
            self,
            "source_block_ids",
            tuple(dict.fromkeys(self.source_block_ids or (self.id,))),
        )
        if (
            self.generated_description is None
            and self.generated_description_source is not None
        ):
            raise ValueError("Generated description provenance requires a description.")
        if self.asset_extension is not None:
            extension = self.asset_extension.lower().lstrip(".")
            if not extension.isalnum():
                raise ValueError("Asset extension contains unsafe characters.")
            object.__setattr__(self, "asset_extension", extension)


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    source_filename: str
    page_count: int
    blocks: tuple[DocumentBlock, ...] = ()
    warnings: tuple[str, ...] = ()
