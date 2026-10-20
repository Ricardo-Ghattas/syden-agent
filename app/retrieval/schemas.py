"""Validated retrieval inputs, diagnostics, and citation-bearing outputs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ingestion.schemas import Record, RecordKind


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class SearchFilters(Model):
    format: Literal["pdf", "xlsx"] | None = None
    document_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    filename: str | None = None
    sheet: str | None = None
    section: str | None = None
    item_code: str | None = None
    kind: RecordKind | None = None
    page: int | None = Field(default=None, ge=1)

    @field_validator("kind", mode="before")
    @classmethod
    def trim_kind(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("filename", "sheet", "section", "item_code", "kind")
    @classmethod
    def nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Filter values must not be blank")
        return value.strip() if value is not None else None


class RankingWeights(Model):
    body: float = Field(default=1.0, ge=0, le=10)
    description: float = Field(default=3.0, ge=0, le=10)
    heading: float = Field(default=2.0, ge=0, le=10)
    metadata: float = Field(default=0.35, ge=0, le=10)


class IndexIssue(Model):
    path: str
    line: int | None = Field(default=None, ge=1)
    code: Literal[
        "missing_directory",
        "missing_file",
        "read_error",
        "malformed_record",
        "duplicate_id",
        "inconsistent_document",
    ]
    message: str
    record_id: str | None = None


class IndexReport(Model):
    files: int = Field(ge=0)
    records: int = Field(ge=0)
    documents: int = Field(ge=0)
    issues: list[IndexIssue] = Field(default_factory=list)


class DocumentInfo(Model):
    document_id: str
    filename: str
    format: Literal["pdf", "xlsx"]
    record_count: int = Field(ge=1)
    sheets: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SearchResult(Model):
    record_id: str
    kind: str
    score: float = Field(ge=0)
    matched_fields: list[str]
    matched_terms: list[str]
    snippet: str
    record: Record


class SearchResponse(Model):
    query: str
    filters: SearchFilters
    results: list[SearchResult]
    total: int = Field(ge=0)
    returned: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    has_more: bool
    next_offset: int | None
    index_complete: bool
    index_issues: list[IndexIssue]
