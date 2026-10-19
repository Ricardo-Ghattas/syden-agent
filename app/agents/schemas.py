"""Application contracts; source references must match supplied retrieval evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class AgentRequest(Model):
    question: str = Field(min_length=1, max_length=16000, pattern=r"\S")


class SourceReference(Model):
    document_id: str
    record_id: str | None
    filename: str
    page: int | None
    sheet: str | None
    row_start: int | None
    row_end: int | None
    cell_range: str | None


class AgentResponse(Model):
    answer: str = Field(min_length=1, pattern=r"\S")
    answer_type: Literal["document", "general", "insufficient"]
    sources: list[SourceReference]
    warnings: list[str]


class AgentError(RuntimeError):
    """A controlled request failure safe to display without upstream payloads."""
