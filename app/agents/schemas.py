"""Application contracts; source references must match supplied retrieval evidence."""

from enum import StrEnum
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


class AnswerContent(Model):
    answer: str = Field(min_length=1, pattern=r"\S")
    answer_type: Literal["document", "general", "insufficient"]
    sources: list[SourceReference]
    warnings: list[str]


class AgentStatus(StrEnum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    AWAITING_TOOL_RESULTS = "awaiting_tool_results"
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LIMIT_REACHED = "limit_reached"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StopReason(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MAX_ITERATIONS = "max_iterations"
    MAX_TOOL_CALLS = "max_tool_calls"
    TIMEOUT = "timeout"
    REPEATED_CALLS = "repeated_calls"
    TOOL_FAILURES = "tool_failures"
    CONTEXT_LIMIT = "context_limit"
    SESSION_LIMIT = "session_limit"
    API_ERROR = "api_error"
    INVALID_RESPONSE = "invalid_response"
    CANCELLED = "cancelled"


class AgentResponse(AnswerContent):
    status: AgentStatus = AgentStatus.COMPLETED
    iterations: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    stop_reason: StopReason | None = None
    session_id: str | None = None
    execution_id: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0, ge=0)


class AgentError(RuntimeError):
    """A controlled request failure safe to display without upstream payloads."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable
