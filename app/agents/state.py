"""Local session memory and observable execution state; no hidden reasoning stored."""

from time import monotonic
from uuid import uuid4

from pydantic import Field

from .schemas import AgentResponse, AgentStatus, Model, SourceReference, StopReason

TERMINAL = {
    AgentStatus.COMPLETED,
    AgentStatus.INSUFFICIENT_EVIDENCE,
    AgentStatus.LIMIT_REACHED,
    AgentStatus.FAILED,
    AgentStatus.CANCELLED,
}


class ConversationTurn(Model):
    question: str
    response: AgentResponse


class ConversationSession(Model):
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    previous_response_id: str | None = None
    history: list[ConversationTurn] = Field(default_factory=list, repr=False)
    sources: list[SourceReference] = Field(default_factory=list, repr=False)
    warnings: list[str] = Field(default_factory=list, repr=False)
    context_bytes: int = Field(default=0, ge=0)
    representation_hashes: set[str] = Field(default_factory=set, repr=False)
    # Only data already supplied to the model can authorize remembered citations.
    evidence: dict[str, dict] = Field(default_factory=dict, repr=False)

    def clear(self) -> None:
        self.previous_response_id = None
        self.history.clear()
        self.sources.clear()
        self.evidence.clear()
        self.context_bytes = 0
        self.warnings.clear()
        self.representation_hashes.clear()


class ToolOutcome(Model):
    name: str
    signature: str = Field(repr=False)
    success: bool
    record_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    returned: int = Field(default=0, ge=0)
    duration_seconds: float = Field(ge=0)


class AgentState(Model):
    session_id: str
    question: str = Field(repr=False)
    execution_id: str = Field(default_factory=lambda: uuid4().hex)
    status: AgentStatus = AgentStatus.INITIALIZED
    iterations: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    outcomes: list[ToolOutcome] = Field(default_factory=list, repr=False)
    sources: list[SourceReference] = Field(default_factory=list, repr=False)
    evidence: dict[str, dict] = Field(default_factory=dict, repr=False)
    examined_record_ids: set[str] = Field(default_factory=set)
    history: list[ConversationTurn] = Field(default_factory=list, repr=False)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    started_at: float = Field(default_factory=monotonic)
    stop_reason: StopReason | None = None
    final_response: AgentResponse | None = Field(default=None, repr=False)

    def transition(self, status: AgentStatus) -> None:
        if self.status in TERMINAL:
            raise ValueError("A finished execution cannot transition again")
        if status == AgentStatus.INITIALIZED:
            raise ValueError("An execution cannot return to initialized")
        if (
            status == AgentStatus.AWAITING_TOOL_RESULTS
            and self.status != AgentStatus.RUNNING
        ):
            raise ValueError("Tool execution requires a running model response")
        if self.status == AgentStatus.INITIALIZED and status != AgentStatus.RUNNING:
            raise ValueError("An execution must start running")
        self.status = status
