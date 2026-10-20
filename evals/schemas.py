"""Versioned, independently labelled evaluation cases."""

from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import Field, model_validator

from app.agents.config import AgentConfig
from app.agents.schemas import AgentStatus, Model, StopReason
from app.retrieval.schemas import SearchFilters


class Fact(Model):
    label: str = Field(pattern=r"^[a-z][a-z_0-9]*$")
    value: str = Field(min_length=1)
    unit: str = ""
    numeric: bool = False
    tolerance: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    support_ids: list[str] = Field(min_length=1)
    field: str = Field(min_length=1)
    operation: Literal["value", "sum", "min", "max", "contains"] = "value"

    @model_validator(mode="after")
    def valid_fact(self):
        if len(self.support_ids) != len(set(self.support_ids)):
            raise ValueError("Duplicate support IDs would double-count evidence")
        if self.numeric:
            try:
                finite = Decimal(self.value).is_finite()
            except InvalidOperation as error:
                raise ValueError("Invalid numerical reference") from error
            if not finite:
                raise ValueError("Numerical references must be finite")
        if not self.numeric and (
            self.tolerance or self.operation in {"sum", "min", "max"}
        ):
            raise ValueError("Arithmetic and tolerances require numerical references")
        return self


class Case(Model):
    version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: str
    layer: Literal["retrieval", "agent"]
    synthetic: bool
    verified: bool
    live_eligible: bool = True
    question: str = Field(min_length=1)
    query: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    k: int = Field(default=5, ge=1, le=100)
    expected_ids: list[str] = Field(default_factory=list)
    exact_item: bool = False
    exhaustive: bool = False
    facts: list[Fact] = Field(default_factory=list)
    should_abstain: bool = False
    expected_status: AgentStatus = AgentStatus.COMPLETED
    expected_stop: StopReason | None = None
    required_tools: list[str] = Field(default_factory=list)
    max_tools: int = Field(default=6, ge=0)
    setup_questions: list[str | None] = Field(default_factory=list)
    config: AgentConfig = Field(default_factory=AgentConfig)

    @model_validator(mode="after")
    def consistent(self):
        if not self.verified:
            return self
        if len(self.expected_ids) != len(set(self.expected_ids)):
            raise ValueError("Duplicate expected IDs")
        if len({fact.label for fact in self.facts}) != len(self.facts):
            raise ValueError("Duplicate fact labels")
        if self.layer == "retrieval" and not self.query:
            raise ValueError("Retrieval cases require a query")
        if self.layer == "agent" and not self.should_abstain and not self.facts:
            raise ValueError("Answer cases require independent objective labels")
        if self.should_abstain and self.facts:
            raise ValueError("Abstention cases cannot require factual answers")
        if any(
            not set(fact.support_ids) <= set(self.expected_ids) for fact in self.facts
        ):
            raise ValueError("Every fact must name expected supporting records")
        return self
