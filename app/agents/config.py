"""Immutable execution budgets, separate from provider settings and mutable state."""

import os

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schemas import AgentError


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    max_iterations: int = Field(default=8, ge=1, le=64)
    max_tool_calls: int = Field(default=20, ge=1, le=200)
    timeout_seconds: float = Field(default=120, gt=0, le=3600)
    repeated_call_limit: int = Field(default=2, ge=1, le=10)
    max_tool_failures: int = Field(default=3, ge=1, le=20)
    max_tool_output_bytes: int = Field(default=60000, ge=2048, le=1000000)
    max_session_turns: int = Field(default=12, ge=1, le=100)
    max_session_context_bytes: int = Field(default=240000, ge=4096, le=4000000)

    @classmethod
    def from_env(cls):
        values = {
            name: os.environ[f"AGENT_{name.upper()}"]
            for name in cls.model_fields
            if f"AGENT_{name.upper()}" in os.environ
        }
        try:
            return cls.model_validate(values)
        except ValidationError as error:
            raise AgentError(
                "Invalid AGENT_* execution limit configuration."
            ) from error
