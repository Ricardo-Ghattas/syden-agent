import pytest

from app.agents.config import AgentConfig
from app.agents.schemas import AgentError


def test_environment_limits_are_validated(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "4")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "7")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "12.5")
    config = AgentConfig.from_env()
    assert config.max_iterations == 4
    assert config.max_tool_calls == 7
    assert config.timeout_seconds == 12.5


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AGENT_MAX_ITERATIONS", "0"),
        ("AGENT_MAX_TOOL_CALLS", "-1"),
        ("AGENT_TIMEOUT_SECONDS", "nan"),
        ("AGENT_REPEATED_CALL_LIMIT", "0"),
        ("AGENT_MAX_TOOL_FAILURES", "0"),
        ("AGENT_MAX_TOOL_OUTPUT_BYTES", "100"),
        ("AGENT_MAX_SESSION_TURNS", "0"),
        ("AGENT_MAX_SESSION_CONTEXT_BYTES", "4095"),
        ("AGENT_MAX_ITERATIONS", "not a number"),
    ],
)
def test_invalid_environment_budget_fails_before_execution(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(AgentError, match="execution limit configuration"):
        AgentConfig.from_env()
