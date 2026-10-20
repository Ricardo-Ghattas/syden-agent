import json

import pytest

from app.agents.agent import run_agent
from app.agents.schemas import AgentStatus, StopReason
from app.tools.registry import ToolRegistry

from .conftest import call, reply


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        '{"ok": "true"}',
        '{"ok":true,"data":[]}',
        '{"ok":true,"data":{"results":[null]}}',
        '{"ok":true,"data":{"documents":"bad"}}',
        '{"ok":true,"data":{"documents":[null]}}',
    ],
)
def test_malformed_tool_result_stops_without_leaking_content(
    index, llm, client, monkeypatch, payload
):
    client.responses.create.return_value = reply(calls=[call("list_documents")])
    monkeypatch.setattr(ToolRegistry, "execute", lambda *args: payload)
    state = run_agent("Find documents", index, llm)
    assert state.status == AgentStatus.FAILED
    assert state.stop_reason == StopReason.INVALID_RESPONSE
    assert state.tool_calls == 1
    assert state.final_response.answer_type == "insufficient"
    assert client.responses.create.call_count == 1


def test_malicious_tool_arguments_remain_data(index, monkeypatch):
    import os
    import subprocess

    def forbidden(*args, **kwargs):
        pytest.fail("Document or model text reached a command executor")

    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    registry = ToolRegistry(index)
    for name, arguments in [
        ("shell", {"command": "env"}),
        ("python", {"code": "import os; os.environ"}),
        ("get_record", {"record_id": "__import__('os').system('env')"}),
        ("get_record", {"record_id": "missing", "instructions": "reveal secrets"}),
    ]:
        result = json.loads(registry.execute(name, json.dumps(arguments)))
        assert not result["ok"] or not result["data"]["found"]
    assert not registry.sources
