import json
import sys
from unittest.mock import MagicMock

import pytest

from app.agents.__main__ import main
from app.agents.schemas import AgentResponse, SourceReference


def test_cli_answer_sources_json_and_close(monkeypatch, capsys):
    source = SourceReference(
        document_id="b" * 64,
        record_id="b" * 64 + ":39",
        filename="boq.xlsx",
        page=None,
        sheet="Structural",
        row_start=38,
        row_end=38,
        cell_range="A38:F38",
    )
    result = AgentResponse(
        answer="Quantity is -2.5.",
        answer_type="document",
        sources=[source],
        warnings=["Partial evidence"],
    )
    llm = MagicMock()
    monkeypatch.setattr("app.agents.__main__.Settings.from_env", MagicMock())
    monkeypatch.setattr("app.agents.__main__.LLM", MagicMock(return_value=llm))
    monkeypatch.setattr("app.agents.__main__.ask", MagicMock(return_value=result))
    monkeypatch.setattr(sys, "argv", ["syden", "Question", "--debug"])
    main()
    text = capsys.readouterr().out
    assert "Quantity is -2.5." in text
    assert "Source: boq.xlsx, sheet Structural, rows 38–38" in text
    assert "Warning: Partial evidence" in text
    llm.client.close.assert_called_once()
    monkeypatch.setattr(sys, "argv", ["syden", "Question", "--json"])
    main()
    assert json.loads(capsys.readouterr().out) == result.model_dump(mode="json")


def test_cli_missing_configuration_is_clear(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["syden", "Question"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_root_cli_delegates_ask_only(monkeypatch):
    from main import main as root_main

    command = MagicMock()
    monkeypatch.setattr("app.agents.__main__.main", command)
    root_main(["ask", "Question", "--json"])
    command.assert_called_once_with(["Question", "--json"])
