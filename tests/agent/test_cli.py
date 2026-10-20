import json
import sys
from unittest.mock import MagicMock

import pytest

from app.agents.__main__ import display, main
from app.agents.schemas import AgentResponse, SourceReference


def test_display_groups_matching_locations_but_json_keeps_record_citations(
    index, capsys
):
    from app.tools.registry import reference

    records = [
        entry.record
        for entry in index.entries.values()
        if entry.record.source.format == "pdf" and entry.record.source.page is not None
    ]
    first = reference(records[0])
    same_page = first.model_copy(update={"record_id": "another-record"})
    other_page = first.model_copy(
        update={"record_id": "different-page", "page": first.page + 1}
    )
    response = AgentResponse(
        answer="Summary",
        answer_type="document",
        sources=[first, same_page, other_page],
        warnings=[],
    )
    display(response)
    assert capsys.readouterr().out.count("Source:") == 2
    display(response, as_json=True)
    assert len(json.loads(capsys.readouterr().out)["sources"]) == 3


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


def test_root_cli_chat_delegates_to_interactive_command(monkeypatch):
    from main import main as root_main

    command = MagicMock()
    monkeypatch.setattr("app.agents.__main__.main", command)
    root_main(["chat", "--debug"])
    command.assert_called_once_with(["--chat", "--debug"])


def test_chat_followups_clear_new_and_exit(monkeypatch, index, llm, capsys):
    from app.agents.__main__ import chat
    from app.agents.config import AgentConfig
    from app.agents.state import ConversationTurn

    questions = iter(
        ["First", "Follow-up", "/clear", "After clear", "/new", "After new", "/exit"]
    )
    calls = []

    def ask(question, index, llm, *, session, config):
        calls.append((question, session.session_id, len(session.history)))
        response = AgentResponse(
            answer="Answer", answer_type="general", sources=[], warnings=[]
        )
        session.history.append(ConversationTurn(question=question, response=response))
        return response

    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))
    monkeypatch.setattr("app.agents.__main__.ask", ask)
    chat(index, llm, AgentConfig())
    assert [call[2] for call in calls] == [0, 1, 0, 0]
    assert calls[0][1] == calls[1][1] == calls[2][1]
    assert calls[3][1] != calls[0][1]
    output = capsys.readouterr().out
    assert "Conversation cleared" in output
    assert "New conversation" in output


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_chat_input_interruptions_are_graceful(monkeypatch, index, llm, capsys, error):
    from app.agents.__main__ import chat
    from app.agents.config import AgentConfig

    def interrupt(prompt):
        raise error()

    monkeypatch.setattr("builtins.input", interrupt)
    chat(index, llm, AgentConfig())
    assert "Conversation ended" in capsys.readouterr().out
