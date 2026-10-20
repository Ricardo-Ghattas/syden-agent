import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.cli import checkout_root, main


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr("app.cli.checkout_root", lambda: None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("SYDEN_DATA_DIR", str(tmp_path / "collection"))


def test_installed_entry_point():
    from importlib.metadata import distribution

    entries = {
        entry.name: entry.value for entry in distribution("syden-agent").entry_points
    }
    assert entries["syden"] == "app.cli:main"


def test_tui_launch_and_client_closed_on_exit(monkeypatch, tmp_path):
    llm, tui = MagicMock(), MagicMock()
    monkeypatch.setattr("app.agents.llm.Settings.from_env", MagicMock())
    monkeypatch.setattr("app.agents.llm.LLM", MagicMock(return_value=llm))
    factory = MagicMock(return_value=tui)
    monkeypatch.setattr("app.tui.app.SydenApp", factory)
    main([])
    assert factory.call_args.args[2] == tmp_path / "collection/processed"
    tui.run.assert_called_once()
    llm.client.close.assert_called_once()


def test_explicit_env_file_and_collection_resolve_outside_repository(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    env = tmp_path / "local.env"
    env.write_text("OPENAI_MODEL=from-file\n")
    agent = MagicMock()
    monkeypatch.setattr("app.agents.__main__.main", agent)
    main(
        [
            "--env-file",
            str(env),
            "--processed-dir",
            "my-documents",
            "ask",
            "Quantity?",
            "--json",
        ]
    )
    assert agent.call_args.args[0] == [
        "--processed-dir",
        str(tmp_path / "my-documents"),
        "Quantity?",
        "--json",
    ]
    import os

    assert os.environ["OPENAI_MODEL"] == "from-file"
    monkeypatch.delenv("OPENAI_MODEL")


def test_user_config_environment_takes_precedence(monkeypatch, tmp_path):
    config = tmp_path / "config/syden/.env"
    config.parent.mkdir(parents=True)
    config.write_text("OPENAI_MODEL=from-file\n")
    monkeypatch.setenv("OPENAI_MODEL", "from-environment")
    agent = MagicMock()
    monkeypatch.setattr("app.agents.__main__.main", agent)
    main(["ask", "Question"])
    import os

    assert os.environ["OPENAI_MODEL"] == "from-environment"


def test_eval_and_ingest_routing(monkeypatch, tmp_path, capsys):
    evaluate = MagicMock()
    process = MagicMock(return_value=tmp_path / "collection/processed/result.jsonl")
    monkeypatch.setattr("evals.runner.main", evaluate)
    monkeypatch.setattr("app.ingestion.pipeline.process_document", process)
    main(["eval", "--mode", "offline"])
    evaluate.assert_called_once_with(["--mode", "offline"])
    main(["ingest", "input.pdf"])
    process.assert_called_once_with(Path("input.pdf"), tmp_path / "collection")
    assert "result.jsonl" in capsys.readouterr().out


def test_missing_key_and_invalid_config_are_safe(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_MODEL", "mock")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "invalid-sensitive-value")
    with pytest.raises(SystemExit):
        main([])
    text = capsys.readouterr().err
    assert "test-secret" not in text
    assert "invalid-sensitive-value" not in text


def test_noninteractive_json_is_unchanged(monkeypatch, tmp_path, capsys):
    from app.agents.schemas import AgentResponse

    response = AgentResponse(
        answer="**USD 12.50**",
        answer_type="general",
        sources=[],
        warnings=["Unverified"],
    )
    client = MagicMock()
    monkeypatch.setattr("app.agents.__main__.Settings.from_env", MagicMock())
    monkeypatch.setattr("app.agents.__main__.LLM", MagicMock(return_value=client))
    monkeypatch.setattr("app.agents.__main__.ask", MagicMock(return_value=response))
    main(["ask", "Amount?", "--json"])
    assert json.loads(capsys.readouterr().out) == response.model_dump(mode="json")
    client.client.close.assert_called_once()


def test_checkout_detection_validates_package_location_not_cwd(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "app").mkdir(parents=True)
    manifest = checkout / "pyproject.toml"
    manifest.write_text('[project]\nname = "syden-agent"\n')
    monkeypatch.setattr("app.cli.__file__", str(checkout / "app/cli.py"))
    monkeypatch.chdir(tmp_path)
    assert checkout_root() == checkout
    manifest.write_text('[project]\nname = "another-project"\n')
    assert checkout_root() is None
    manifest.write_text("invalid TOML =")
    assert checkout_root() is None
    manifest.unlink()
    assert checkout_root() is None


@pytest.mark.parametrize("outside_checkout", [False, True])
def test_plain_syden_reuses_checkout_env_and_collection(
    monkeypatch, tmp_path, outside_checkout
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".env").write_text(
        "OPENAI_API_KEY=test-unused\nOPENAI_MODEL=checkout-model\n"
    )
    monkeypatch.setattr("app.cli.checkout_root", lambda: checkout)
    monkeypatch.delenv("SYDEN_DATA_DIR")
    for name in ["OPENAI_API_KEY", "OPENAI_MODEL"]:
        monkeypatch.setenv(name, "temporary")
        monkeypatch.delenv(name)
    (tmp_path / ".env").write_text("OPENAI_MODEL=unrelated-cwd-model\n")
    monkeypatch.chdir(tmp_path if outside_checkout else checkout)
    llm, tui = MagicMock(), MagicMock()
    factory = MagicMock(return_value=tui)
    monkeypatch.setattr("app.agents.llm.LLM", MagicMock(return_value=llm))
    monkeypatch.setattr("app.tui.app.SydenApp", factory)
    main([])
    assert factory.call_args.args[2] == checkout / "data/processed"
    import os

    assert os.environ["OPENAI_MODEL"] == "checkout-model"
    tui.run.assert_called_once()
    llm.client.close.assert_called_once()


def test_explicit_and_user_config_override_checkout_env(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".env").write_text("OPENAI_MODEL=checkout-model\n")
    user_env = tmp_path / "config/syden/.env"
    user_env.parent.mkdir(parents=True)
    user_env.write_text("OPENAI_MODEL=user-model\n")
    explicit = tmp_path / "explicit.env"
    explicit.write_text("OPENAI_MODEL=explicit-model\n")
    monkeypatch.setattr("app.cli.checkout_root", lambda: checkout)
    monkeypatch.setattr("app.agents.__main__.main", MagicMock())
    import os

    for options, expected in [
        ([], "user-model"),
        (["--env-file", str(explicit)], "explicit-model"),
    ]:
        monkeypatch.setenv("OPENAI_MODEL", "temporary")
        monkeypatch.delenv("OPENAI_MODEL")
        main([*options, "ask", "Question"])
        assert os.environ["OPENAI_MODEL"] == expected
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    main(["--env-file", str(explicit), "ask", "Question"])
    assert os.environ["OPENAI_MODEL"] == "environment-model"


def test_explicit_collection_and_data_environment_override_checkout(
    monkeypatch, tmp_path
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr("app.cli.checkout_root", lambda: checkout)
    monkeypatch.chdir(tmp_path)
    agent = MagicMock()
    monkeypatch.setattr("app.agents.__main__.main", agent)
    main(["ask", "Question"])
    assert agent.call_args.args[0][1] == str(tmp_path / "collection/processed")
    main(["--processed-dir", "explicit-collection", "ask", "Question"])
    assert agent.call_args.args[0][1] == str(tmp_path / "explicit-collection")


def test_standalone_install_keeps_user_data_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SYDEN_DATA_DIR")
    agent = MagicMock()
    monkeypatch.setattr("app.agents.__main__.main", agent)
    main(["ask", "Question"])
    assert agent.call_args.args[0][1] == str(
        Path.home() / ".local/share/syden/processed"
    )
