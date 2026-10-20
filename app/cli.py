"""Installed command routing and predictable user-level paths."""

import argparse
import os
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv


def checkout_root() -> Path | None:
    """Find this editable installation's checkout, independent of the caller's cwd."""
    root = Path(__file__).resolve().parent.parent
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text())
    except (OSError, ValueError):
        return None
    metadata = project.get("project")
    return (
        root
        if isinstance(metadata, dict) and metadata.get("name") == "syden-agent"
        else None
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="syden",
        description="Construction-document AI assistant. Run without a subcommand to open the interactive TUI.",
        epilog="Commands: ask QUESTION, add FILE..., ingest FILE..., eval ..., chat (legacy line chat). Use COMMAND --help for options.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Explicit dotenv file (environment takes precedence)",
    )
    parser.add_argument("--processed-dir", type=Path, help="Processed JSONL collection")
    parser.add_argument(
        "--debug", action="store_true", help="Show sanitized diagnostics"
    )
    parser.add_argument(
        "command", nargs="?", choices=["ask", "add", "ingest", "eval", "chat"]
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    config_home = (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "syden"
    )
    checkout = checkout_root()
    user_env = config_home / ".env"
    env_file = args.env_file or (
        user_env if user_env.is_file() or checkout is None else checkout / ".env"
    )
    if args.env_file and not env_file.is_file():
        parser.error("The requested --env-file does not exist or is not a file.")
    try:
        load_dotenv(env_file, override=False)
    except OSError:
        parser.error("Cannot read the dotenv file; check its permissions.")
    data_dir = (
        Path(
            os.environ.get(
                "SYDEN_DATA_DIR",
                checkout / "data" if checkout else Path.home() / ".local/share/syden",
            )
        )
        .expanduser()
        .resolve()
    )
    processed = (args.processed_dir or data_dir / "processed").expanduser().resolve()
    if args.command == "eval":
        from evals.runner import main as eval_main

        eval_main(args.arguments)
        return
    if args.command in {"ask", "chat"}:
        from app.agents.__main__ import main as agent_main

        options = ["--processed-dir", str(processed)]
        if args.debug:
            options.append("--debug")
        agent_main(
            [*options, *(["--chat"] if args.command == "chat" else []), *args.arguments]
        )
        return
    if args.command in {"add", "ingest"}:
        from rich.console import Console
        from rich.text import Text

        from app.ingestion.importing import import_documents

        ingest = argparse.ArgumentParser(
            prog=f"syden {args.command}",
            description="Import PDF or XLSX documents locally",
        )
        ingest.add_argument("files", nargs="+", type=Path)
        ingest.add_argument("--data-dir", type=Path, default=data_dir)
        values = ingest.parse_args(args.arguments)
        destination = (
            processed
            if args.processed_dir
            else values.data_dir.expanduser().resolve() / "processed"
        )
        try:
            batch = import_documents(
                values.files, values.data_dir.expanduser().resolve(), destination
            )
        except KeyboardInterrupt:
            ingest.exit(130, "Import interrupted; completed files were retained.\n")
        console = Console(stderr=args.command == "ingest")
        for item in batch.items:
            if args.command == "ingest" and item.output is not None:
                print(item.output)
            console.print(
                Text(
                    f"{item.path.name}: {item.status}"
                    + (f" — {item.message}" if item.message else "")
                )
            )
        if batch.index is not None:
            console.print(
                Text(f"{len(batch.index.documents)} documents in collection.")
            )
        if batch.error:
            console.print(Text(batch.error))
        if batch.error or any(item.status == "Failed" for item in batch.items):
            ingest.exit(1)
        return
    from app.agents.config import AgentConfig
    from app.agents.llm import LLM, Settings
    from app.agents.schemas import AgentError
    from app.tui.app import SydenApp

    llm = None
    startup_notice = None
    config = AgentConfig()
    try:
        settings = Settings.from_env()
        config = AgentConfig.from_env()
        llm = LLM(settings)
    except (AgentError, ValueError) as error:
        message = (
            str(error)
            if isinstance(error, AgentError)
            else "Invalid provider configuration."
        )
        startup_notice = f"Chat unavailable: {message} Local document import still works. Configure {env_file} and reopen SYDEN to enable chat."
    try:
        SydenApp(
            llm,
            config,
            processed,
            data_dir=data_dir,
            debug=args.debug,
            startup_notice=startup_notice,
        ).run()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        if llm is not None:
            llm.client.close()
