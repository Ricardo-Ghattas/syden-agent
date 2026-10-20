"""Single questions and in-memory interactive conversations over one shared client."""

import argparse
import logging

from pydantic import ValidationError
from rich.console import Console
from rich.text import Text

from app.retrieval import DocumentIndex

from .agent import ask
from .config import AgentConfig
from .llm import LLM, Settings
from .rendering import render_response
from .schemas import AgentError, AgentResponse, AgentStatus
from .state import ConversationSession


def display(
    response: AgentResponse, *, as_json: bool = False, debug: bool = False
) -> None:
    if as_json:
        print(response.model_dump_json(indent=2))
        return
    render_response(response, debug=debug)


def chat(
    index: DocumentIndex,
    llm: LLM,
    config: AgentConfig,
    *,
    as_json: bool = False,
    debug: bool = False,
) -> None:
    session = ConversationSession()
    console = Console()
    if not as_json:
        console.print(Text("SYDEN", style="bold blue"))
        console.print(
            Text(
                "/clear clears memory · /new starts a conversation · /exit quits",
                style="dim",
            )
        )
        if debug:
            console.print(Text(f"Session {session.session_id}", style="dim"))
    while True:
        try:
            if not as_json:
                console.print(Text("You › ", style="bold blue"), end="")
            question = input("").strip()
        except (EOFError, KeyboardInterrupt):
            if not as_json:
                console.print(Text("Conversation ended.", style="dim"))
            break
        command = question.casefold()
        if command in {"/exit", "/quit"}:
            break
        if command == "/clear":
            session.clear()
            if not as_json:
                console.print(Text("Conversation cleared", style="dim"))
            continue
        if command == "/new":
            session = ConversationSession()
            if not as_json:
                console.print(Text("New conversation", style="dim"))
            continue
        if not question:
            continue
        try:
            response = ask(question, index, llm, session=session, config=config)
        except ValidationError:
            Console(stderr=True).print(
                Text("Invalid question; use between 1 and 16000 characters.")
            )
            continue
        display(response, as_json=as_json, debug=debug)
        if response.status == AgentStatus.CANCELLED:
            break


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Ask SYDEN about processed construction documents"
    )
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--chat", action="store_true", help="Start an in-memory conversation"
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--json", action="store_true", help="Print validated responses as JSON"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show execution status, session IDs, and sanitized execution events",
    )
    args = parser.parse_args(argv)
    if args.chat == (args.question is not None):
        parser.error("Provide a question or --chat, but not both")
    if args.debug:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("app.agents").setLevel(logging.INFO)
        logging.getLogger("app.tools.registry").setLevel(logging.INFO)
    try:
        settings = Settings.from_env()
        config = AgentConfig.from_env()
        llm = LLM(settings)
        try:
            index = DocumentIndex(args.processed_dir)
            if args.chat:
                chat(index, llm, config, as_json=args.json, debug=args.debug)
            else:
                response = ask(args.question, index, llm, config=config)
                display(response, as_json=args.json, debug=args.debug)
                if response.status in {
                    AgentStatus.FAILED,
                    AgentStatus.LIMIT_REACHED,
                    AgentStatus.CANCELLED,
                }:
                    parser.exit(1)
        finally:
            llm.client.close()
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    except (AgentError, ValidationError) as error:
        message = (
            str(error)
            if isinstance(error, AgentError)
            else "Invalid question or configuration."
        )
        parser.exit(1, f"{message}\n")


if __name__ == "__main__":
    main()
