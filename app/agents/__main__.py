"""Single questions and in-memory interactive conversations over one shared client."""

import argparse
import logging

from pydantic import ValidationError

from app.retrieval import DocumentIndex

from .agent import ask
from .config import AgentConfig
from .llm import LLM, Settings
from .schemas import AgentError, AgentResponse, AgentStatus
from .state import ConversationSession


def display(response: AgentResponse, *, as_json: bool = False) -> None:
    if as_json:
        print(response.model_dump_json(indent=2))
        return
    print(response.answer)
    displayed = set()
    for source in response.sources:
        location = [source.filename]
        if source.page is not None:
            location.append(f"page {source.page}")
        if source.sheet is not None:
            location.append(f"sheet {source.sheet}")
        if source.row_start is not None:
            location.append(f"rows {source.row_start}–{source.row_end}")
        label = ", ".join(location)
        key = (source.document_id, label)
        if key not in displayed:
            print("Source: " + label)
            displayed.add(key)
    for warning in response.warnings:
        print(f"Warning: {warning}")
    if response.stop_reason is not None:
        print(f"Status: {response.status.value}; reason: {response.stop_reason.value}")


def chat(
    index: DocumentIndex, llm: LLM, config: AgentConfig, *, as_json: bool = False
) -> None:
    session = ConversationSession()
    print(
        f"SYDEN — session {session.session_id}\n/clear clears memory; /new starts a conversation; /exit quits."
    )
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversation ended.")
            break
        command = question.casefold()
        if command in {"/exit", "/quit"}:
            break
        if command == "/clear":
            session.clear()
            print(f"Conversation cleared — session {session.session_id}")
            continue
        if command == "/new":
            session = ConversationSession()
            print(f"New conversation — session {session.session_id}")
            continue
        if not question:
            continue
        try:
            response = ask(question, index, llm, session=session, config=config)
        except ValidationError:
            print("Invalid question; use between 1 and 16000 characters.")
            continue
        display(response, as_json=as_json)
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
        help="Log sanitized execution events without evidence or private reasoning",
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
                chat(index, llm, config, as_json=args.json)
            else:
                response = ask(args.question, index, llm, config=config)
                display(response, as_json=args.json)
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
