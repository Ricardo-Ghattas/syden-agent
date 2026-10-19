"""Single-question CLI; no persistent conversation or implicit live test calls."""

import argparse
import logging

from pydantic import ValidationError

from app.retrieval import DocumentIndex

from .agent import ask
from .llm import LLM, Settings
from .schemas import AgentError


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Ask SYDEN about processed construction documents"
    )
    parser.add_argument("question")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--json", action="store_true", help="Print the validated response as JSON"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log model/status/usage and tool names/durations, without evidence contents",
    )
    args = parser.parse_args(argv)
    if args.debug:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("app.agents").setLevel(logging.INFO)
        logging.getLogger("app.tools.registry").setLevel(logging.INFO)
    try:
        settings = Settings.from_env()
        llm = LLM(settings)
        try:
            response = ask(args.question, DocumentIndex(args.processed_dir), llm)
        finally:
            llm.client.close()
    except (AgentError, ValidationError) as error:
        message = (
            str(error)
            if isinstance(error, AgentError)
            else "Invalid question or configuration."
        )
        parser.exit(1, f"{message}\n")
    if args.json:
        print(response.model_dump_json(indent=2))
        return
    print(response.answer)
    for source in response.sources:
        location = [source.filename]
        if source.page is not None:
            location.append(f"page {source.page}")
        if source.sheet is not None:
            location.append(f"sheet {source.sheet}")
        if source.row_start is not None:
            location.append(f"rows {source.row_start}–{source.row_end}")
        print("Source: " + ", ".join(location))
    for warning in response.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
