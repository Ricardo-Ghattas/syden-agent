"""Document ingestion, agent interaction, and evaluation commands."""

import argparse
import sys
from pathlib import Path

from app.ingestion.extractors.excel import SpreadsheetConversionError
from app.ingestion.pipeline import process_document


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "eval":
        from evals.runner import main as eval_main

        eval_main(argv[1:])
        return
    if argv and argv[0] in {"ask", "chat"}:
        from app.agents.__main__ import main as ask_main

        ask_main(argv[1:] if argv[0] == "ask" else ["--chat", *argv[1:]])
        return
    parser = argparse.ArgumentParser(
        description="Convert PDF/XLSX into JSONL; use 'ask QUESTION' or 'chat' to query, or 'eval' to evaluate."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    for path in args.files:
        try:
            output = process_document(path, args.data_dir)
        except (OSError, ValueError, RuntimeError, SpreadsheetConversionError) as error:
            parser.exit(1, f"{path}: {error}\n")
        print(output)


if __name__ == "__main__":
    main()
