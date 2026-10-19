"""Document processing entry point; independent of agent and retrieval code."""

import argparse
from pathlib import Path

from app.ingestion.extractors.excel import SpreadsheetConversionError
from app.ingestion.pipeline import process_document


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF and XLSX documents into validated JSONL."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    for path in args.files:
        try:
            output = process_document(path, args.data_dir)
        except (OSError, ValueError, RuntimeError, SpreadsheetConversionError) as error:
            parser.exit(1, f"{path}: {error}\n")
        print(output)


if __name__ == "__main__":
    main()
