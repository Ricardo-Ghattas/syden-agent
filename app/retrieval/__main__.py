"""Inspect and query processed documents without an LLM."""

import argparse
import json

from .indexing import DocumentIndex
from .search import search_documents


def main():
    parser = argparse.ArgumentParser(
        description="Search SYDEN's processed JSONL documents"
    )
    parser.add_argument("--processed-dir", default="data/processed")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser(
        "search", help="Search with optional JSON metadata filters"
    )
    search.add_argument("query")
    search.add_argument("--filters", default="{}")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--offset", type=int, default=0)
    commands.add_parser("list", help="List indexed documents, sections, and worksheets")
    record = commands.add_parser("get", help="Get the complete structured record by ID")
    record.add_argument("record_id")
    args = parser.parse_args()
    index = DocumentIndex(args.processed_dir)
    try:
        if args.command == "search":
            result = search_documents(
                index, args.query, json.loads(args.filters), args.limit, args.offset
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "list":
            print(
                json.dumps(
                    {
                        "index": index.report.model_dump(mode="json"),
                        "documents": [
                            doc.model_dump(mode="json")
                            for doc in index.list_documents()
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            result = index.get_record(args.record_id)
            print(
                json.dumps(
                    {
                        "index": index.report.model_dump(mode="json"),
                        "record": result.model_dump(mode="json") if result else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            if result is None:
                parser.exit(1)
    except ValueError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
