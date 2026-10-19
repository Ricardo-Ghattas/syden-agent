"""Allowlisted read-only retrieval with a per-question evidence budget."""

import json
import logging
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.agents.schemas import SourceReference
from app.ingestion.schemas import Record
from app.retrieval import DocumentIndex, search_documents

from .schemas import ARGUMENT_MODELS, GetArguments, ListArguments, SearchArguments

logger = logging.getLogger(__name__)


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def reference(record: Record) -> SourceReference:
    source = record.source
    return SourceReference(
        document_id=record.document_id,
        record_id=record.id,
        filename=source.filename,
        page=source.page,
        sheet=source.sheet,
        row_start=source.row_start,
        row_end=source.row_end,
        cell_range=source.cell_range,
    )


class ToolRegistry:
    """Request-local dispatcher; the shared index is never mutated by tools."""

    def __init__(self, index: DocumentIndex, max_output_bytes: int = 60000):
        if max_output_bytes < 2048:
            raise ValueError("Tool output budget must be at least 2048 bytes")
        self.index = index
        self.remaining_bytes = max_output_bytes
        self.sources: list[SourceReference] = []
        self.warnings: list[str] = []
        self.handlers = {
            "search_documents": self.search,
            "get_record": self.get,
            "list_documents": self.list,
        }

    def execute(self, name: str, arguments: str) -> str:
        started = perf_counter()
        success = False
        try:
            if name not in self.handlers:
                return self.error(
                    "unknown_tool", "Only registered retrieval tools are allowed."
                )
            if len(arguments) > 16000:
                return self.error(
                    "invalid_arguments", "Tool arguments exceed the size limit."
                )
            try:
                args = ARGUMENT_MODELS[name].model_validate_json(arguments)
            except ValidationError:
                return self.error(
                    "invalid_arguments",
                    "Arguments must match the tool's strict JSON schema.",
                )
            data, sources = self.handlers[name](args)
            output = encode({"ok": True, "data": data})
            size = len(output.encode("utf-8"))
            if size > self.remaining_bytes:
                return self.error(
                    "context_limit",
                    "Result exceeds the remaining evidence budget; narrow the query or document scope.",
                )
            self.remaining_bytes -= size
            self.sources.extend(sources)
            self.warnings.extend(data.get("warnings", []))
            success = True
            return output
        except (
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
            LookupError,
            ArithmeticError,
        ) as error:
            # Provider-bound errors must not disclose paths, document payloads, or exception text.
            logger.warning("Retrieval failed error_type=%s", type(error).__name__)
            return self.error(
                "retrieval_failed",
                "Retrieval failed; no evidence from this call was supplied.",
            )
        finally:
            logger.info(
                "tool=%s success=%s duration_ms=%.1f",
                name if name in self.handlers else "unknown",
                success,
                (perf_counter() - started) * 1000,
            )

    def error(self, code: str, message: str) -> str:
        self.warnings.append(f"{code}: {message}")
        return encode({"ok": False, "error": {"code": code, "message": message}})

    def index_warnings(self) -> list[str]:
        return (
            [
                "The document index is incomplete; some files or records could not be loaded."
            ]
            if self.index.report.issues
            else []
        )

    def search(self, args) -> tuple[dict[str, Any], list[SourceReference]]:
        assert isinstance(args, SearchArguments)
        results = []
        sources = []
        offset = args.offset
        warnings = self.index_warnings()
        data: dict[str, Any] = {
            "query": args.query,
            "filters": args.filters.model_dump(mode="json") if args.filters else None,
            "offset": offset,
            "total": 0,
            "returned": 0,
            "has_more": False,
            "next_offset": None,
            "index_complete": not self.index.report.issues,
            "complete": False,
            "warnings": warnings,
            "results": results,
        }
        stopped = False
        while True:
            page = search_documents(
                self.index, args.query, args.filters, args.limit, offset
            )
            data["total"] = page.total
            for result in page.results:
                record = result.record
                citation = reference(record)
                projected = record.model_dump(
                    mode="json", exclude={"metadata", "cells"}
                )
                projected["cells"] = [
                    cell.model_dump(
                        mode="json",
                        include={
                            "coordinate",
                            "value",
                            "formula",
                            "cached_value",
                            "value_type",
                            "number_format",
                            "currency_or_unit",
                        },
                    )
                    for cell in record.cells
                ]
                entry = {
                    "record_id": result.record_id,
                    "record": projected,
                    "citation": citation.model_dump(mode="json"),
                }
                results.append(entry)
                # Reserve space for completion flags and warnings; never cut a record's values.
                if len(encode(data).encode("utf-8")) + 1024 > self.remaining_bytes:
                    results.pop()
                    stopped = True
                    break
                sources.append(citation)
            if stopped or not args.exhaustive or not page.has_more:
                break
            assert page.next_offset is not None
            offset = page.next_offset
        data["returned"] = len(results)
        next_offset = args.offset + len(results)
        data["has_more"] = next_offset < data["total"]
        data["next_offset"] = next_offset if data["has_more"] else None
        data["complete"] = (
            args.offset == 0 and not data["has_more"] and data["index_complete"]
        )
        if stopped:
            warnings.append(
                "Evidence budget reached; whole records were omitted. Narrow the query before calculating totals."
            )
        if not data["complete"]:
            warnings.append(
                "Search evidence is incomplete; do not treat these results as a complete BOQ or verified total."
            )
        return data, sources

    def get(self, args) -> tuple[dict[str, Any], list[SourceReference]]:
        assert isinstance(args, GetArguments)
        record = self.index.get_record(args.record_id)
        citation = reference(record) if record else None
        return {
            "record": record.model_dump(mode="json") if record else None,
            "citation": citation.model_dump(mode="json") if citation else None,
            "found": record is not None,
            "index_complete": not self.index.report.issues,
            "warnings": self.index_warnings(),
        }, [citation] if citation else []

    def list(self, args) -> tuple[dict[str, Any], list[SourceReference]]:
        assert isinstance(args, ListArguments)
        documents = self.index.list_documents()
        citations = [
            SourceReference(
                document_id=doc.document_id,
                record_id=None,
                filename=doc.filename,
                page=None,
                sheet=None,
                row_start=None,
                row_end=None,
                cell_range=None,
            )
            for doc in documents
        ]
        return {
            "documents": [
                {
                    **doc.model_dump(mode="json"),
                    "citation": citation.model_dump(mode="json"),
                }
                for doc, citation in zip(documents, citations, strict=True)
            ],
            "index_complete": not self.index.report.issues,
            "warnings": self.index_warnings(),
        }, citations
