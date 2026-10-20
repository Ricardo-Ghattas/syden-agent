"""Allowlisted read-only retrieval with a per-question evidence budget."""

import hashlib
import json
import logging
from time import monotonic, perf_counter
from typing import Any

from pydantic import ValidationError

from app.agents.schemas import SourceReference
from app.ingestion.schemas import Record
from app.retrieval import DocumentIndex, search_documents
from app.retrieval.schemas import SearchFilters

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

    def __init__(
        self,
        index: DocumentIndex,
        max_output_bytes: int = 60000,
        *,
        deadline: float | None = None,
    ):
        if max_output_bytes < 2048:
            raise ValueError("Tool output budget must be at least 2048 bytes")
        self.index = index
        self.remaining_bytes = max_output_bytes
        self.deadline = deadline
        self.evidence: dict[str, dict] = {}
        self.representation_hashes: set[str] = set()
        self.result_fingerprint = ""
        self._pending_records: dict[str, dict] = {}
        self._pending_hashes: set[str] = set()
        self.sources: list[SourceReference] = []
        self.supplied_this_run: list[SourceReference] = []
        self.warnings: list[str] = []
        self.handlers = {
            "search_documents": self.search,
            "get_record": self.get,
            "list_documents": self.list,
        }

    def execute(self, name: str, arguments: str) -> str:
        started = perf_counter()
        success = False
        self._pending_records = {}
        self._pending_hashes = set()
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
            except ValidationError as error:
                known_fields = set(ARGUMENT_MODELS[name].model_fields) | set(
                    SearchFilters.model_fields
                )
                problems = []
                for issue in error.errors(
                    include_input=False, include_context=False, include_url=False
                )[:5]:
                    field = (
                        ".".join(
                            str(part) if part in known_fields else "?"
                            for part in issue["loc"]
                        )
                        or "arguments"
                    )
                    message = (
                        issue["msg"]
                        if issue["type"] != "json_invalid"
                        else "Provide a valid JSON object"
                    )
                    problems.append(f"{field}: {message}")
                return self.error(
                    "invalid_arguments",
                    "Invalid arguments: "
                    + "; ".join(problems)
                    + ". Use JSON null for unused filters. Correct these fields before retrying.",
                )
            self.check_deadline()
            data, sources = self.handlers[name](args)
            self.check_deadline()
            output = encode({"ok": True, "data": data})
            size = len(output.encode("utf-8"))
            if size > self.remaining_bytes:
                return self.error(
                    "context_limit",
                    "Result exceeds the remaining evidence budget; narrow the query or document scope.",
                )
            self.remaining_bytes -= size
            comparable = dict(data)
            comparable.pop("query", None)
            comparable.pop("filters", None)
            if isinstance(data.get("record"), dict):
                item = data["record"]
                comparable["record"] = self._pending_records.get(
                    item.get("id", item.get("record_id")), item
                )
            if "results" in data:
                comparable["results"] = [
                    {**item, "record": self._pending_records[item["record_id"]]}
                    for item in data["results"]
                ]
            self.result_fingerprint = hashlib.sha256(
                encode(comparable).encode()
            ).hexdigest()
            self.evidence.update(self._pending_records)
            self.representation_hashes.update(self._pending_hashes)
            self.sources.extend(
                source for source in sources if source not in self.sources
            )
            self.supplied_this_run.extend(
                source for source in sources if source not in self.supplied_this_run
            )
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
        output = encode({"ok": False, "error": {"code": code, "message": message}})
        self.result_fingerprint = hashlib.sha256(output.encode()).hexdigest()
        return output

    def check_deadline(self) -> None:
        if self.deadline is not None and monotonic() >= self.deadline:
            raise TimeoutError("Retrieval deadline reached")

    def compact_record(self, record: dict) -> dict:
        digest = hashlib.sha256(encode(record).encode()).hexdigest()
        self._pending_records[record["id"]] = record
        self._pending_hashes.add(digest)
        if digest in self.representation_hashes:
            return {
                "already_supplied": True,
                "record_id": record["id"],
                "representation_sha256": digest,
            }
        return record

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
            self.check_deadline()
            page = search_documents(
                self.index, args.query, args.filters, args.limit, offset
            )
            data["total"] = page.total
            for result in page.results:
                self.check_deadline()
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
                    "record": self.compact_record(projected),
                    "citation": citation.model_dump(mode="json"),
                }
                results.append(entry)
                # Reserve space for completion flags and warnings; never cut a record's values.
                if len(encode(data).encode("utf-8")) + 1024 > self.remaining_bytes:
                    results.pop()
                    self._pending_records.pop(record.id, None)
                    self._pending_hashes.discard(
                        hashlib.sha256(encode(projected).encode()).hexdigest()
                    )
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
            "record": self.compact_record(record.model_dump(mode="json"))
            if record
            else None,
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
