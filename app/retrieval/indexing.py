"""Load validated JSONL once and build deterministic lexical lookup structures."""

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.ingestion.schemas import Record

from .schemas import DocumentInfo, IndexIssue, IndexReport
from .text import normalize_text, tokenize


def load_records(paths: Iterable[Path]) -> tuple[dict[str, Record], list[IndexIssue]]:
    """Keep valid lines; report rejected records and keep the first duplicate ID."""
    records = {}
    issues = []
    identities = {}
    for path in sorted(Path(path) for path in paths):
        try:
            with path.open("rb") as stream:
                for line_number, raw in enumerate(stream, 1):
                    if not raw.strip():
                        continue
                    try:
                        record = Record.model_validate_json(raw)
                    except ValidationError as error:
                        detail = error.errors(include_input=False, include_url=False)[0]
                        issues.append(
                            IndexIssue(
                                path=str(path),
                                line=line_number,
                                code="malformed_record",
                                message=f"{detail['loc']}: {detail['msg']}",
                            )
                        )
                        continue
                    if record.id in records:
                        issues.append(
                            IndexIssue(
                                path=str(path),
                                line=line_number,
                                code="duplicate_id",
                                message="Duplicate ID rejected; first record in sorted file/line order retained",
                                record_id=record.id,
                            )
                        )
                        continue
                    identity = (record.source.filename, record.source.format)
                    if (
                        record.document_id in identities
                        and identities[record.document_id] != identity
                    ):
                        issues.append(
                            IndexIssue(
                                path=str(path),
                                line=line_number,
                                code="inconsistent_document",
                                message="Records sharing a document ID must agree on filename and format",
                                record_id=record.id,
                            )
                        )
                        continue
                    identities[record.document_id] = identity
                    records[record.id] = record
        except OSError as error:
            issues.append(
                IndexIssue(
                    path=str(path),
                    code="missing_file"
                    if isinstance(error, FileNotFoundError)
                    else "read_error",
                    message=str(error),
                )
            )
    return records, issues


def _fields(record: Record) -> dict[str, str]:
    body = [record.text]
    body.extend(
        " | ".join("" if value is None else str(value) for value in row)
        for row in record.rows
    )
    body.extend(str(cell.value) for cell in record.cells if cell.value is not None)
    description = ""
    if record.boq is not None:
        body.extend(
            str(value)
            for value in record.boq.model_dump().values()
            if value is not None
        )
        if record.boq.description is not None:
            description = str(record.boq.description)
    return {
        "body": " ".join(
            dict.fromkeys(normalize_text(value) for value in body if value)
        ),
        "description": normalize_text(description),
        "heading": normalize_text(
            " ".join(
                [*record.semantic_path, record.text if record.kind == "heading" else ""]
            )
        ),
        "metadata": normalize_text(
            " ".join([record.source.filename, record.source.sheet or ""])
        ),
    }


@dataclass(frozen=True, slots=True)
class IndexedRecord:
    record: Record
    fields: dict[str, str]
    terms: dict[str, Counter[str]]
    lengths: dict[str, int]
    item_code: str | None


class DocumentIndex:
    """An explicit in-memory snapshot; call reload() after replacing processed files."""

    def __init__(self, processed_dir: str | Path = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.reload()

    def reload(self) -> IndexReport:
        paths = []
        issues = []
        try:
            paths = sorted(
                path
                for path in self.processed_dir.iterdir()
                if path.suffix.casefold() == ".jsonl" and path.is_file()
            )
        except OSError as error:
            issues.append(
                IndexIssue(
                    path=str(self.processed_dir),
                    code="missing_directory"
                    if isinstance(error, FileNotFoundError)
                    else "read_error",
                    message=str(error),
                )
            )
        records, load_issues = load_records(paths)
        issues.extend(load_issues)
        entries = {}
        postings = defaultdict(set)
        item_codes = defaultdict(set)
        frequencies = {
            field: Counter() for field in ("body", "description", "heading", "metadata")
        }
        totals = Counter()
        documents = {}
        for record in records.values():
            fields = _fields(record)
            terms = {field: Counter(tokenize(text)) for field, text in fields.items()}
            lengths = {field: sum(counts.values()) for field, counts in terms.items()}
            item_code = (
                normalize_text(str(record.boq.item))
                if record.boq is not None and record.boq.item is not None
                else None
            )
            entries[record.id] = IndexedRecord(
                record, fields, terms, lengths, item_code
            )
            for field, counts in terms.items():
                frequencies[field].update(counts.keys())
                totals[field] += lengths[field]
                for term in counts:
                    postings[term].add(record.id)
            if item_code:
                item_codes[item_code].add(record.id)
            if record.document_id not in documents:
                documents[record.document_id] = DocumentInfo(
                    document_id=record.document_id,
                    filename=record.source.filename,
                    format=record.source.format,
                    record_count=1,
                )
            else:
                documents[record.document_id].record_count += 1
            document = documents[record.document_id]
            if record.source.sheet is not None:
                document.sheets.append(record.source.sheet)
            document.sections.extend(record.semantic_path)
            document.warnings.extend(record.warnings)
        for document in documents.values():
            document.sheets = sorted(set(document.sheets))
            document.sections = sorted(set(document.sections))
            document.warnings = sorted(set(document.warnings))
        self.entries = entries
        self.postings = dict(postings)
        self.vocabulary = sorted(postings)
        self.item_codes = dict(item_codes)
        self.frequencies = frequencies
        self.average_lengths = {
            field: totals[field] / max(1, len(entries)) for field in frequencies
        }
        self.documents = documents
        self.report = IndexReport(
            files=len(paths),
            records=len(entries),
            documents=len(documents),
            issues=issues,
        )
        return self.report.model_copy(deep=True)

    def expand_term(self, term: str) -> tuple[str, ...]:
        """Expand alphabetic prefixes only; numeric codes must not match longer codes."""
        matches = [term] if term in self.postings else []
        if len(term) >= 3 and term.isalpha():
            position = bisect_left(self.vocabulary, term)
            while position < len(self.vocabulary) and self.vocabulary[
                position
            ].startswith(term):
                value = self.vocabulary[position]
                if value != term:
                    matches.append(value)
                position += 1
        return tuple(matches)

    def get_record(self, record_id: str) -> Record | None:
        """Return a complete independent record, or None when the ID is absent."""
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Record ID must be a non-empty string")
        entry = self.entries.get(record_id)
        return entry.record.model_copy(deep=True) if entry else None

    def list_documents(self) -> list[DocumentInfo]:
        return [
            self.documents[key].model_copy(deep=True) for key in sorted(self.documents)
        ]
