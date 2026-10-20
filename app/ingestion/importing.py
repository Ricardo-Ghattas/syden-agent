"""Serial, collection-locked imports using the existing atomic document processor."""

import fcntl
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from app.retrieval import DocumentIndex

from .pipeline import process_document
from .schemas import Record
from .storage import source_sha256

MAX_IMPORT_FILES = 64
SUPPORTED_SUFFIXES = {".pdf", ".xlsx"}


@dataclass(frozen=True)
class ImportItem:
    path: Path
    status: str = "Queued"
    message: str = ""
    document_id: str | None = None
    output: Path | None = None


@dataclass
class ImportBatch:
    items: list[ImportItem] = field(default_factory=list)
    index: DocumentIndex | None = None
    error: str | None = None


def import_documents(
    paths: Sequence[Path],
    data_dir: Path,
    processed_dir: Path,
    *,
    cancelled: Event | None = None,
    on_progress: Callable[[int, ImportItem], None] | None = None,
) -> ImportBatch:
    if not 1 <= len(paths) <= MAX_IMPORT_FILES:
        return ImportBatch(error=f"Select between 1 and {MAX_IMPORT_FILES} files.")
    batch = ImportBatch([ImportItem(Path(path)) for path in paths])

    def update(position, status, message="", document_id=None, output=None):
        item = ImportItem(
            batch.items[position].path, status, message, document_id, output
        )
        batch.items[position] = item
        if on_progress:
            on_progress(position, item)

    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
        with (processed_dir / ".syden-import.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                batch.error = "Another import is using this collection. Try again when it finishes."
                for position in range(len(batch.items)):
                    update(position, "Failed", "Collection locked; try again.")
                return batch
            current = DocumentIndex(processed_dir)
            known = (
                {doc.document_id: doc.format for doc in current.list_documents()}
                if not current.report.issues
                else {}
            )
            for position, item in enumerate(batch.items):
                if cancelled is not None and cancelled.is_set():
                    update(position, "Cancelled", "Not processed")
                    continue
                update(position, "Processing")
                try:
                    path = item.path.expanduser().resolve(strict=True)
                    if not path.is_file():
                        update(position, "Failed", "Select a regular file.")
                        continue
                    if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
                        update(
                            position, "Failed", "Only PDF and XLSX files are supported."
                        )
                        continue
                    digest = source_sha256(path)
                    if known.get(digest) == path.suffix.casefold()[1:]:
                        update(
                            position,
                            "Duplicate",
                            "Previously imported; identical content.",
                            digest,
                        )
                        continue
                    output = process_document(
                        path, data_dir, processed_dir=processed_dir
                    )
                    # Use the processor's archived identity, even if the input changed during import.
                    with output.open() as stream:
                        root = Record.model_validate_json(stream.readline())
                    known[root.document_id] = root.source.format
                    update(
                        position, "Ready", "Imported locally", root.document_id, output
                    )
                except (OSError, ValueError, RuntimeError):
                    update(
                        position,
                        "Failed",
                        "Cannot import; check the file, permissions, or document format.",
                    )
                except Exception:  # noqa: BLE001 - report unexpected extractor failures without confidential payloads.
                    update(
                        position,
                        "Failed",
                        "Document processing failed; remaining files will continue.",
                    )
            candidate = DocumentIndex(processed_dir)
            required = {
                item.document_id
                for item in batch.items
                if item.status in {"Ready", "Duplicate"}
            }
            if candidate.report.issues or not required <= set(candidate.documents):
                batch.error = "Files were saved, but index validation failed. The previous index remains active. Fix invalid collection files and retry the import."
            else:
                batch.index = candidate
    except Exception:  # noqa: BLE001 - local storage/index failures must not leak document contents.
        for position, item in enumerate(batch.items):
            if item.status in {"Queued", "Processing"}:
                update(position, "Failed", "Collection unavailable; try again.")
        batch.error = "Could not access or refresh the collection. Check storage permissions and retry; the previous index remains active."
    if batch.error:
        for position, item in enumerate(batch.items):
            if item.status == "Ready":
                update(
                    position,
                    "Saved",
                    "Not indexed yet; fix the collection and retry.",
                    item.document_id,
                    item.output,
                )
    return batch
