"""Validate and serialize records as UTF-8 JSONL with atomic replacement."""

import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..schemas import Record


def export_jsonl(records: Iterable[Record], path: Path) -> Path:
    """Write records in input order and return the output path.

    Each record is revalidated before serialization. The destination is replaced
    only after all records have been written successfully.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for record in records:
                validated = Record.model_validate(record.model_dump())
                stream.write(validated.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
