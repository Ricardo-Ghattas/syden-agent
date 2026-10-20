"""Archive an original, extract its evidence, normalize, and export JSONL."""

from pathlib import Path

from .exporters.jsonl import export_jsonl
from .extractors.excel import SpreadsheetConverter
from .extractors.pdf import read_pdf
from .schemas import Source
from .storage import archive_original
from .transformers.normalize import normalize


def process_document(
    path: str | Path,
    data_dir: str | Path = "data",
    *,
    processed_dir: str | Path | None = None,
) -> Path:
    """Archive a PDF or XLSX file and return its validated JSONL output path.

    Originals are retained in ``data_dir/raw``; JSONL is written atomically to
    ``data_dir/processed``. Extraction or validation failures retain the archive
    and leave any previous output intact. ``processed_dir`` overrides only the
    JSONL destination; archives remain under ``data_dir/raw``.
    """
    path = Path(path).resolve(strict=True)

    # * Check that the file is a PDF or XLSX
    suffix = path.suffix.casefold()
    if not path.is_file() or suffix not in {".pdf", ".xlsx"}:
        raise ValueError("Supported inputs are PDF (.pdf) and Excel (.xlsx) files.")
    data_dir = Path(data_dir).resolve()

    # * Archive the original file and in data/raw
    raw_path, digest = archive_original(path, data_dir / "raw")

    # * Read the document and extract its evidence.
    if suffix == ".pdf":
        document = read_pdf(raw_path)
    else:
        document = SpreadsheetConverter().convert(raw_path)

    # * Create the source metadata. Record the filename, file format, archive location, and SHA-256 hash.
    source = Source(
        filename=path.name,
        raw_path=raw_path.relative_to(data_dir).as_posix(),
        sha256=digest,
        format="pdf" if suffix == ".pdf" else "xlsx",
    )
    records = normalize(document, source)
    # * Extractors see the archived name; expose the caller's original filename in the document record.
    records[0].text = path.name
    output_dir = (
        Path(processed_dir).resolve()
        if processed_dir is not None
        else data_dir / "processed"
    )
    return export_jsonl(records, output_dir / f"{raw_path.stem}.jsonl")
