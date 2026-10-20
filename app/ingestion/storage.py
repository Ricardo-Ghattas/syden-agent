"""Archive original snapshots and hash their bytes for source provenance."""

import hashlib
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile


# * Compute the SHA-256 hash of a file's bytes.
def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# * Archive the original file and return its path and SHA-256 digest.
def archive_original(path: Path, raw_dir: Path) -> tuple[Path, str]:
    """Return the archived snapshot path and its SHA-256 digest.

    Identical snapshots are reused. A modified archive raises ``ValueError``
    instead of being overwritten; failed copies leave no temporary snapshot.
    """
    suffix = path.suffix.casefold()
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Hash the archived bytes, so extraction and provenance always refer to the same snapshot.
    temporary = None
    try:
        with NamedTemporaryFile(dir=raw_dir, suffix=suffix, delete=False) as stream:
            temporary = Path(stream.name)
            with path.open("rb") as original:
                shutil.copyfileobj(original, stream)
        digest = source_sha256(temporary)
        raw_path = raw_dir / f"{path.stem[:100]}-{digest}{suffix}"
        if raw_path.exists():
            if source_sha256(raw_path) != digest:
                raise ValueError(f"Archived file has been modified: {raw_path}")
        else:
            temporary.replace(raw_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return raw_path, digest
