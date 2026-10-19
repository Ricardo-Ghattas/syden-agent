"""Apply exact, composable filters independently of relevance scoring."""

from app.ingestion.schemas import Record

from .schemas import SearchFilters
from .text import normalize_text


def matches_filters(record: Record, filters: SearchFilters) -> bool:
    source = record.source
    exact = {
        "format": source.format,
        "document_id": record.document_id,
        "filename": source.filename,
        "sheet": source.sheet,
        "kind": record.kind,
        "item_code": None
        if record.boq is None or record.boq.item is None
        else str(record.boq.item),
    }
    for key, actual in exact.items():
        expected = getattr(filters, key)
        if expected is not None and (
            actual is None or normalize_text(actual) != normalize_text(expected)
        ):
            return False
    if filters.page is not None and source.page != filters.page:
        return False
    return filters.section is None or any(
        normalize_text(part) == normalize_text(filters.section)
        for part in record.semantic_path
    )
