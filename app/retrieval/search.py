"""Unified keyword retrieval over an existing in-memory index snapshot."""

from .filtering import matches_filters
from .indexing import DocumentIndex, IndexedRecord
from .ranking import rank
from .schemas import RankingWeights, SearchFilters, SearchResponse, SearchResult
from .text import contains_phrase, parse_query


def _snippet(entry: IndexedRecord, fields: list[str]) -> str:
    record = entry.record
    if (
        "description" in fields
        and record.boq is not None
        and record.boq.description is not None
    ):
        return str(record.boq.description)[:320]
    if "heading" in fields and record.semantic_path:
        return " > ".join(record.semantic_path)[:320]
    return (record.text or entry.fields["body"] or entry.fields["metadata"])[:320]


def search_documents(
    index: DocumentIndex,
    query: str,
    filters: SearchFilters | dict | None = None,
    limit: int = 10,
    offset: int = 0,
    *,
    weights: RankingWeights | None = None,
) -> SearchResponse:
    """Return OR-matched keywords with exact quoted phrases and stable pagination.

    Filters are AND-composed. No index files are read during search; call reload()
    after reprocessing. Every response exposes index errors and total matches.
    """
    parsed = parse_query(query)
    if isinstance(filters, SearchFilters):
        filters = filters.model_dump()
    filters = SearchFilters.model_validate({} if filters is None else filters)
    weights = RankingWeights.model_validate(
        weights.model_dump() if weights is not None else {}
    )
    if type(limit) is not int or limit < 1 or type(offset) is not int or offset < 0:
        raise ValueError(
            "limit must be a positive integer and offset a non-negative integer"
        )
    expanded = {term: index.expand_term(term) for term in parsed.terms}
    candidates = set()
    for alternatives in expanded.values():
        for term in alternatives:
            candidates.update(index.postings[term])
    for code in (*parsed.terms, parsed.text.strip('"')):
        candidates.update(index.item_codes.get(code, ()))
    matches = []
    for record_id in candidates:
        entry = index.entries[record_id]
        if not matches_filters(entry.record, filters):
            continue
        if parsed.quoted and not all(
            any(contains_phrase(text, phrase) for text in entry.fields.values())
            for phrase in parsed.phrases
        ):
            continue
        score, fields, terms = rank(entry, parsed, expanded, index, weights)
        matches.append((score, record_id, fields, terms))
    matches.sort(
        key=lambda value: (
            -value[0],
            index.entries[value[1]].record.document_id,
            index.entries[value[1]].record.order_index,
            value[1],
        )
    )
    total = len(matches)
    page = matches[offset : offset + limit]
    results = []
    for score, record_id, fields, terms in page:
        entry = index.entries[record_id]
        results.append(
            SearchResult(
                record_id=record_id,
                kind=entry.record.kind,
                score=score,
                matched_fields=fields,
                matched_terms=terms,
                snippet=_snippet(entry, fields),
                record=entry.record.model_copy(deep=True),
            )
        )
    has_more = offset + len(results) < total
    return SearchResponse(
        query=query,
        filters=filters,
        results=results,
        total=total,
        returned=len(results),
        limit=limit,
        offset=offset,
        has_more=has_more,
        next_offset=offset + len(results) if has_more else None,
        index_complete=not index.report.issues,
        index_issues=[issue.model_copy(deep=True) for issue in index.report.issues],
    )
