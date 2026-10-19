"""Bounded field-weighted BM25 with exact-code, phrase, and coverage boosts."""

from math import log1p

from .indexing import DocumentIndex, IndexedRecord
from .schemas import RankingWeights
from .text import Query, contains_phrase


def bm25(tf: int, length: int, average: float, count: int, frequency: int) -> float:
    if tf == 0:
        return 0.0
    idf = log1p((count - frequency + 0.5) / (frequency + 0.5))
    return idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / max(average, 1.0)))


def rank(
    entry: IndexedRecord,
    query: Query,
    expanded: dict[str, tuple[str, ...]],
    index: DocumentIndex,
    weights: RankingWeights,
) -> tuple[float, list[str], list[str]]:
    matched_fields = []
    matched_terms = set()
    field_score = 0.0
    for field, counts in entry.terms.items():
        value = 0.0
        for term, alternatives in expanded.items():
            candidates = []
            for alternative in alternatives:
                if counts[alternative]:
                    matched_terms.add(term)
                    candidates.append(
                        bm25(
                            counts[alternative],
                            entry.lengths[field],
                            index.average_lengths[field],
                            len(index.entries),
                            index.frequencies[field][alternative],
                        )
                        * (1.0 if alternative == term else 0.65)
                    )
            value += max(candidates, default=0.0)
        if value:
            matched_fields.append(field)
            # Saturation bounds custom weights so an exact item-code bonus always wins.
            field_score += getattr(weights, field) * value / (1.0 + value)
    exact_code = entry.item_code is not None and (
        entry.item_code == query.text.strip('"') or entry.item_code in query.terms
    )
    if exact_code:
        matched_fields.insert(0, "item_code")
    phrase = any(
        contains_phrase(text, value)
        for text in entry.fields.values()
        for value in query.phrases
    )
    coverage = len(matched_terms) / len(query.terms)
    score = 100.0 * exact_code + 20.0 * phrase + 10.0 * coverage + field_score
    return score, matched_fields, sorted(matched_terms)
