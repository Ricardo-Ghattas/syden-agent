"""Small metrics with explicit denominators; absent measurements stay null."""

from math import ceil
from statistics import median


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def retrieval_metrics(ids: list[str], expected: list[str], k: int) -> dict:
    relevant = set(expected)
    top = ids[:k]
    hits = len(set(top) & relevant)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant) if relevant else None,
        "reciprocal_rank": next(
            (1 / rank for rank, rid in enumerate(top, 1) if rid in relevant), 0.0
        )
        if relevant
        else None,
        "exact_item_accuracy": bool(ids and ids[0] in relevant) if relevant else None,
    }


def latency(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "median_seconds": median(values) if values else None,
        "p95_seconds": sorted(values)[ceil(0.95 * len(values)) - 1]
        if len(values) >= 20
        else None,
    }
