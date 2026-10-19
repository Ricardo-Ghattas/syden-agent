"""Run a small labeled lexical-retrieval evaluation without an LLM."""

import argparse
import json
from pathlib import Path

from pydantic import Field, model_validator

from app.ingestion.schemas import Source

from .indexing import DocumentIndex
from .schemas import Model, SearchFilters
from .search import search_documents


class EvaluationCase(Model):
    name: str
    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    k: int = Field(ge=1)
    expected_ids: list[str] = Field(min_length=1)
    task: str = "lexical"

    @model_validator(mode="after")
    def valid_labels(self):
        if self.task not in {"lexical", "exact_code", "exhaustive"}:
            raise ValueError("Unknown evaluation task")
        if len(set(self.expected_ids)) != len(self.expected_ids):
            raise ValueError("Evaluation labels contain duplicate IDs")
        return self


class EvaluationDataset(Model):
    cases: list[EvaluationCase] = Field(min_length=1)
    sources: dict[str, Source]


def evaluate(index: DocumentIndex, dataset: EvaluationDataset) -> dict:
    if index.report.issues:
        raise ValueError(
            "Cannot evaluate an incomplete index; inspect index.report.issues"
        )
    results = []
    exact = []
    exhaustive = []
    source_checks = source_correct = 0
    for case in dataset.cases:
        expected = set(case.expected_ids)
        if not expected.issubset(index.entries) or not expected.issubset(
            dataset.sources
        ):
            raise ValueError(
                f"Evaluation case {case.name!r} references missing records or source labels"
            )
        response = search_documents(index, case.query, case.filters, limit=case.k)
        ids = [result.record_id for result in response.results]
        correct = len(expected.intersection(ids))
        row = {
            "name": case.name,
            "query": case.query,
            "k": case.k,
            "precision_at_k": correct / case.k,
            "recall_at_k": correct / len(expected),
            "total_matches": response.total,
            "returned_ids": ids,
        }
        if case.task == "exact_code":
            row["exact_code_correct"] = bool(ids and ids[0] in expected)
            exact.append(row["exact_code_correct"])
        checked_results = response.results
        if case.task == "exhaustive":
            checked_results = []
            while True:
                page = search_documents(
                    index, case.query, case.filters, case.k, len(checked_results)
                )
                checked_results.extend(page.results)
                if not page.has_more:
                    break
            checked_ids = [result.record_id for result in checked_results]
            completeness = len(expected.intersection(checked_ids)) / len(expected)
            unique = len(checked_ids) == len(set(checked_ids))
            row.update(exhaustive_recall=completeness, no_duplicates=unique)
            exhaustive.append(completeness if unique else 0.0)
        for result in checked_results:
            record_id = result.record_id
            if record_id in dataset.sources:
                original = dataset.sources[record_id].model_dump(mode="json")
                source_checks += 1
                source_correct += (
                    result.record.source.model_dump(mode="json") == original
                )
        results.append(row)
    count = len(results)
    return {
        "cases": results,
        "mean_precision_at_k": sum(row["precision_at_k"] for row in results) / count,
        "mean_recall_at_k": sum(row["recall_at_k"] for row in results) / count,
        "exact_code_accuracy": sum(exact) / len(exact) if exact else None,
        "exhaustive_completeness": sum(exhaustive) / len(exhaustive)
        if exhaustive
        else None,
        "source_reference_correctness": source_correct / source_checks
        if source_checks
        else None,
        "source_references_checked": source_checks,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate lexical retrieval against labeled queries"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    try:
        dataset = EvaluationDataset.model_validate_json(args.dataset.read_bytes())
        result = evaluate(DocumentIndex(args.processed_dir), dataset)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
