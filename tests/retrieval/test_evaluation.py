from pathlib import Path

import pytest

from app.retrieval.evaluate import EvaluationDataset, evaluate

FIXTURES = Path(__file__).parent / "fixtures"


def test_evaluation_reports_retrieval_quality_and_known_synonym_gap(index):
    dataset = EvaluationDataset.model_validate_json(
        (FIXTURES / "queries.json").read_bytes()
    )
    report = evaluate(index, dataset)
    assert report["exact_code_accuracy"] == 1.0
    assert report["exhaustive_completeness"] == 1.0
    assert report["source_reference_correctness"] == 1.0
    assert report["source_references_checked"] >= 35
    excavation = next(
        case for case in report["cases"] if case["name"] == "all excavation rows"
    )
    assert excavation["total_matches"] == 35
    assert excavation["no_duplicates"]
    gap = next(
        case for case in report["cases"] if case["name"] == "known lexical synonym gap"
    )
    assert gap["recall_at_k"] == 0
    assert gap["returned_ids"] == []


def test_evaluation_checks_ground_truth_sources_and_missing_labels(index):
    dataset = EvaluationDataset.model_validate_json(
        (FIXTURES / "queries.json").read_bytes()
    )
    dataset.sources["b" * 64 + ":39"].row_start = 999
    report = evaluate(index, dataset)
    assert report["source_reference_correctness"] < 1
    dataset.cases[0].expected_ids = ["missing"]
    with pytest.raises(ValueError, match="missing records"):
        evaluate(index, dataset)
