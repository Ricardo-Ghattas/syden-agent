import copy
import json
import socket
from decimal import Decimal

import pytest

from app.agents.agent import run_agent
from app.ingestion.pipeline import process_document
from app.retrieval import DocumentIndex, search_documents
from app.tools.registry import reference
from evals.graders import fact_value, grade_answer
from evals.metrics import latency, retrieval_metrics
from evals.provider import offline_llm
from evals.reporting import compare, write_report
from evals.runner import ROOT, evaluate_agent, load_cases, main, run
from evals.schemas import Case

DATASET = ROOT / "datasets" / "golden.jsonl"


@pytest.fixture
def corpus():
    return DocumentIndex(ROOT / "corpus")


@pytest.fixture
def quantity_case():
    return next(
        case for case in load_cases(DATASET) if case.id == "foundation_quantity"
    )


def test_complete_offline_regression_needs_no_credentials_or_network(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    def deny(*args, **kwargs):
        pytest.fail("Offline evaluation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    report = run(DATASET)
    assert report["summary"]["executed"] == 54
    assert report["summary"]["failed"] == 0
    assert report["summary"]["input_tokens"] is None
    assert report["model"] is None
    assert report["summary"]["answer_accuracy_kind"] == "offline_script_contract"
    write_report(report, tmp_path)
    serialized = (tmp_path / "latest.json").read_text()
    assert "Ignore previous instructions" not in serialized
    assert '"command":' not in serialized
    assert "__import__" not in serialized
    assert "hidden_reasoning" not in serialized
    assert (tmp_path / "latest.md").exists()


def test_metrics_have_correct_denominators_and_rank():
    result = retrieval_metrics(["wrong", "b", "a"], ["a", "b", "c"], 2)
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 1 / 3
    assert result["reciprocal_rank"] == 0.5
    assert retrieval_metrics([], [], 5)["recall_at_k"] is None
    assert latency([1.0] * 19)["p95_seconds"] is None
    assert latency(list(range(1, 21)))["p95_seconds"] == 19


@pytest.mark.parametrize(
    "actual,number,unit",
    [
        ("200 m³", True, True),
        ("200.000 m3", True, True),
        ("200 kg", True, False),
        ("201 m3", False, True),
        ("1,200 m3", False, True),
        ("200,5 m3", False, False),
        ("-200 m3", False, True),
        ("2000 m3", False, True),
        ("NaN m3", False, False),
    ],
)
def test_numeric_grader_preserves_units_precision_and_sign(
    quantity_case, actual, number, unit
):
    assert fact_value(actual, quantity_case.facts[0]) == (number, unit)


def test_tolerance_is_explicit_and_identifiers_keep_leading_zeros(quantity_case):
    fact = quantity_case.facts[0].model_copy(
        update={"value": "1.23", "tolerance": Decimal("0.005")}
    )
    assert fact_value("1.234 m3", fact) == (True, True)
    assert fact_value("1.24 m3", fact) == (False, True)
    code = fact.model_copy(update={"numeric": False, "value": "001"})
    assert fact_value("1", code)[0] is False


@pytest.mark.parametrize(
    "text,sources,failure",
    [
        ("quantity: 201 m3", ["a" * 64 + ":2"], "NUMERICAL_ERROR"),
        ("quantity: 200 kg", ["a" * 64 + ":2"], "UNIT_ERROR"),
        ("quantity: 200 m3", ["a" * 64 + ":3"], "UNSUPPORTED_CLAIM"),
        ("quantity: 200 m3\nsecret: fictional", ["a" * 64 + ":2"], "UNSUPPORTED_CLAIM"),
        ("quantity: 999 m3\nquantity: 200 m3", ["a" * 64 + ":2"], "UNSUPPORTED_CLAIM"),
    ],
)
def test_wrong_answers_and_valid_but_wrong_citations_fail(
    corpus, quantity_case, text, sources, failure
):
    steps = [
        {
            "calls": [
                {"name": "get_record", "arguments": {"record_id": rid}}
                for rid in ["a" * 64 + ":2", "a" * 64 + ":3"]
            ]
        },
        {"answer": text, "sources": sources},
    ]
    result = evaluate_agent(
        quantity_case, corpus, offline_llm(steps, corpus), "offline"
    )
    assert failure in result["failures"]
    assert not all(result["checks"].values())


def test_fabricated_location_and_missing_evidence_are_distinct(corpus, quantity_case):
    rid = quantity_case.expected_ids[0]
    steps = [
        {"calls": [{"name": "get_record", "arguments": {"record_id": rid}}]},
        {"answer": "quantity: 200 m3", "sources": [rid]},
    ]
    state = run_agent(quantity_case.question, corpus, offline_llm(steps, corpus))
    state.final_response.sources[0] = reference(corpus.get_record(rid)).model_copy(
        update={"row_start": 999}
    )
    result = grade_answer(quantity_case, state, corpus, [])
    assert "CITATION_ERROR" in result["failures"]
    state.final_response.sources[0] = reference(corpus.get_record(rid))
    state.evidence[rid].pop("boq")
    assert (
        "UNSUPPORTED_CLAIM"
        in grade_answer(quantity_case, state, corpus, [])["failures"]
    )


def test_bad_golden_labels_fail_before_execution(tmp_path):
    rows = DATASET.read_text().splitlines()
    case = json.loads(rows[14])
    case["facts"][0]["value"] = "999"
    target = tmp_path / "bad.jsonl"
    target.write_text(json.dumps(case) + "\n")
    with pytest.raises(ValueError, match="ground truth"):
        run(target)


def test_unverified_cases_are_skipped_and_not_in_metrics(quantity_case, tmp_path):
    case = quantity_case.model_copy(update={"verified": False})
    path = tmp_path / "unverified.jsonl"
    path.write_text(case.model_dump_json() + "\n")
    report = run(path)
    assert report["summary"]["executed"] == 0
    assert report["summary"]["skipped"] == 1
    assert report["summary"]["answer_accuracy"] is None


def test_selection_live_gate_and_ci_failure_exit(tmp_path):
    with pytest.raises(ValueError, match="allow-paid"):
        run(DATASET, mode="live")
    with pytest.raises(ValueError, match="No matching"):
        run(DATASET, case_id="does_not_exist")
    report = run(DATASET, category="conversation")
    assert report["summary"]["executed"] == 4
    scripts = json.loads((ROOT / "datasets" / "scripts.json").read_text())
    scripts["foundation_quantity"][-1]["answer"] = "quantity: 999 m3"
    path = tmp_path / "scripts.json"
    path.write_text(json.dumps(scripts))
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--case",
                "foundation_quantity",
                "--scripts",
                str(path),
                "--output-dir",
                str(tmp_path / "reports"),
            ]
        )
    assert caught.value.code == 1


def test_regressions_are_detected_even_when_aggregate_pass_does_not_change():
    report = run(DATASET, case_id="foundation_quantity")
    changed = copy.deepcopy(report)
    changed["cases"][0]["checks"]["citation_support"] = False
    assert compare(changed, report)["regressions"][0]["checks"] == ["citation_support"]
    changed["corpus_sha256"] = "different"
    with pytest.raises(ValueError, match="corpus_sha256"):
        compare(changed, report)


def test_raw_workbook_to_ingestion_retrieval_agent_and_grading(workbook, tmp_path):
    process_document(workbook, tmp_path / "data")
    index = DocumentIndex(tmp_path / "data" / "processed")
    record = search_documents(index, "001", {"kind": "boq_item"}).results[0].record
    case = Case.model_validate_json(
        json.dumps(
            {
                "id": "pipeline_quantity",
                "category": "integration",
                "layer": "agent",
                "synthetic": True,
                "verified": True,
                "question": "For 001, reply only quantity: value m3.",
                "expected_ids": [record.id],
                "facts": [
                    {
                        "label": "quantity",
                        "value": "0",
                        "unit": "m3",
                        "numeric": True,
                        "support_ids": [record.id],
                        "field": "boq.quantity",
                    }
                ],
            }
        )
    )
    steps = [
        {
            "calls": [
                {
                    "name": "search_documents",
                    "arguments": {
                        "query": "001",
                        "filters": {"kind": "boq_item"},
                        "limit": 1,
                        "offset": 0,
                        "exhaustive": False,
                    },
                }
            ]
        },
        {"answer": "quantity: 0 m3", "sources": [record.id]},
    ]
    result = evaluate_agent(case, index, offline_llm(steps, index), "offline")
    assert all(result["checks"].values())


def test_currency_is_not_interchangeable_with_another_currency(corpus):
    case = next(
        case for case in load_cases(DATASET) if case.id == "alternative_currency"
    )
    assert fact_value("26250 USD", case.facts[0]) == (True, False)


def test_non_synthetic_report_redacts_content(corpus, quantity_case):
    case = quantity_case.model_copy(
        update={"synthetic": False, "question": "CONFIDENTIAL QUESTION"}
    )
    scripts = json.loads((ROOT / "datasets" / "scripts.json").read_text())
    result = evaluate_agent(
        case, corpus, offline_llm(scripts[case.id], corpus), "offline"
    )
    serialized = json.dumps(result)
    assert "CONFIDENTIAL" not in serialized
    assert "synthetic-boq.xlsx" not in serialized
    assert result["available_source_references"] is None


def test_live_fault_only_selection_is_skipped_without_client_or_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report = run(
        DATASET, mode="live", allow_paid=True, case_id="authentication_failure"
    )
    assert report["summary"]["executed"] == 0
    assert report["summary"]["skipped"] == 1
    assert report["summary"]["answer_accuracy"] is None
