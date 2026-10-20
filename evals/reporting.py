"""Reports deliberately omit provider text, raw arguments, and document contents."""

import json
from pathlib import Path

from .metrics import latency, mean


def summarize(rows: list[dict], mode: str) -> dict:
    executed = [row for row in rows if not row.get("skipped")]
    retrieval = [row for row in executed if row["layer"] == "retrieval"]
    agent = [row for row in executed if row["layer"] == "agent"]
    citations = sum(row.get("citation_count", 0) for row in agent)
    claims = sum(row.get("claim_count", 0) for row in agent)
    numerical_claims = sum(row.get("numerical_claim_count", 0) for row in agent)
    return {
        "executed": len(executed),
        "passed": sum(row["passed"] for row in executed),
        "failed": sum(not row["passed"] for row in executed),
        "skipped": len(rows) - len(executed),
        "retrieval_cases": len(retrieval),
        "agent_cases": len(agent),
        "by_category": {
            category: {
                "executed": len(group),
                "failed": sum(not row["passed"] for row in group),
            }
            for category in sorted({row["category"] for row in executed})
            if (group := [row for row in executed if row["category"] == category])
        },
        "failure_breakdown": {
            failure: sum(failure in row["failures"] for row in executed)
            for failure in sorted(
                {failure for row in executed for failure in row["failures"]}
            )
        },
        **{
            key: mean([row.get(key) for row in retrieval])
            for key in [
                "precision_at_k",
                "recall_at_k",
                "reciprocal_rank",
                "exact_item_accuracy",
                "exhaustive_completeness",
            ]
        },
        "answer_accuracy": mean([row.get("answer_correct") for row in agent]),
        "objective_answer_cases": sum(
            row.get("answer_correct") is not None for row in agent
        ),
        "abstention_contract_accuracy": mean(
            [row.get("abstention_correct") for row in agent]
        ),
        "answer_accuracy_kind": "offline_script_contract"
        if mode == "offline"
        else "live_objective_label_accuracy",
        "numerical_accuracy": mean([row.get("numerical_accuracy") for row in agent]),
        "unit_correctness": mean([row.get("unit_correctness") for row in agent]),
        "item_selection_correctness": mean(
            [row.get("item_selection_correctness") for row in agent]
        ),
        "numerical_claim_count": numerical_claims,
        "unsupported_or_incorrect_numerical_claim_rate": sum(
            row.get("unsupported_or_incorrect_numerical_claims", 0) for row in agent
        )
        / numerical_claims
        if numerical_claims
        else None,
        "citation_validity_rate": sum(
            row.get("citation_valid_count", 0) for row in agent
        )
        / citations
        if citations
        else None,
        "citation_support_rate": sum(row.get("supported_claims", 0) for row in agent)
        / claims
        if claims
        else None,
        "evaluated_citations": citations,
        "evaluated_claims": claims,
        "grounded_answer_rate": mean([row.get("grounded") for row in agent]),
        "tool_use_correctness": mean([row.get("tool_correct") for row in agent]),
        "complete_evidence_rate": mean(
            [
                row["checks"].get("evidence_complete")
                for row in agent
                if row.get("expected_record_count", 0)
            ]
        ),
        "unsupported_claim_rate": sum(
            row.get("claim_count", 0) - row.get("supported_claims", 0) for row in agent
        )
        / claims
        if claims
        else None,
        "latency": {
            "retrieval_only": latency([row["latency_seconds"] for row in retrieval]),
            "agent": latency([row["latency_seconds"] for row in agent]),
        },
        "input_tokens": sum(row["input_tokens"] for row in agent)
        if agent and all(row.get("input_tokens") is not None for row in agent)
        else None,
        "output_tokens": sum(row["output_tokens"] for row in agent)
        if agent and all(row.get("output_tokens") is not None for row in agent)
        else None,
        "total_tokens": sum(row["input_tokens"] + row["output_tokens"] for row in agent)
        if agent
        and all(
            row.get("input_tokens") is not None and row.get("output_tokens") is not None
            for row in agent
        )
        else None,
        "llm_requests": sum(row.get("iterations", 0) for row in agent),
        "tool_calls": sum(row.get("tool_calls", 0) for row in agent),
        "llm_request_latency": latency(
            [
                duration
                for row in agent
                for duration in row.get("llm_request_latencies", [])
            ]
        ),
        "tool_latency": latency(
            [duration for row in agent for duration in row.get("tool_latencies", [])]
        ),
        "estimated_cost": None,
    }


def compare(report: dict, baseline: dict) -> dict:
    for key in (
        "report_version",
        "mode",
        "dataset_sha256",
        "corpus_sha256",
        "scripts_sha256",
        "selected_ids",
    ):
        if report[key] != baseline[key]:
            raise ValueError(f"Cannot compare different {key}")
    old = {row["id"]: row for row in baseline["cases"]}
    regressions, improvements = [], []
    for row in report["cases"]:
        previous = old[row["id"]]
        if row.get("skipped") or previous.get("skipped"):
            continue
        lost = [
            key
            for key, value in row["checks"].items()
            if previous["checks"].get(key) is True and value is False
        ]
        if lost:
            regressions.append({"id": row["id"], "checks": lost})
        if row["passed"] and not previous["passed"]:
            improvements.append(row["id"])
    return {"regressions": regressions, "improvements": improvements}


def write_report(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    lines = [
        "# SYDEN evaluation",
        "",
        f"Mode: **{report['mode']}**; dataset: `{report['dataset_sha256']}`.",
        "",
        f"Executed {summary['executed']}: {summary['passed']} passed, {summary['failed']} failed; {summary['skipped']} skipped.",
        "",
        "| Metric | Measured result |",
        "| --- | --- |",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {json.dumps(value, sort_keys=True)} |")
    lines.extend(["", "## Failures", ""])
    for row in report["cases"]:
        if not row.get("skipped") and not row["passed"]:
            lines.append(
                f"- `{row['id']}` ({row['category']}): {', '.join(row['failures'])}. Debug: {row['debug_area']}; failed checks: {', '.join(key for key, value in row['checks'].items() if not value)}."
            )
    lines.extend(
        ["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]]]
    )
    if "comparison" in report:
        lines.extend(
            [
                "",
                "## Baseline comparison",
                "",
                "```json",
                json.dumps(report["comparison"], indent=2),
                "```",
            ]
        )
    (directory / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
