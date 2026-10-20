"""Reproducible evaluation of actual retrieval and bounded agent execution."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.agents.agent import run_agent
from app.agents.llm import LLM, Settings
from app.agents.schemas import AgentError
from app.agents.state import ConversationSession
from app.retrieval import DocumentIndex, search_documents

from .graders import grade_answer, verify_labels
from .metrics import retrieval_metrics
from .provider import MeasuredLLM, offline_llm
from .reporting import compare, summarize, write_report
from .schemas import Case

ROOT = Path(__file__).parent


def load_cases(path: Path) -> list[Case]:
    cases = [
        Case.model_validate_json(line)
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Dataset must contain cases with unique IDs")
    return cases


def evaluate_retrieval(case, index) -> dict:
    started = perf_counter()
    page = search_documents(index, case.query, case.filters, case.k)
    ids = [result.record_id for result in page.results]
    metrics = retrieval_metrics(ids, case.expected_ids, case.k)
    if not case.exact_item:
        metrics["exact_item_accuracy"] = None
    while case.exhaustive and page.has_more:
        if page.next_offset is None or page.next_offset <= page.offset:
            raise ValueError("Retrieval pagination made no progress")
        page = search_documents(
            index, case.query, case.filters, case.k, page.next_offset
        )
        ids.extend(result.record_id for result in page.results)
    metrics["exhaustive_completeness"] = (
        len(set(ids) & set(case.expected_ids)) / len(case.expected_ids)
        if case.exhaustive and case.expected_ids
        else None
    )
    checks = {
        "relevant_records": set(case.expected_ids) <= set(ids)
        if case.exhaustive
        else set(case.expected_ids) <= set(ids[: case.k]),
        "empty_expected": bool(ids) == bool(case.expected_ids),
        "unique_records": len(ids) == len(set(ids)),
        "exact_item": not case.exact_item or metrics["exact_item_accuracy"] is True,
    }
    return {
        **metrics,
        "checks": checks,
        "returned_ids": ids,
        "latency_seconds": perf_counter() - started,
        "failures": [] if all(checks.values()) else ["RETRIEVAL_MISS"],
        "debug_area": "retrieval",
    }


def evaluate_agent(case, index, llm, mode) -> dict:
    measured = MeasuredLLM(llm)
    session = ConversationSession()
    for question in case.setup_questions:
        if question is None:
            session.clear()
        else:
            setup = run_agent(
                question, index, measured, session=session, config=case.config
            )
            if setup.status.value not in {"completed", "insufficient_evidence"}:
                return {
                    "checks": {"setup_completed": False},
                    "failures": ["CONTEXT_ERROR"],
                    "latency_seconds": sum(measured.latencies),
                    "debug_area": "conversation_setup",
                }
    started = perf_counter()
    start_request = len(measured.latencies)
    start_event = len(measured.trace)
    state = run_agent(
        case.question, index, measured, session=session, config=case.config
    )
    elapsed = perf_counter() - started
    trace = measured.trace[start_event:]
    trace.extend(
        {
            "event": "tool_finished",
            "tool": outcome.name,
            "success": outcome.success,
            "returned": outcome.returned,
            "error_code": outcome.error_code,
            "duration_seconds": outcome.duration_seconds,
        }
        for outcome in state.outcomes
    )
    trace.append(
        {
            "event": "agent_stopped",
            "status": state.status.value,
            "stop_reason": state.stop_reason.value,
        }
    )
    result = grade_answer(case, state, index, trace)
    return {
        **result,
        "question": case.question
        if case.synthetic
        else "[omitted: non-synthetic question]",
        "expected_behavior": {
            "status": case.expected_status.value,
            "abstain": case.should_abstain,
            "facts": [fact.model_dump(mode="json") for fact in case.facts]
            if case.synthetic
            else [fact.label for fact in case.facts],
        },
        "actual_behavior": {
            "status": state.status.value,
            "answer_type": state.final_response.answer_type,
            "stop_reason": state.stop_reason.value,
            "satisfied_facts": [
                fact.label
                for fact in case.facts
                if result["checks"][f"fact:{fact.label}"]
            ],
        },
        "available_source_references": [
            source.model_dump(mode="json") for source in state.sources
        ]
        if case.synthetic
        else None,
        "latency_seconds": elapsed,
        "llm_latency_seconds": sum(measured.latencies[start_request:]),
        "llm_request_latencies": measured.latencies[start_request:],
        "tool_latencies": [outcome.duration_seconds for outcome in state.outcomes],
        "tool_latency_seconds": sum(
            outcome.duration_seconds for outcome in state.outcomes
        ),
        "iterations": state.iterations,
        "tool_calls": state.tool_calls,
        "input_tokens": state.input_tokens
        if mode == "live" and measured.usage_complete
        else None,
        "output_tokens": state.output_tokens
        if mode == "live" and measured.usage_complete
        else None,
        "trace": trace,
        "available_record_ids": sorted(state.examined_record_ids),
        "cited_record_ids": [
            source.record_id for source in state.final_response.sources
        ],
        "debug_area": "retrieval"
        if not result["checks"]["evidence_complete"]
        else "agent_answer_or_tools",
    }


def run(
    dataset: Path,
    *,
    corpus: Path = ROOT / "corpus",
    scripts_path: Path = ROOT / "datasets" / "scripts.json",
    mode: str = "offline",
    allow_paid: bool = False,
    case_id: str | None = None,
    category: str | None = None,
    baseline: Path | None = None,
) -> dict:
    if mode not in {"offline", "live"}:
        raise ValueError("Unknown evaluation mode")
    if mode == "live" and not allow_paid:
        raise ValueError(
            "Live evaluation requires --allow-paid to acknowledge API charges"
        )
    cases = load_cases(dataset)
    selected = [
        case
        for case in cases
        if (case_id is None or case.id == case_id)
        and (category is None or case.category == category)
    ]
    if not selected:
        raise ValueError("No matching evaluation cases")
    index = DocumentIndex(corpus)
    if index.report.issues:
        raise ValueError("Evaluation corpus is incomplete; fix index issues first")
    for case in selected:
        if case.verified:
            verify_labels(case, index)
    scripts = json.loads(scripts_path.read_bytes()) if mode == "offline" else {}
    llm = (
        LLM(Settings.from_env())
        if mode == "live"
        and any(
            case.layer == "agent" and case.verified and case.live_eligible
            for case in selected
        )
        else None
    )
    rows = []
    try:
        for case in selected:
            row = {"id": case.id, "category": case.category, "layer": case.layer}
            if not case.verified or (mode == "live" and not case.live_eligible):
                rows.append(
                    {
                        **row,
                        "skipped": "unverified"
                        if not case.verified
                        else "offline_fault_injection",
                    }
                )
                continue
            if case.layer == "retrieval":
                result = evaluate_retrieval(case, index)
            else:
                if mode == "offline" and case.id not in scripts:
                    raise ValueError(
                        f"Missing independent offline transcript for {case.id}"
                    )
                provider = (
                    offline_llm(scripts[case.id], index) if mode == "offline" else llm
                )
                result = evaluate_agent(case, index, provider, mode)
            rows.append({**row, **result, "passed": all(result["checks"].values())})
    finally:
        if llm is not None:
            llm.client.close()
    corpus_hash = hashlib.sha256()
    for path in sorted(corpus.glob("*.jsonl")):
        corpus_hash.update(path.name.encode() + b"\0" + path.read_bytes())
    report = {
        "report_version": 1,
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "corpus_sha256": corpus_hash.hexdigest(),
        "scripts_sha256": hashlib.sha256(scripts_path.read_bytes()).hexdigest()
        if mode == "offline"
        else None,
        "selected_ids": [case.id for case in selected],
        "synthetic_only": all(case.synthetic for case in selected),
        "model": llm.settings.model if llm is not None else None,
        "summary": summarize(rows, mode),
        "cases": rows,
        "limitations": [
            "Offline scripted results test contracts, not live model accuracy. Live accuracy remains unmeasured in offline mode.",
            "Grounding scores cover only independently labelled objective facts and supporting records; they do not establish arbitrary semantic entailment.",
            "Questions request labelled facts. Explanatory prose needs manual review; no automated subjective judge is enabled.",
            "Synthetic fixtures do not establish performance on real construction documents. Latency excludes index loading and conversation setup; provider latency includes wrapper overhead.",
            "P95 is omitted below 20 samples per layer. Offline token counts and cost estimates are unavailable. Cost estimation is not implemented.",
            "Application boundaries are tested; resisting every prompt injection is not guaranteed. Real document totals require independently verified scope.",
        ],
    }
    if baseline:
        report["comparison"] = compare(report, json.loads(baseline.read_bytes()))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate SYDEN using synthetic or verified fixtures"
    )
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "datasets" / "golden.jsonl"
    )
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    parser.add_argument(
        "--scripts", type=Path, default=ROOT / "datasets" / "scripts.json"
    )
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--category")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "live" and not args.allow_paid:
        parser.error(
            "Live evaluation incurs API charges; explicitly opt in with --allow-paid"
        )
    try:
        report = run(
            args.dataset,
            corpus=args.corpus,
            scripts_path=args.scripts,
            mode=args.mode,
            allow_paid=args.allow_paid,
            case_id=args.case_id,
            category=args.category,
            baseline=args.baseline,
        )
        write_report(report, args.output_dir)
    except (OSError, ValueError, AgentError):
        parser.exit(
            2,
            "Evaluation setup failed; check dataset labels, corpus integrity, selection, API configuration, and paid opt-in. No provider payload was printed.\n",
        )
    summary = report["summary"]
    print(
        f"{args.mode}: {summary['executed']} executed; {summary['passed']} passed; {summary['failed']} failed; {summary['skipped']} skipped. Reports: {args.output_dir / 'latest.json'}, {args.output_dir / 'latest.md'}"
    )
    if summary["failed"] or report.get("comparison", {}).get("regressions"):
        parser.exit(1)


if __name__ == "__main__":
    main()
