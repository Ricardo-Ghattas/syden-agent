"""Objective labelled-fact checks; no LLM judge and no production arithmetic."""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from app.tools.registry import reference

from .metrics import mean


def canonical(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def fact_value(actual: str, fact) -> tuple[bool, bool | None]:
    if not fact.numeric:
        return canonical(actual) == canonical(fact.value), None
    # Commas are accepted only as unambiguous thousands separators.
    match = re.fullmatch(
        r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\d.,])\s*((?:[^\d.,\s].*)?)",
        canonical(actual),
    )
    if not match:
        return False, False
    number, unit = match.groups()
    return (
        abs(Decimal(number.replace(",", "")) - Decimal(fact.value)) <= fact.tolerance,
        canonical(unit) == canonical(fact.unit),
    )


def field_value(record: dict, path: str):
    value = record
    for part in path.split("."):
        value = value[part]
    return value


def verify_labels(case, index) -> None:
    """Reject invalid golden labels before grading any model output."""
    for rid in case.expected_ids:
        if index.get_record(rid) is None:
            raise ValueError(f"Case {case.id}: missing labelled record")
    for fact in case.facts:
        try:
            records = [
                index.get_record(rid).model_dump(mode="json")
                for rid in fact.support_ids
            ]
            values = [field_value(record, fact.field) for record in records]
            if fact.operation == "contains":
                needle = canonical(f"{fact.value} {fact.unit}".strip())
                correct = all(
                    re.search(
                        r"(?<!\w)" + re.escape(needle) + r"(?!\w)",
                        canonical(str(value)),
                    )
                    for value in values
                )
            elif fact.numeric:
                numbers = [Decimal(str(value)) for value in values]
                if fact.operation == "value":
                    result = numbers[0] if len(numbers) == 1 else None
                else:
                    result = {"sum": sum, "min": min, "max": max}[fact.operation](
                        numbers
                    )
                correct = result is not None and result == Decimal(fact.value)
                if fact.field.startswith("boq."):
                    for record in records:
                        unit = record["boq"]["unit"]
                        currency = record["metadata"].get("currency", "")
                        expected_unit = {
                            "boq.quantity": unit,
                            "boq.rate": f"{currency}/{unit}",
                            "boq.amount": currency,
                        }.get(fact.field, fact.unit)
                        correct = correct and canonical(fact.unit) == canonical(
                            str(expected_unit)
                        )
            else:
                correct = len(values) == 1 and canonical(str(values[0])) == canonical(
                    fact.value
                )
        except (KeyError, TypeError, InvalidOperation) as error:
            raise ValueError(f"Case {case.id}: invalid evidence field") from error
        if not correct:
            raise ValueError(
                f"Case {case.id}: ground truth disagrees with corpus ({fact.label})"
            )


def grade_answer(case, state, index, trace: list[dict]) -> dict:
    response = state.final_response
    checks = {}
    failures = []
    checks["status"] = response.status == case.expected_status
    checks["stop_reason"] = (
        case.expected_stop is None or state.stop_reason == case.expected_stop
    )
    checks["abstention"] = (
        response.answer_type == "insufficient"
    ) == case.should_abstain
    cited = set()
    validity = []
    for source in response.sources:
        record = index.get_record(source.record_id) if source.record_id else None
        valid = (
            record is not None
            and source == reference(record)
            and source in state.sources
        )
        validity.append(valid)
        if valid:
            cited.add(source.record_id)
    checks["citation_validity"] = all(validity)
    if not checks["citation_validity"]:
        failures.append("CITATION_ERROR")

    # Controlled questions request only labelled lines; extra prose is ungraded evidence.
    lines = [line.strip() for line in response.answer.splitlines() if line.strip()]
    parsed = [
        re.fullmatch(r"([a-z][a-z_0-9]*)\s*[:=]\s*(.+)", line, re.IGNORECASE)
        for line in lines
    ]
    labelled = {}
    duplicates = False
    for match in parsed:
        if match:
            label, value = match.groups()
            label = label.casefold()
            duplicates |= label in labelled
            labelled[label] = value
    numbers, units, supports, facts = [], [], [], []
    numerical_support, item_selection = [], []
    for fact in case.facts:
        number_ok, unit_ok = fact_value(labelled.get(fact.label, ""), fact)
        correct = number_ok and unit_ok is not False
        if fact.numeric:
            numbers.append(number_ok)
            units.append(unit_ok)
        facts.append(correct)
        observed = True
        for rid in fact.support_ids:
            try:
                observed &= field_value(state.evidence[rid], fact.field) == field_value(
                    index.get_record(rid).model_dump(mode="json"), fact.field
                )
            except (KeyError, TypeError):
                observed = False
        supports.append(
            correct
            and observed
            and set(fact.support_ids) <= cited
            and set(fact.support_ids) <= state.examined_record_ids
        )
        if fact.numeric:
            numerical_support.append(supports[-1])
            item_selection.append(observed and set(fact.support_ids) <= cited)
        checks[f"fact:{fact.label}"] = correct
    checks["answer_format"] = not case.facts or (
        all(parsed)
        and not duplicates
        and set(labelled) == {fact.label for fact in case.facts}
    )
    checks["evidence_complete"] = set(case.expected_ids) <= state.examined_record_ids
    checks["citation_support"] = all(supports)
    tools = {outcome.name for outcome in state.outcomes if outcome.success}
    checks["authorized_tools"] = tools <= {
        "search_documents",
        "get_record",
        "list_documents",
    }
    if not checks["authorized_tools"]:
        failures.append("SECURITY_VIOLATION")
    checks["tool_behavior"] = (
        set(case.required_tools) <= tools and state.tool_calls <= case.max_tools
    )
    if case.exhaustive:
        checks["tool_behavior"] &= any(
            event.get("tool") == "search_documents"
            and event.get("arguments", {}).get("exhaustive")
            and event.get("arguments", {}).get("offset") == 0
            for event in trace
        )
        checks["tool_behavior"] &= any(
            event.get("event") == "tool_observed"
            and event.get("complete") is True
            and event.get("success") is True
            and event.get("index_complete") is True
            for event in trace
        )
    if numbers and not all(numbers):
        failures.append("NUMERICAL_ERROR")
    if units and not all(units):
        failures.append("UNIT_ERROR")
    if not checks["evidence_complete"]:
        failures.append("INCOMPLETE_RETRIEVAL")
    if not all(supports) or not checks["answer_format"]:
        failures.append("UNSUPPORTED_CLAIM")
    if not checks["tool_behavior"]:
        failures.append("TOOL_SELECTION_ERROR")
    if not checks["status"] or not checks["stop_reason"]:
        failures.append(
            {
                "timeout": "TIMEOUT",
                "api_error": "API_ERROR",
                "invalid_response": "OUTPUT_VALIDATION_ERROR",
            }.get(state.stop_reason, "LIMIT_EXCEEDED")
        )
    if not checks["abstention"]:
        failures.append("CONTEXT_ERROR")
    if facts and not all(facts) and not numbers:
        failures.append("WRONG_RECORD")
    return {
        "checks": checks,
        "failures": list(dict.fromkeys(failures)),
        "answer_correct": (
            all(facts)
            and checks["abstention"]
            and checks["answer_format"]
            and checks["status"]
        )
        if case.facts
        else None,
        "abstention_correct": checks["abstention"] if case.should_abstain else None,
        "expected_record_count": len(case.expected_ids),
        "numerical_accuracy": mean(numbers),
        "unit_correctness": mean(units),
        "item_selection_correctness": mean(item_selection),
        "numerical_claim_count": len(numbers),
        "unsupported_or_incorrect_numerical_claims": sum(
            not supported for supported in numerical_support
        ),
        "citation_valid_count": sum(validity),
        "citation_count": len(validity),
        "supported_claims": sum(supports),
        "claim_count": len(supports)
        + (
            sum(
                match is None
                or match.group(1).casefold() not in {fact.label for fact in case.facts}
                for match in parsed
            )
            + int(duplicates)
            if case.facts
            else 0
        ),
        "grounded": all(supports)
        and checks["answer_format"]
        and checks["citation_validity"]
        and checks["evidence_complete"]
        if case.facts
        else None,
        "tool_correct": checks["tool_behavior"],
    }
