"""Small structured events with argument values and confidential content excluded."""

import json
import logging

from app.retrieval.schemas import SearchFilters

logger = logging.getLogger("app.agents.events")


def event(name: str, **fields) -> None:
    logger.info(json.dumps({"event": name, **fields}, separators=(",", ":")))


def argument_summary(arguments: str) -> dict:
    try:
        value = json.loads(arguments)
    except ValueError:
        return {"valid_json": False}
    if not isinstance(value, dict):
        return {"valid_json": False}
    return {
        "query_chars": len(value["query"])
        if isinstance(value.get("query"), str)
        else 0,
        "filter_fields": sorted(set(value["filters"]) & set(SearchFilters.model_fields))
        if isinstance(value.get("filters"), dict)
        else [],
        "limit": value.get("limit") if type(value.get("limit")) is int else None,
        "offset": value.get("offset") if type(value.get("offset")) is int else None,
        "exhaustive": value.get("exhaustive")
        if type(value.get("exhaustive")) is bool
        else None,
        "record_id_present": isinstance(value.get("record_id"), str),
    }
