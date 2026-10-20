import json
import logging

import pytest

from app.agents.schemas import AnswerContent
from app.tools.registry import ToolRegistry
from app.tools.schemas import TOOL_DEFINITIONS

from .conftest import search_args


def assert_strict(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema.get("properties", {}))
        for value in schema.values():
            assert_strict(value)
    elif isinstance(schema, list):
        for value in schema:
            assert_strict(value)


def test_native_schemas_are_strict_and_nullable():
    names = set()
    for tool in TOOL_DEFINITIONS:
        assert tool["type"] == "function"
        names.add(tool["name"])
        assert tool["strict"] is True
        assert tool.get("description")
        assert_strict(tool["parameters"])
    assert names == {"search_documents", "get_record", "list_documents"}
    assert_strict(AnswerContent.model_json_schema())
    search = next(
        tool for tool in TOOL_DEFINITIONS if tool["name"] == "search_documents"
    )
    kinds = search["parameters"]["$defs"]["SearchFilters"]["properties"]["kind"][
        "anyOf"
    ][0]["enum"]
    assert set(kinds) == {
        "document",
        "sheet",
        "region",
        "heading",
        "paragraph",
        "table",
        "row",
        "boq_item",
    }


@pytest.mark.parametrize(
    ("name", "arguments", "code"),
    [
        ("eval", "{}", "unknown_tool"),
        ("search_documents", "{", "invalid_arguments"),
        ("search_documents", "[]", "invalid_arguments"),
        ("search_documents", "null", "invalid_arguments"),
        ("search_documents", json.dumps(search_args(limit=True)), "invalid_arguments"),
        ("search_documents", json.dumps(search_args(limit=101)), "invalid_arguments"),
        (
            "search_documents",
            json.dumps(search_args(filters={"kind": "unknown"})),
            "invalid_arguments",
        ),
        ("search_documents", json.dumps(search_args(offset=-1)), "invalid_arguments"),
        (
            "search_documents",
            json.dumps(search_args(extra="oops")),
            "invalid_arguments",
        ),
        ("search_documents", "{}", "invalid_arguments"),
        ("list_documents", '{"path":"/etc/passwd"}', "invalid_arguments"),
        ("get_record", '{"record_id":42}', "invalid_arguments"),
        ("get_record", "x" * 16001, "invalid_arguments"),
    ],
)
def test_invalid_calls_are_structured_and_do_not_supply_sources(
    index, name, arguments, code
):
    registry = ToolRegistry(index)
    output = json.loads(registry.execute(name, arguments))
    assert output == {
        "ok": False,
        "error": {"code": code, "message": output["error"]["message"]},
    }
    assert not registry.sources


def test_each_registered_operation_preserves_values_and_sources(index):
    registry = ToolRegistry(index)
    result = json.loads(
        registry.execute("search_documents", json.dumps(search_args()))
    )["data"]
    assert result["complete"]
    assert result["total"] == result["returned"] == 1
    record = result["results"][0]["record"]
    assert record["boq"]["quantity"] == -2.5
    assert record["boq"]["item"] == "2.3"
    citation = result["results"][0]["citation"]
    assert citation["filename"] == "boq.xlsx"
    assert citation["sheet"] == "Structural"
    assert citation["row_start"] == 38
    full = json.loads(
        registry.execute("get_record", json.dumps({"record_id": record["id"]}))
    )["data"]
    original = index.get_record(record["id"])
    assert original is not None
    assert full["record"] == original.model_dump(mode="json")
    assert full["citation"] == citation
    documents = json.loads(registry.execute("list_documents", "{}"))["data"]
    assert len(documents["documents"]) == 2
    assert documents["index_complete"]
    assert all(item["citation"]["record_id"] is None for item in documents["documents"])
    missing = json.loads(registry.execute("get_record", '{"record_id":"missing"}'))[
        "data"
    ]
    assert missing["record"] is None
    assert missing["citation"] is None
    assert not missing["found"]


def test_optional_search_controls_use_safe_defaults(index):
    payload = json.loads(
        ToolRegistry(index).execute("search_documents", '{"query":"2.3"}')
    )
    assert payload["ok"]
    assert payload["data"]["offset"] == 0
    assert payload["data"]["returned"] == 1


def test_validation_error_explains_bounds_without_echoing_private_arguments(index):
    arguments = search_args(
        query="PRIVATE QUERY", limit=101, **{"PRIVATE FIELD": "PRIVATE VALUE"}
    )
    payload = json.loads(
        ToolRegistry(index).execute("search_documents", json.dumps(arguments))
    )
    assert not payload["ok"]
    assert "limit" in payload["error"]["message"]
    assert "100" in payload["error"]["message"]
    assert "PRIVATE" not in json.dumps(payload)


def test_invalid_filter_types_get_specific_correction_feedback(index):
    payload = json.loads(
        ToolRegistry(index).execute(
            "search_documents",
            json.dumps(search_args(filters={"kind": "paragraphs", "page": "all"})),
        )
    )
    message = payload["error"]["message"]
    assert "filters.kind" in message
    assert "paragraph" in message
    assert "filters.page" in message
    assert "integer" in message
    assert "JSON null" in message


def test_numeric_page_strings_preserve_the_requested_filter(index):
    payload = json.loads(
        ToolRegistry(index).execute(
            "search_documents",
            json.dumps(
                search_args(
                    query="35 MPa",
                    filters={"format": "pdf", "page": "2", "kind": " paragraph "},
                )
            ),
        )
    )
    assert payload["ok"]
    assert payload["data"]["filters"]["page"] == 2
    assert payload["data"]["results"]
    assert all(
        item["record"]["source"]["page"] == 2 for item in payload["data"]["results"]
    )


@pytest.mark.parametrize("page", [True, False, 2.5, "2.5", "0", "-1"])
def test_page_compatibility_keeps_invalid_values_rejected(index, page):
    payload = json.loads(
        ToolRegistry(index).execute(
            "search_documents", json.dumps(search_args(filters={"page": page}))
        )
    )
    assert not payload["ok"]
    assert payload["error"]["code"] == "invalid_arguments"


def test_tool_null_controls_are_normalized_without_changing_named_worksheets(index):
    payload = json.loads(
        ToolRegistry(index).execute(
            "search_documents",
            json.dumps(
                search_args(
                    query="excavation",
                    filters={
                        "format": "xlsx",
                        "document_id": "null",
                        "page": "None",
                        "kind": "boq_item",
                    },
                )
            ),
        )
    )
    assert payload["ok"]
    assert payload["data"]["filters"]["document_id"] is None
    assert payload["data"]["filters"]["page"] is None
    assert payload["data"]["results"]
    from app.tools.schemas import SearchArguments

    parsed = SearchArguments.model_validate_json(
        json.dumps(search_args(filters={"sheet": "None"}))
    )
    assert parsed.filters.sheet == "None"


def test_exhaustive_search_uses_existing_pagination(index):
    registry = ToolRegistry(index, 300000)
    data = json.loads(
        registry.execute(
            "search_documents",
            json.dumps(
                search_args(
                    query="excav",
                    filters={"kind": "boq_item"},
                    limit=3,
                    exhaustive=True,
                )
            ),
        )
    )["data"]
    assert data["total"] == data["returned"] == 35
    assert data["complete"]
    assert data["next_offset"] is None
    assert len(registry.sources) == 35


def test_partial_and_nonzero_offset_results_warn(index):
    registry = ToolRegistry(index)
    data = json.loads(
        registry.execute(
            "search_documents", json.dumps(search_args(query="excav", limit=2))
        )
    )["data"]
    assert data["has_more"]
    assert data["next_offset"] == 2
    assert not data["complete"]
    assert data["warnings"]
    later = json.loads(
        registry.execute(
            "search_documents",
            json.dumps(search_args(query="excav", offset=34, exhaustive=True)),
        )
    )["data"]
    assert not later["has_more"]
    assert not later["complete"]


def test_budget_never_cuts_records_or_authorizes_hidden_sources(index):
    registry = ToolRegistry(index, 7000)
    output = registry.execute(
        "search_documents", json.dumps(search_args(query="excav", exhaustive=True))
    )
    data = json.loads(output)["data"]
    assert len(output.encode()) <= 7000
    assert data["returned"] < data["total"]
    assert not data["complete"]
    assert len(registry.sources) == data["returned"]
    for item in data["results"]:
        original = index.get_record(item["record_id"])
        assert original is not None
        assert item["record"]["boq"] == original.model_dump(mode="json")["boq"]
    assert any("budget" in warning for warning in data["warnings"])


def test_oversized_full_record_returns_error_without_citation(index):
    record_id = "b" * 64 + ":39"
    record = index.entries[record_id].record
    record.text = "x" * 5000
    registry = ToolRegistry(index, 2048)
    output = json.loads(
        registry.execute("get_record", json.dumps({"record_id": record_id}))
    )
    assert output["error"]["code"] == "context_limit"
    assert not registry.sources


def test_retrieval_exception_does_not_leak_payload(index, monkeypatch, caplog):
    def fail():
        raise OSError("SECRET CONTENT AND PATH")

    monkeypatch.setattr(index, "list_documents", fail)
    registry = ToolRegistry(index)
    with caplog.at_level(logging.INFO):
        output = registry.execute("list_documents", "{}")
    assert json.loads(output)["error"]["code"] == "retrieval_failed"
    assert "SECRET" not in output + caplog.text
    assert "success=False" in caplog.text


def test_incomplete_index_is_reported(tmp_path):
    from app.retrieval import DocumentIndex

    (tmp_path / "invalid.jsonl").write_text("bad json\n")
    registry = ToolRegistry(DocumentIndex(tmp_path))
    data = json.loads(
        registry.execute("search_documents", json.dumps(search_args(exhaustive=True)))
    )["data"]
    assert not data["complete"]
    assert not data["index_complete"]
    assert data["warnings"]


def test_no_matches_is_explicit(index):
    data = json.loads(
        ToolRegistry(index).execute(
            "search_documents", json.dumps(search_args(query="unmatched_xyz"))
        )
    )["data"]
    assert data["complete"]
    assert data["returned"] == data["total"] == 0
