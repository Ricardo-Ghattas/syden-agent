"""Strict native function definitions generated from validated argument models."""

from openai import pydantic_responses_function_tool
from openai.types.responses import ToolParam
from pydantic import Field, field_validator

from app.retrieval.schemas import Model, SearchFilters


class SearchArguments(Model):
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Keywords, exact item code, or quoted phrase; not a full natural-language question.",
    )
    filters: SearchFilters | None = Field(
        default=None,
        description="AND-composed source/BOQ filters; null for no filtering.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Page size (1–100); exhaustive mode visits subsequent pages automatically.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="First matching result to return; use 0 for exhaustive numerical questions.",
    )
    exhaustive: bool = Field(
        default=False,
        description="Retrieve all keyword matches through existing pagination within the context budget. Inspect complete and warnings before using totals.",
    )

    @field_validator("filters", mode="before")
    @classmethod
    def normalize_filters(cls, value):
        # Provider tool calls sometimes encode nullable controls as strings.
        if isinstance(value, str) and value.strip().casefold() in {"", "null", "none"}:
            return None
        if isinstance(value, dict):
            value = dict(value)
            for field in ("format", "document_id", "kind", "page"):
                item = value.get(field)
                if isinstance(item, str):
                    item = item.strip()
                    if item.casefold() in {"", "null", "none"}:
                        value[field] = None
                    elif field == "page" and item.isdecimal():
                        value[field] = int(item)
        return value


class GetArguments(Model):
    record_id: str = Field(
        min_length=1,
        max_length=256,
        description="Exact record ID returned by search_documents.",
    )


class ListArguments(Model):
    pass


ARGUMENT_MODELS = {
    "search_documents": SearchArguments,
    "get_record": GetArguments,
    "list_documents": ListArguments,
}

DESCRIPTIONS = {
    "search_documents": "Search PDF text, headings, tables and Excel BOQ records with metadata filters. Use for document facts, item codes, quantities, rates and comparisons. Returns source citations and completeness indicators; exhaustive=true gathers all matches within a bounded context.",
    "get_record": "Get the complete original structured record by a known record ID, including all source metadata. Use to inspect full evidence rather than a search projection.",
    "list_documents": "List indexed original filenames, document IDs, worksheets and sections. Use to discover collection scope or resolve ambiguous document names. Does not retrieve BOQ line values.",
}

TOOL_DEFINITIONS: list[ToolParam] = [
    pydantic_responses_function_tool(model, name=name, description=DESCRIPTIONS[name])
    for name, model in ARGUMENT_MODELS.items()
]
