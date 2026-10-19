import pytest
from pydantic import ValidationError

from app.retrieval import SearchFilters, search_documents


@pytest.mark.parametrize(
    "filters,check",
    [
        ({"format": "pdf"}, lambda r: r.source.format == "pdf"),
        ({"format": "xlsx"}, lambda r: r.source.format == "xlsx"),
        (
            {"filename": "SPECIFICATION.PDF"},
            lambda r: r.source.filename == "specification.pdf",
        ),
        ({"sheet": "structural"}, lambda r: r.source.sheet == "Structural"),
        ({"document_id": "b" * 64}, lambda r: r.document_id == "b" * 64),
        ({"section": "1 concrete"}, lambda r: "1 Concrete" in r.semantic_path),
        ({"page": 2}, lambda r: r.source.page == 2),
        ({"item_code": "003"}, lambda r: r.boq.item == "003"),
    ],
)
def test_actual_source_fields_filter_keyword_results(index, filters, check):
    response = search_documents(index, "concrete", filters)
    assert response.results
    assert all(check(result.record) for result in response.results)


def test_filters_are_and_composed(index):
    response = search_documents(
        index,
        "reinforcement steel",
        SearchFilters(
            format="xlsx",
            filename="boq.xlsx",
            sheet="Structural",
            section="Reinforcement",
            kind="boq_item",
            item_code="2.3",
        ),
    )
    assert response.total == 1
    assert response.results[0].record.source.row_start == 38
    assert (
        search_documents(
            index, "reinforcement", {"format": "pdf", "sheet": "Structural"}
        ).total
        == 0
    )
    assert search_documents(index, "concrete", {"filename": "absent.pdf"}).total == 0


@pytest.mark.parametrize(
    "filters",
    [
        {"category": "concrete"},
        {"document_type": "pdf"},
        {"format": "xls"},
        {"kind": "answer"},
        {"page": 0},
        {"page": True},
        {"page": "2"},
        {"sheet": 123},
        {"sheet": " "},
        {"filename": ""},
        {"document_id": "bad"},
        [],
        "pdf",
        False,
    ],
)
def test_invalid_filters_do_not_silently_change_results(index, filters):
    with pytest.raises(ValidationError):
        search_documents(index, "concrete", filters)
