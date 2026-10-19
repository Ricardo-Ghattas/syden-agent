import pytest

from app.retrieval import search_documents
from app.retrieval.text import tokenize

B = "b" * 64


def test_queries_are_case_and_whitespace_insensitive(index):
    first = search_documents(index, "reinforcement steel")
    second = search_documents(index, "  ReInFoRcEmEnT\n STEEL  ")
    assert [result.record_id for result in first.results] == [
        result.record_id for result in second.results
    ]
    assert [result.score for result in first.results] == [
        result.score for result in second.results
    ]


def test_exact_item_codes_and_leading_zeroes_are_preserved(index):
    response = search_documents(index, "Retrieve BOQ item 2.3")
    assert response.total == 1
    assert response.results[0].record_id == f"{B}:39"
    assert response.results[0].matched_fields[0] == "item_code"
    assert search_documents(index, "2.30").results[0].record_id == f"{B}:40"
    item = search_documents(index, "003").results[0].record.boq
    assert item is not None
    assert item.item == "003"
    assert f"{B}:41" not in {
        r.record_id for r in search_documents(index, "3", limit=100).results
    }


def test_terms_and_numerical_identifiers_remain_searchable(index):
    terms = tokenize("C30/37 RC-25/30 2.3 001 -2.5 35 MPa")
    assert all(
        value in terms
        for value in ["c30/37", "rc-25/30", "2.3", "001", "-2.5", "35", "mpa"]
    )
    result = search_documents(index, "C30/37")
    assert result.total == 2
    assert any(r.record.source.page == 2 for r in result.results)
    assert any(r.record.source.row_start == 40 for r in result.results)
    assert search_documents(index, '"35 MPa"').results[0].record.source.page == 2
    assert search_documents(index, "1234.567890").results[0].record_id == f"{B}:39"


def test_prefix_queries_find_all_relevant_boq_rows(index):
    response = search_documents(index, "excav", {"kind": "boq_item"})
    assert response.total == 35
    for result in response.results:
        assert result.record.boq is not None
        description = result.record.boq.description
        assert isinstance(description, str)
        assert "excavation" in description.casefold()
    assert search_documents(index, "reinforc", {"kind": "boq_item"}).total == 2


def test_quoted_phrase_is_required_while_unquoted_terms_use_or(index):
    phrase = search_documents(index, '"reinforcement steel"')
    assert phrase.total == 3
    assert all(
        "reinforcement steel" in (r.record.text.casefold()) for r in phrase.results
    )
    reverse = search_documents(index, '"steel reinforcement"')
    assert reverse.total == 0
    assert search_documents(index, "steel reinforcement").total > 0


def test_exhaustive_pagination_has_no_gaps_or_duplicates(index):
    found = []
    offset = 0
    while True:
        response = search_documents(
            index,
            "Find excavation-related items",
            {"kind": "boq_item"},
            limit=10,
            offset=offset,
        )
        assert response.total == 35
        assert response.returned == len(response.results)
        found.extend(result.record_id for result in response.results)
        if not response.has_more:
            assert response.next_offset is None
            break
        assert response.next_offset is not None
        assert response.next_offset == offset + response.returned
        offset = response.next_offset
    assert len(found) == len(set(found)) == 35
    assert set(found) == {f"{B}:{i}" for i in range(4, 39)}
    assert (
        search_documents(index, "excavation", {"kind": "boq_item"}, limit=100).total
        == 35
    )
    beyond = search_documents(index, "excavation", offset=1000)
    assert beyond.results == [] and not beyond.has_more


@pytest.mark.parametrize("query", ["", " \n ", "?!", "find all items", '"', '""'])
def test_invalid_queries_raise_clear_errors(index, query):
    with pytest.raises(ValueError):
        search_documents(index, query)


@pytest.mark.parametrize(
    "limit,offset",
    [(0, 0), (-1, 0), (1, -1), (True, 0), (1, False), (1.5, 0), (1, "0")],
)
def test_invalid_pagination_is_rejected(index, limit, offset):
    with pytest.raises(ValueError, match="limit"):
        search_documents(index, "concrete", limit=limit, offset=offset)


def test_no_match_and_index_not_reparsed_for_each_query(index, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Queries must not reopen JSONL files")

    monkeypatch.setattr("pathlib.Path.open", fail)
    assert search_documents(index, "unrelatedavionics").total == 0
    assert search_documents(index, "concrete").total > 0
    assert search_documents(index, "steel").total > 0
    assert index.get_record(f"{B}:39") is not None
    assert len(index.list_documents()) == 2


def test_search_results_cannot_mutate_the_cached_source(index):
    first = search_documents(index, "2.3")
    assert first.results[0].record.boq is not None
    first.results[0].record.boq.quantity = 9999
    second = search_documents(index, "2.3")
    assert second.results[0].record.boq is not None
    assert second.results[0].record.boq.quantity == -2.5
    assert second.results[0].record.source.sheet == "Structural"
    assert second.results[0].record.source.cell_range == "A38:F38"
