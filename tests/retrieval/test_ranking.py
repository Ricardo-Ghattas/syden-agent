import pytest
from pydantic import ValidationError

from app.ingestion.schemas import BoqItem, Record, Source
from app.retrieval import DocumentIndex, RankingWeights, search_documents
from app.retrieval.ranking import bm25


def ranking_index(tmp_path, records):
    source = Source(
        filename="spec.pdf",
        raw_path="raw/spec.pdf",
        format="pdf",
        sha256="a" * 64,
        page=1,
    )
    values = [
        Record(
            id=f"{source.sha256}:{i}",
            document_id=source.sha256,
            order_index=i,
            source=source,
            **record,
        )
        for i, record in enumerate(records, 1)
    ]
    (tmp_path / "rank.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in values)
    )
    return DocumentIndex(tmp_path)


def test_exact_code_outweighs_body_mentions_even_with_custom_weights(tmp_path):
    # Row bounds are part of the source contract for BOQ records.
    source = Source(
        filename="boq.xlsx",
        raw_path="raw/boq.xlsx",
        format="xlsx",
        sha256="a" * 64,
        sheet="Structural",
        row_start=3,
        row_end=3,
    )
    records = [
        Record(
            id="mention",
            document_id=source.sha256,
            order_index=1,
            kind="row",
            source=source,
            text="2.3 " * 100,
        ),
        Record(
            id="code",
            document_id=source.sha256,
            order_index=2,
            kind="boq_item",
            source=source,
            text="Excavation",
            boq=BoqItem(item="2.3", description="Excavation"),
        ),
    ]
    (tmp_path / "rank.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records)
    )
    index = DocumentIndex(tmp_path)
    response = search_documents(
        index,
        "2.3",
        weights=RankingWeights(body=10.0, description=0.0, heading=0.0, metadata=0.0),
    )
    assert response.results[0].record_id == "code"
    assert response.results[0].score > response.results[1].score


def test_exact_phrase_beats_reversed_terms(tmp_path):
    index = ranking_index(
        tmp_path,
        [
            {"kind": "paragraph", "text": "Steel reinforcement requires inspection"},
            {"kind": "paragraph", "text": "Reinforcement steel requires inspection"},
        ],
    )
    results = search_documents(index, "reinforcement steel").results
    assert results[0].record.text.startswith("Reinforcement steel")
    assert results[0].score > results[1].score


def test_heading_boost_and_weights_change_relative_priority(tmp_path):
    index = ranking_index(
        tmp_path,
        [
            {"kind": "paragraph", "text": "Concrete"},
            {"kind": "heading", "text": "Concrete", "heading_level": 1},
        ],
    )
    assert search_documents(index, "concrete").results[0].kind == "heading"
    plain = search_documents(index, "concrete", weights=RankingWeights(heading=0.0))
    assert plain.results[0].record.order_index == 1


def test_boq_description_boost_is_configurable(index):
    response = search_documents(index, "reinforcement steel")
    assert response.results[0].kind == "boq_item"
    assert "description" in response.results[0].matched_fields
    assert all(result.score > 0 for result in response.results)
    with pytest.raises(ValidationError):
        RankingWeights(description=-1.0)
    with pytest.raises(ValidationError):
        RankingWeights(body=float("nan"))


def test_equal_scores_have_stable_source_order_across_reload(tmp_path):
    index = ranking_index(tmp_path, [{"kind": "paragraph", "text": "Concrete"}] * 3)
    expected = [
        result.record_id for result in search_documents(index, "concrete").results
    ]
    assert [index.get_record(key).order_index for key in expected] == [1, 2, 3]
    path = tmp_path / "rank.jsonl"
    path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    index.reload()
    assert [
        result.record_id for result in search_documents(index, "concrete").results
    ] == expected


def test_bm25_prefers_rare_terms_shorter_text_and_saturates_frequency():
    assert bm25(1, 10, 20, 100, 1) > bm25(1, 10, 20, 100, 90)
    assert bm25(1, 10, 20, 100, 10) > bm25(1, 100, 20, 100, 10)
    assert bm25(2, 10, 20, 100, 10) < 2 * bm25(1, 10, 20, 100, 10)
    assert bm25(0, 10, 20, 100, 10) == 0
