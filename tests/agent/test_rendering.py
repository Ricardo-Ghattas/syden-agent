"""Rendered values are synthetic presentation fixtures, never financial evidence."""

from io import StringIO

import pytest
from rich.console import Console

from app.agents.rendering import render_response, source_labels
from app.agents.schemas import AgentResponse, SourceReference


def source(**changes):
    return SourceReference(
        **{
            "document_id": "document-a",
            "record_id": "record-a",
            "filename": "Example.xlsx",
            "page": None,
            "sheet": "Concrete",
            "row_start": 8,
            "row_end": 8,
            "cell_range": "A8:F8",
            **changes,
        }
    )


def response(answer, **changes):
    return AgentResponse(
        **{
            "answer": answer,
            "answer_type": "document",
            "sources": [source()],
            "warnings": [],
            **changes,
        }
    )


def rendered(value, width=80, color=False):
    stream = StringIO()
    console = Console(
        file=stream,
        width=width,
        force_terminal=color,
        color_system="standard" if color else None,
    )
    render_response(value, console=console)
    return stream.getvalue()


@pytest.mark.parametrize(
    "answer,figure",
    [
        ("**820.75 m³** of RC raft foundation concrete C35/45.", "820.75 m³"),
        ("**USD 123,456.70** — recorded concrete subtotal.", "USD 123,456.70"),
        ("**−2.500 m²** — recorded adjustment.", "−2.500 m²"),
    ],
)
def test_key_figures_preserve_precision_units_and_have_bold_style(answer, figure):
    text = rendered(response(answer))
    assert figure in text
    assert "**" not in text
    assert text.index(figure) < text.index("Sources")
    assert "\x1b[1m" + figure in rendered(response(answer), color=True)


TABLE = """| Item | Quantity | Amount (USD) |
| :--- | ---: | ---: |
| Raft | 820.75 m³ | 123,456.70 |
| Slab | 210.00 m³ | 31,500.00 |"""


def test_comparison_uses_aligned_table_with_no_markdown_delimiters():
    text = rendered(response(TABLE))
    assert "|" not in text
    assert "---:" not in text
    raft = next(line for line in text.splitlines() if "Raft" in line)
    slab = next(line for line in text.splitlines() if "Slab" in line)
    assert raft.index("m³") == slab.index("m³")
    assert raft.index("123,456.70") + 10 == slab.index("31,500.00") + 9


@pytest.mark.parametrize("width", [20, 32, 48, 80])
def test_narrow_terminals_preserve_all_table_values_and_warnings(width):
    value = response(TABLE, warnings=["Partial evidence; total is unverified."])
    text = rendered(value, width=width)
    assert all(len(line) <= width for line in text.splitlines())
    compact = "".join(text.split())
    for token in ["Raft", "Slab", "820.75", "210.00", "123,456.70", "31,500.00"]:
        assert token in compact
    assert "Partialevidence;totalisunverified." in compact
    if width <= 32:
        assert "Quantity:" in text
        assert "Amount (USD):" in text


def test_missing_information_and_material_warnings_stay_visible():
    value = response(
        "The currency is not specified. Provide the priced BOQ to confirm it.",
        answer_type="insufficient",
        warnings=[
            "Conflicting rates were found.",
            "Evidence is incomplete.",
            "Evidence is incomplete.",
        ],
    )
    text = rendered(value)
    assert "The currency is not specified." in text
    assert "Conflicting rates were found." in text
    assert text.count("Evidence is incomplete.") == 1
    assert text.index("Sources") < text.index("Notes")


def test_citations_merge_adjacent_rows_but_keep_gaps_and_documents_distinct():
    sources = [
        source(),
        source(record_id="same-row-different-record"),
        source(row_start=9, row_end=10),
        source(row_start=13, row_end=14),
        source(row_start=12, row_end=12),
        source(sheet="Steel", row_start=8, row_end=8),
        source(document_id="document-b"),
    ]
    value = response("Answer", sources=sources)
    before = value.model_dump_json()
    assert source_labels(sources) == [
        "Example.xlsx, sheet Concrete, rows 8–10, 12–14",
        "Example.xlsx, sheet Steel, row 8",
        "Example.xlsx, sheet Concrete, row 8",
    ]
    assert "rows 8–14" not in rendered(value)
    assert value.model_dump_json() == before
    assert len(value.sources) == 7


def test_citations_keep_pages_and_cell_only_locations():
    sources = [
        source(
            filename="Spec.pdf",
            sheet=None,
            row_start=None,
            row_end=None,
            cell_range=None,
            page=2,
        ),
        source(
            filename="Spec.pdf",
            sheet=None,
            row_start=None,
            row_end=None,
            cell_range=None,
            page=4,
        ),
        source(row_start=None, row_end=None, cell_range="C7"),
    ]
    labels = source_labels(sources)
    assert labels == [
        "Spec.pdf, page 2",
        "Spec.pdf, page 4",
        "Example.xlsx, sheet Concrete, cells C7",
    ]


def test_untrusted_source_names_and_warnings_are_literal_text():
    value = response(
        "Answer",
        sources=[source(filename="[red]BOQ[/red].xlsx")],
        warnings=["[bold]Verify[/bold] the source."],
    )
    text = rendered(value)
    assert "[red]BOQ[/red].xlsx" in text
    assert "[bold]Verify[/bold]" in text
