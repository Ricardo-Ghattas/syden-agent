"""Terminal presentation only; response values and source metadata stay untouched."""

from collections import defaultdict
from typing import ClassVar

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown, MarkdownElement, TableElement
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .schemas import AgentResponse, AgentStatus, SourceReference

THEME = Theme(
    {
        "markdown.strong": "bold",
        "markdown.h1": "bold blue",
        "markdown.h2": "bold blue",
        "markdown.h3": "bold blue",
        "markdown.table.header": "bold",
        "markdown.table.border": "dim",
        "markdown.link": "underline blue",
    }
)


class CompactTable(TableElement):
    """Use parsed Markdown cells; stack labelled values when columns won't fit."""

    def __rich_console__(self, console, options):
        headers = self.header.row.cells if self.header and self.header.row else []
        rows = self.body.rows if self.body else []
        minimum = sum(max(12, cell.content.cell_len) for cell in headers)
        minimum += max(0, len(headers) - 1) * 2
        if options.max_width < minimum:
            for index, row in enumerate(rows):
                if index:
                    yield Text("")
                for header, cell in zip(headers, row.cells):
                    line = Text.assemble((header.content.plain + ": ", "dim"))
                    line.append_text(cell.content)
                    yield line
            return
        table = Table(
            box=box.SIMPLE_HEAD,
            padding=(0, 1),
            pad_edge=False,
            collapse_padding=True,
            border_style="dim",
            show_edge=False,
        )
        for header in headers:
            table.add_column(
                header.content,
                header_style="bold",
                justify=header.justify if header.justify != "default" else "left",
                overflow="fold",
            )
        for row in rows:
            table.add_row(*(cell.content for cell in row.cells))
        yield table


class AnswerMarkdown(Markdown):
    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "table_open": CompactTable,
    }


def source_labels(sources: list[SourceReference]) -> list[str]:
    grouped = defaultdict(list)
    for source in sources:
        key = (
            source.document_id,
            source.filename,
            source.page,
            source.sheet,
            source.cell_range if source.row_start is None else None,
        )
        if source.row_start is not None:
            grouped[key].append((source.row_start, source.row_end or source.row_start))
        else:
            grouped[key]
    labels = []
    for (_, filename, page, sheet, cells), ranges in grouped.items():
        location = [filename]
        if page is not None:
            location.append(f"page {page}")
        if sheet is not None:
            location.append(f"sheet {sheet}")
        merged = []
        for start, end in sorted(set(ranges)):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        if merged:
            label = (
                "row" if len(merged) == 1 and merged[0][0] == merged[0][1] else "rows"
            )
            spans = ", ".join(
                str(start) if start == end else f"{start}–{end}"
                for start, end in merged
            )
            location.append(f"{label} {spans}")
        elif cells:
            location.append(f"cells {cells}")
        labels.append(", ".join(location))
    return labels


def sources_renderable(sources: list[SourceReference]) -> Group:
    documents = defaultdict(list)
    for source in sources:
        documents[(source.document_id, source.filename)].append(source)
    lines = []
    for (_, filename), references in documents.items():
        lines.append(Text(filename, style="dim"))
        for label in source_labels(references):
            location = label[len(filename) :].removeprefix(", ")
            if location:
                lines.append(Padding(Text(location, style="dim"), (0, 0, 0, 2)))
    return Group(*lines)


def response_renderable(response: AgentResponse, *, debug: bool = False) -> Group:
    parts = [AnswerMarkdown(response.answer, hyperlinks=False)]
    if response.sources:
        parts.extend(
            [
                Text(""),
                Text("Sources", style="bold blue"),
                sources_renderable(response.sources),
            ]
        )
    if response.warnings:
        parts.extend([Text(""), Text("Notes", style="bold blue")])
        parts.extend(
            Text("• " + warning) for warning in dict.fromkeys(response.warnings)
        )
    if response.stop_reason is not None and (
        debug or response.status != AgentStatus.COMPLETED
    ):
        parts.extend(
            [
                Text(""),
                Text(
                    f"Status: {response.status.value}; reason: {response.stop_reason.value}",
                    style="dim",
                ),
            ]
        )
    return Group(*parts)


def render_response(
    response: AgentResponse, *, console: Console | None = None, debug: bool = False
) -> None:
    console = console if console is not None else Console()
    with console.use_theme(THEME):
        console.print()
        console.print(response_renderable(response, debug=debug))
        console.print()
