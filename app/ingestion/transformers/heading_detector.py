"""Identify headings from explicit labels and repeated numbering patterns."""

from __future__ import annotations

import re
from collections import Counter

from ..extractors.pdf_models import (
    ConvertedDocument,
    DocumentBlock,
    DocumentBlockRole,
    DocumentBlockType,
)

_TOC_HEADING = re.compile(
    r"^(?:table of contents|contents|content|contents at a glance|المحتويات|الفهرس|فهرس المحتويات|table des matières|sommaire)$",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*|(?:chapter|part|appendix)\s+[\w.-]+)\b", re.IGNORECASE
)


class HeadingDetector:
    """Select headings from provider labels and repeated explicit numbering."""

    def detect(
        self, document: ConvertedDocument, excluded_pages: tuple[int, ...] = ()
    ) -> tuple[DocumentBlock, ...]:
        excluded = set(excluded_pages)
        candidates = [
            block
            for block in document.blocks
            if block.page_number not in excluded
            and block.role is DocumentBlockRole.BODY
            and block.block_type
            not in {DocumentBlockType.TABLE, DocumentBlockType.FIGURE}
            and (
                block.block_type is not DocumentBlockType.LIST_ITEM or block.is_heading
            )
            and len(block.text) <= 120
            and len(block.text.split()) <= 14
            and not block.text.rstrip().endswith((".", "!", "?"))
            and not _TOC_HEADING.fullmatch(block.text.strip())
        ]
        # An explicit heading level remains evidence even when the boolean label is missing.
        labelled = [
            block
            for block in candidates
            if block.is_heading
            or block.heading_level is not None
            or block.block_type is DocumentBlockType.HEADING
        ]
        numbered = [
            block for block in candidates if _NUMBERED_HEADING.match(block.text)
        ]
        numbered_pages = Counter(block.page_number for block in numbered)
        # Require numbering on multiple pages to avoid promoting isolated numbered paragraphs.
        inferred = numbered if len(numbered) >= 2 and len(numbered_pages) >= 2 else []
        selected = {block.id: block for block in (*labelled, *inferred)}
        return tuple(sorted(selected.values(), key=lambda block: block.order_index))
