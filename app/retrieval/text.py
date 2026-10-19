"""Lexical normalization for search views; source record values are never changed."""

import re
import unicodedata
from dataclasses import dataclass

_TOKEN = re.compile(r"[+-]?\w+(?:[./:-]\w+)*%?", re.UNICODE)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "for",
        "to",
        "with",
        "by",
        "from",
        "at",
        "is",
        "are",
        "be",
        "as",
        "it",
        "this",
        "that",
        "all",
        "find",
        "retrieve",
        "show",
        "please",
        "what",
        "which",
        "how",
        "items",
        "item",
        "entries",
        "records",
        "related",
        "associated",
        "given",
        "boq",
    ]
)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokenize(text: str) -> list[str]:
    terms = []
    for token in _TOKEN.findall(normalize_text(text)):
        terms.append(token)
        if "-" in token and any(char.isalpha() for char in token):
            terms.extend(part for part in token.split("-") if part)
    return [term for term in terms if term not in _STOPWORDS]


def contains_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    terms: tuple[str, ...]
    phrases: tuple[str, ...]
    quoted: bool


def parse_query(query: str) -> Query:
    if not isinstance(query, str):
        raise TypeError("Query must be a string")
    text = normalize_text(query)
    if text.count('"') % 2:
        raise ValueError("Query has an unmatched double quote")
    quoted = tuple(normalize_text(value) for value in re.findall(r'"([^\"]*)"', text))
    if any(not phrase for phrase in quoted):
        raise ValueError("Quoted phrases must not be empty")
    terms = tuple(dict.fromkeys(tokenize(text)))
    if not terms:
        raise ValueError("Query must contain searchable words or item codes")
    phrases = quoted or ((text,) if len(terms) > 1 else ())
    return Query(text, terms, phrases, bool(quoted))
