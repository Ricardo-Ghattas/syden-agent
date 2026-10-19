"""Local lexical retrieval; no agent, provider, or ingestion pipeline dependency."""

from .indexing import DocumentIndex
from .schemas import RankingWeights, SearchFilters, SearchResponse
from .search import search_documents

__all__ = [
    "DocumentIndex",
    "RankingWeights",
    "SearchFilters",
    "SearchResponse",
    "search_documents",
]
