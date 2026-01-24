"""
Retrieval domain models.

These models represent search queries and results in the RAG system.
The retrieval system returns grounding context, never answers.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from genai_rag_service.domain.document import DocumentChunk


@dataclass(frozen=True)
class SearchQuery:
    """
    A search request to the RAG system.
    
    Search queries support both vector-only and hybrid (vector + keyword) search.
    Filters can be applied to narrow results based on metadata.
    
    Attributes:
        query_text: The natural language query
        top_k: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0.0 to 1.0)
        filters: Metadata filters to apply (key-value matching)
        search_type: Type of search to perform
        keyword_weight: Weight for keyword matching in hybrid search (0.0 to 1.0)
    """
    
    query_text: str
    top_k: int = 10
    similarity_threshold: float = 0.0
    filters: Mapping[str, Any] = field(default_factory=dict)
    search_type: Literal["vector", "hybrid"] = "vector"
    keyword_weight: float = 0.3
    
    def __post_init__(self) -> None:
        """Validate query on creation."""
        if not self.query_text:
            raise ValueError("query_text cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be between 0.0 and 1.0")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_text": self.query_text,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "filters": dict(self.filters),
            "search_type": self.search_type,
            "keyword_weight": self.keyword_weight,
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchQuery":
        """Deserialize from dictionary."""
        return cls(
            query_text=data["query_text"],
            top_k=data.get("top_k", 10),
            similarity_threshold=data.get("similarity_threshold", 0.0),
            filters=data.get("filters", {}),
            search_type=data.get("search_type", "vector"),
            keyword_weight=data.get("keyword_weight", 0.3),
        )


@dataclass(frozen=True)
class SearchResult:
    """
    A single search result from the RAG system.
    
    Results include the matched chunk, similarity score, and match type.
    Chunks are returned in order of relevance.
    
    Attributes:
        chunk: The matched document chunk
        score: Similarity/relevance score (0.0 to 1.0)
        match_type: How the match was found
    """
    
    chunk: DocumentChunk
    score: float
    match_type: Literal["vector", "keyword", "hybrid"]
    
    def __post_init__(self) -> None:
        """Validate result on creation."""
        if self.score < 0.0:
            raise ValueError("score must be non-negative")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "match_type": self.match_type,
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchResult":
        """Deserialize from dictionary."""
        return cls(
            chunk=DocumentChunk.from_dict(data["chunk"]),
            score=data["score"],
            match_type=data["match_type"],
        )


@dataclass(frozen=True)
class RetrievalContext:
    """
    The grounding context returned from retrieval.
    
    This is what the RAG service returns to callers - ranked chunks with
    metadata about the retrieval operation. This is NOT an answer;
    it is context for downstream processing.
    
    Attributes:
        results: Ranked search results (most relevant first)
        query: The original search query
        latency_ms: Time taken for retrieval in milliseconds
        total_chunks_searched: Number of chunks considered
    """
    
    results: tuple[SearchResult, ...]
    query: SearchQuery
    latency_ms: float
    total_chunks_searched: int
    
    def __post_init__(self) -> None:
        """Validate context on creation."""
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.total_chunks_searched < 0:
            raise ValueError("total_chunks_searched must be non-negative")
    
    @property
    def result_count(self) -> int:
        """Return the number of results."""
        return len(self.results)
    
    @property
    def has_results(self) -> bool:
        """Return whether any results were found."""
        return len(self.results) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "results": [result.to_dict() for result in self.results],
            "query": self.query.to_dict(),
            "latency_ms": self.latency_ms,
            "total_chunks_searched": self.total_chunks_searched,
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalContext":
        """Deserialize from dictionary."""
        return cls(
            results=tuple(
                SearchResult.from_dict(result) for result in data.get("results", [])
            ),
            query=SearchQuery.from_dict(data["query"]),
            latency_ms=data["latency_ms"],
            total_chunks_searched=data["total_chunks_searched"],
        )
    
    @classmethod
    def empty(cls, query: SearchQuery, latency_ms: float = 0.0) -> "RetrievalContext":
        """Create an empty context (no results)."""
        return cls(
            results=(),
            query=query,
            latency_ms=latency_ms,
            total_chunks_searched=0,
        )


__all__ = [
    "SearchQuery",
    "SearchResult",
    "RetrievalContext",
]
