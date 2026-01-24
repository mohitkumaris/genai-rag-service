"""
Domain models and Port interfaces for the RAG service.

This file consolidates all core domain entities and strict abstract interfaces (Ports).
Core logic in `rag/` must ONLY depend on these definitions, never on `adapters/` or `app/`.
"""

from typing import Protocol, Any, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from collections.abc import Sequence, Mapping


# =============================================================================
# DOMAIN MODELS
# =============================================================================

@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata for a document."""
    source_uri: str
    content_hash: str
    version: str = "1.0.0"
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    custom_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_uri:
            raise ValueError("source_uri cannot be empty")
        if not self.content_hash:
            raise ValueError("content_hash cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
            "version": self.version,
            "ingested_at": self.ingested_at.isoformat(),
            **self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        meta = data.copy()
        ingested_at = datetime.fromisoformat(meta.pop("ingested_at"))
        return cls(
            source_uri=meta.pop("source_uri"),
            content_hash=meta.pop("content_hash"),
            version=meta.pop("version", "1.0.0"),
            ingested_at=ingested_at,
            custom_metadata=meta,
        )


@dataclass(frozen=True)
class DocumentChunk:
    """An atomic chunk of text for retrieval."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: DocumentMetadata

    @classmethod
    def create(
        cls,
        document_id: str,
        content: str,
        chunk_index: int,
        token_count: int,
        metadata: DocumentMetadata,
    ) -> "DocumentChunk":
        # Deterministic ID for idempotency: hash(doc_id + index + content)
        composite = f"{document_id}:{chunk_index}:{content}"
        chunk_id = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:32]
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            token_count=token_count,
            metadata=metadata,
        )


@dataclass(frozen=True)
class Document:
    """A complete document."""
    document_id: str
    chunks: tuple[DocumentChunk, ...]
    metadata: DocumentMetadata

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class EmbeddingVector:
    """An embedding vector."""
    chunk_id: str
    vector: tuple[float, ...]
    model_id: str

    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass(frozen=True)
class SearchQuery:
    """A search query."""
    query_text: str
    top_k: int = 10
    search_type: str = "vector"  # vector, hybrid
    similarity_threshold: float = 0.0
    filters: Mapping[str, Any] | None = None
    keyword_weight: float = 0.3  # For hybrid search


@dataclass(frozen=True)
class SearchResult:
    """A single search result."""
    chunk: DocumentChunk
    score: float
    match_type: str  # vector, keyword, hybrid


@dataclass(frozen=True)
class RetrievalContext:
    """The result of a retrieval operation (grounding context)."""
    results: tuple[SearchResult, ...]
    query: SearchQuery
    latency_ms: float
    total_chunks_searched: int

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0


# =============================================================================
# PORTS (Interfaces)
# =============================================================================

@runtime_checkable
class VectorStorePort(Protocol):
    """Port for vector storage and retrieval."""
    
    async def upsert_vectors(
        self,
        vectors: Sequence[EmbeddingVector],
        chunks: Sequence[DocumentChunk],
    ) -> int: ...

    async def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[str, float]]: ...

    async def hybrid_search(
        self,
        query_vector: tuple[float, ...],
        query_text: str,
        top_k: int,
        vector_weight: float = 0.7,
        filters: Mapping[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[str, float]]: ...

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]: ...
    
    async def document_exists(self, document_id: str) -> bool: ...
    
    async def delete_by_document(self, document_id: str) -> int: ...
    
    async def get_chunk_count(self) -> int: ...


@runtime_checkable
class EmbeddingProviderPort(Protocol):
    """Port for embedding generation."""
    
    async def embed_texts(
        self,
        texts: Sequence[str],
        model_id: str,
    ) -> list[tuple[float, ...]]: ...
    
    async def embed_query(
        self,
        query: str,
        model_id: str,
    ) -> tuple[float, ...]: ...


@runtime_checkable
class BlobStoragePort(Protocol):
    """Port for raw document storage."""
    
    async def get_document(self, uri: str) -> bytes: ...
    async def put_document(self, uri: str, content: bytes) -> None: ...
    async def exists(self, uri: str) -> bool: ...
    async def list_documents(self, prefix: str = "") -> list[str]: ...
