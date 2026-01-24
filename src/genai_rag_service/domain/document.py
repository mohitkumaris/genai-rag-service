"""
Document domain models.

These models represent documents and their chunks in the RAG system.
All models are immutable for safe concurrent access and predictable behavior.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib


def _generate_content_hash(content: str) -> str:
    """Generate a deterministic SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _generate_chunk_id(document_id: str, chunk_index: int, content_hash: str) -> str:
    """
    Generate a deterministic chunk ID.
    
    The ID is derived from the document ID, chunk position, and content hash.
    This ensures that re-chunking the same document produces identical IDs.
    """
    components = f"{document_id}:{chunk_index}:{content_hash[:16]}"
    return hashlib.sha256(components.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class DocumentMetadata:
    """
    Immutable metadata about a document's origin and versioning.
    
    This metadata is attached to documents and their chunks for traceability.
    
    Attributes:
        source_uri: Original location of the document (e.g., blob URL, file path)
        content_hash: SHA-256 hash of raw content for idempotency detection
        version: User-defined version string for tracking updates
        ingested_at: UTC timestamp when the document was processed
        custom_metadata: Extensible key-value pairs for filtering and context
    """
    
    source_uri: str
    content_hash: str
    version: str = "1.0.0"
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    custom_metadata: Mapping[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metadata on creation."""
        if not self.source_uri:
            raise ValueError("source_uri cannot be empty")
        if not self.content_hash:
            raise ValueError("content_hash cannot be empty")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage/logging."""
        return {
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
            "version": self.version,
            "ingested_at": self.ingested_at.isoformat(),
            "custom_metadata": dict(self.custom_metadata),
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DocumentMetadata":
        """Deserialize from dictionary."""
        ingested_at = data.get("ingested_at")
        if isinstance(ingested_at, str):
            ingested_at = datetime.fromisoformat(ingested_at)
        elif ingested_at is None:
            ingested_at = datetime.now(timezone.utc)
        
        return cls(
            source_uri=data["source_uri"],
            content_hash=data["content_hash"],
            version=data.get("version", "1.0.0"),
            ingested_at=ingested_at,
            custom_metadata=data.get("custom_metadata", {}),
        )


@dataclass(frozen=True)
class DocumentChunk:
    """
    A single chunk of a document, ready for embedding and retrieval.
    
    Chunks are the atomic unit of retrieval in the RAG system.
    Each chunk has a deterministic ID derived from its content and position.
    
    Attributes:
        chunk_id: Unique, deterministic identifier
        document_id: Reference to parent document
        content: The actual text content of this chunk
        chunk_index: Zero-indexed position within the document
        token_count: Number of tokens (for context window management)
        metadata: Inherited metadata from parent document
    """
    
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: DocumentMetadata
    
    def __post_init__(self) -> None:
        """Validate chunk on creation."""
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not self.document_id:
            raise ValueError("document_id cannot be empty")
        if not self.content:
            raise ValueError("content cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage/logging."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "metadata": self.metadata.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DocumentChunk":
        """Deserialize from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            content=data["content"],
            chunk_index=data["chunk_index"],
            token_count=data["token_count"],
            metadata=DocumentMetadata.from_dict(data["metadata"]),
        )
    
    @classmethod
    def create(
        cls,
        document_id: str,
        content: str,
        chunk_index: int,
        token_count: int,
        metadata: DocumentMetadata,
    ) -> "DocumentChunk":
        """
        Factory method to create a chunk with a deterministic ID.
        
        The chunk ID is computed from the document ID, index, and content hash.
        This ensures idempotent re-chunking.
        """
        content_hash = _generate_content_hash(content)
        chunk_id = _generate_chunk_id(document_id, chunk_index, content_hash)
        
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
    """
    A document in the RAG system.
    
    Documents are the top-level container for content. They are split into
    chunks for embedding and retrieval.
    
    Attributes:
        document_id: Unique identifier for the document
        chunks: Immutable sequence of document chunks
        metadata: Document-level metadata
    """
    
    document_id: str
    chunks: tuple[DocumentChunk, ...]
    metadata: DocumentMetadata
    
    def __post_init__(self) -> None:
        """Validate document on creation."""
        if not self.document_id:
            raise ValueError("document_id cannot be empty")
    
    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in this document."""
        return len(self.chunks)
    
    @property
    def total_tokens(self) -> int:
        """Return the total token count across all chunks."""
        return sum(chunk.token_count for chunk in self.chunks)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage/logging."""
        return {
            "document_id": self.document_id,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        """Deserialize from dictionary."""
        return cls(
            document_id=data["document_id"],
            chunks=tuple(
                DocumentChunk.from_dict(chunk) for chunk in data.get("chunks", [])
            ),
            metadata=DocumentMetadata.from_dict(data["metadata"]),
        )
    
    @classmethod
    def create(
        cls,
        source_uri: str,
        content: str,
        version: str = "1.0.0",
        custom_metadata: Mapping[str, Any] | None = None,
    ) -> "Document":
        """
        Factory method to create a document without chunks.
        
        The document ID is a hash of the source URI and content hash.
        Chunks should be added using the chunking service.
        """
        content_hash = _generate_content_hash(content)
        document_id = _generate_chunk_id(source_uri, 0, content_hash)
        
        metadata = DocumentMetadata(
            source_uri=source_uri,
            content_hash=content_hash,
            version=version,
            custom_metadata=custom_metadata or {},
        )
        
        return cls(
            document_id=document_id,
            chunks=(),
            metadata=metadata,
        )


# Re-export utility functions for use elsewhere
__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "_generate_content_hash",
    "_generate_chunk_id",
]
