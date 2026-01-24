"""
Ingestion domain models.

These models represent ingestion requests and results for batch document processing.
Ingestion is designed to be idempotent - re-ingesting the same document is a no-op.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IngestionRequest:
    """
    A request to ingest one or more documents.
    
    Ingestion requests support batch processing with configurable options.
    Each document in the batch is processed independently.
    
    Attributes:
        documents: Sequence of documents to ingest
        chunking_strategy: Strategy to use for chunking
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
    """
    
    documents: Sequence[Mapping[str, Any]]
    chunking_strategy: str = "fixed_size"
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    def __post_init__(self) -> None:
        """Validate request on creation."""
        if not self.documents:
            raise ValueError("documents cannot be empty")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    @property
    def document_count(self) -> int:
        """Return the number of documents in the request."""
        return len(self.documents)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "documents": list(self.documents),
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IngestionRequest":
        """Deserialize from dictionary."""
        return cls(
            documents=data["documents"],
            chunking_strategy=data.get("chunking_strategy", "fixed_size"),
            chunk_size=data.get("chunk_size", 512),
            chunk_overlap=data.get("chunk_overlap", 50),
        )


@dataclass(frozen=True)
class IngestionError:
    """
    An error encountered during document ingestion.
    
    Errors are recorded per-document to allow partial success in batch ingestion.
    
    Attributes:
        document_uri: URI of the document that failed
        error_code: Machine-readable error code
        message: Human-readable error description
        is_retryable: Whether the error is transient
    """
    
    document_uri: str
    error_code: str
    message: str
    is_retryable: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_uri": self.document_uri,
            "error_code": self.error_code,
            "message": self.message,
            "is_retryable": self.is_retryable,
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IngestionError":
        """Deserialize from dictionary."""
        return cls(
            document_uri=data["document_uri"],
            error_code=data["error_code"],
            message=data["message"],
            is_retryable=data.get("is_retryable", False),
        )


@dataclass(frozen=True)
class IngestionResult:
    """
    The result of a document ingestion operation.
    
    Results track what was processed, what was skipped (idempotent),
    and any errors encountered.
    
    Attributes:
        ingested_count: Number of new documents ingested
        skipped_count: Number of documents skipped (already exist)
        chunk_count: Total number of chunks created
        document_ids: IDs of successfully ingested documents
        errors: List of errors encountered during ingestion
    """
    
    ingested_count: int
    skipped_count: int
    chunk_count: int
    document_ids: tuple[str, ...]
    errors: tuple[IngestionError, ...] = field(default_factory=tuple)
    
    def __post_init__(self) -> None:
        """Validate result on creation."""
        if self.ingested_count < 0:
            raise ValueError("ingested_count must be non-negative")
        if self.skipped_count < 0:
            raise ValueError("skipped_count must be non-negative")
        if self.chunk_count < 0:
            raise ValueError("chunk_count must be non-negative")
    
    @property
    def total_processed(self) -> int:
        """Return total documents processed (ingested + skipped)."""
        return self.ingested_count + self.skipped_count
    
    @property
    def has_errors(self) -> bool:
        """Return whether any errors occurred."""
        return len(self.errors) > 0
    
    @property
    def error_count(self) -> int:
        """Return the number of errors."""
        return len(self.errors)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ingested_count": self.ingested_count,
            "skipped_count": self.skipped_count,
            "chunk_count": self.chunk_count,
            "document_ids": list(self.document_ids),
            "errors": [error.to_dict() for error in self.errors],
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IngestionResult":
        """Deserialize from dictionary."""
        return cls(
            ingested_count=data["ingested_count"],
            skipped_count=data.get("skipped_count", 0),
            chunk_count=data.get("chunk_count", 0),
            document_ids=tuple(data.get("document_ids", [])),
            errors=tuple(
                IngestionError.from_dict(error) for error in data.get("errors", [])
            ),
        )
    
    @classmethod
    def empty(cls) -> "IngestionResult":
        """Create an empty result (nothing processed)."""
        return cls(
            ingested_count=0,
            skipped_count=0,
            chunk_count=0,
            document_ids=(),
            errors=(),
        )


__all__ = [
    "IngestionRequest",
    "IngestionResult",
    "IngestionError",
]
