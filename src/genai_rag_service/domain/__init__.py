"""
Domain models for the RAG service.

This package contains pure Python dataclasses representing the core concepts:
- Documents and chunks
- Embeddings
- Retrieval queries and results
- Ingestion requests and results

All models are immutable (frozen dataclasses) for safety and predictability.
"""

from genai_rag_service.domain.document import (
    Document,
    DocumentChunk,
    DocumentMetadata,
)
from genai_rag_service.domain.embedding import (
    EmbeddingConfig,
    EmbeddingVector,
)
from genai_rag_service.domain.ingestion import (
    IngestionError,
    IngestionRequest,
    IngestionResult,
)
from genai_rag_service.domain.retrieval import (
    RetrievalContext,
    SearchQuery,
    SearchResult,
)

__all__ = [
    # Document
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    # Embedding
    "EmbeddingVector",
    "EmbeddingConfig",
    # Retrieval
    "SearchQuery",
    "SearchResult",
    "RetrievalContext",
    # Ingestion
    "IngestionRequest",
    "IngestionResult",
    "IngestionError",
]
