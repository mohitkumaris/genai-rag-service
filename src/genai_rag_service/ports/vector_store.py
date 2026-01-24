"""
Vector store port interface.

Defines the abstract interface for vector storage and search operations.
Implementations can use Azure AI Search, Pinecone, Qdrant, etc.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from genai_rag_service.domain.document import DocumentChunk
from genai_rag_service.domain.embedding import EmbeddingVector


class VectorStorePort(Protocol):
    """
    Port for vector storage and search operations.
    
    This interface abstracts vector database operations including:
    - Upserting vectors with associated metadata
    - Similarity search (vector-only and hybrid)
    - Chunk management (retrieve, delete)
    
    All methods are async for non-blocking I/O.
    """
    
    async def upsert_vectors(
        self,
        vectors: Sequence[EmbeddingVector],
        chunks: Sequence[DocumentChunk],
    ) -> int:
        """
        Upsert vectors with associated chunk metadata.
        
        This operation is idempotent - upserting the same vectors again
        will update existing entries rather than creating duplicates.
        
        Args:
            vectors: Embedding vectors to store
            chunks: Corresponding document chunks (must match 1:1)
            
        Returns:
            Number of vectors upserted
            
        Raises:
            VectorStoreError: If there's a storage error
            ValueError: If vectors and chunks don't match
        """
        ...
    
    async def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by score descending
        """
        ...
    
    async def hybrid_search(
        self,
        query_vector: tuple[float, ...],
        query_text: str,
        top_k: int,
        vector_weight: float = 0.7,
        filters: Mapping[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Hybrid vector + keyword search.
        
        Combines semantic similarity with keyword matching for improved
        retrieval quality on queries containing specific terms.
        
        Args:
            query_vector: The query embedding vector
            query_text: The original query text for keyword matching
            top_k: Maximum number of results to return
            vector_weight: Weight for vector scores (1 - vector_weight = keyword weight)
            filters: Optional metadata filters
            similarity_threshold: Minimum combined score
            
        Returns:
            List of (chunk_id, combined_score) tuples, sorted by score descending
        """
        ...
    
    async def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: The chunk identifier
            
        Returns:
            The document chunk if found, None otherwise
        """
        ...
    
    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]:
        """
        Retrieve multiple chunks by ID.
        
        Args:
            chunk_ids: Sequence of chunk identifiers
            
        Returns:
            List of found chunks (may be fewer than requested if some don't exist)
        """
        ...
    
    async def delete_by_document(self, document_id: str) -> int:
        """
        Delete all vectors for a document.
        
        Args:
            document_id: The document identifier
            
        Returns:
            Number of vectors deleted
        """
        ...
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a single chunk/vector.
        
        Args:
            chunk_id: The chunk identifier
            
        Returns:
            True if deleted, False if it didn't exist
        """
        ...
    
    async def document_exists(self, document_id: str) -> bool:
        """
        Check if a document has vectors in the store.
        
        Args:
            document_id: The document identifier
            
        Returns:
            True if at least one chunk exists for this document
        """
        ...
    
    async def get_chunk_count(self) -> int:
        """
        Get the total number of chunks in the store.
        
        Returns:
            Total chunk count
        """
        ...


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    
    def __init__(self, message: str, operation: str | None = None) -> None:
        super().__init__(message)
        self.operation = operation


__all__ = [
    "VectorStorePort",
    "VectorStoreError",
]
