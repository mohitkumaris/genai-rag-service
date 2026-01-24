"""
In-memory vector store adapter.

A simple in-memory implementation of the VectorStorePort for testing
and development. Uses brute-force cosine similarity for search.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from genai_rag_service.domain.document import DocumentChunk
from genai_rag_service.domain.embedding import EmbeddingVector


def _cosine_similarity(vec1: tuple[float, ...], vec2: tuple[float, ...]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (0.0 to 1.0)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def _keyword_score(query_text: str, chunk_content: str) -> float:
    """
    Calculate a simple keyword match score.
    
    Args:
        query_text: The query string
        chunk_content: The chunk content to match against
        
    Returns:
        Keyword match score (0.0 to 1.0)
    """
    query_words = set(query_text.lower().split())
    content_words = set(chunk_content.lower().split())
    
    if not query_words:
        return 0.0
    
    matches = len(query_words & content_words)
    return matches / len(query_words)


class InMemoryVectorStore:
    """
    In-memory implementation of VectorStorePort.
    
    This adapter stores vectors and chunks in dictionaries and performs
    brute-force similarity search. Suitable for testing and small datasets.
    
    Features:
    - Cosine similarity search
    - Simple keyword search for hybrid queries
    - Metadata filtering
    
    Not suitable for production use with large datasets.
    """
    
    def __init__(self) -> None:
        """Initialize the in-memory vector store."""
        self._vectors: dict[str, EmbeddingVector] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        self._document_index: dict[str, set[str]] = {}  # document_id -> chunk_ids
    
    async def upsert_vectors(
        self,
        vectors: Sequence[EmbeddingVector],
        chunks: Sequence[DocumentChunk],
    ) -> int:
        """
        Upsert vectors with associated chunk metadata.
        
        Args:
            vectors: Embedding vectors to store
            chunks: Corresponding document chunks (must match 1:1)
            
        Returns:
            Number of vectors upserted
            
        Raises:
            ValueError: If vectors and chunks don't match
        """
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Vector count ({len(vectors)}) must match chunk count ({len(chunks)})"
            )
        
        count = 0
        for vector, chunk in zip(vectors, chunks):
            if vector.chunk_id != chunk.chunk_id:
                raise ValueError(
                    f"Vector chunk_id ({vector.chunk_id}) must match "
                    f"chunk chunk_id ({chunk.chunk_id})"
                )
            
            self._vectors[vector.chunk_id] = vector
            self._chunks[chunk.chunk_id] = chunk
            
            # Update document index
            doc_id = chunk.document_id
            if doc_id not in self._document_index:
                self._document_index[doc_id] = set()
            self._document_index[doc_id].add(chunk.chunk_id)
            
            count += 1
        
        return count
    
    async def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        results: list[tuple[str, float]] = []
        
        for chunk_id, vector in self._vectors.items():
            # Apply filters
            if filters and not self._matches_filters(chunk_id, filters):
                continue
            
            # Calculate similarity
            score = _cosine_similarity(query_vector, vector.vector)
            
            if score >= similarity_threshold:
                results.append((chunk_id, score))
        
        # Sort by score descending and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
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
        
        Combines cosine similarity with keyword matching using weighted fusion.
        
        Args:
            query_vector: The query embedding vector
            query_text: The original query text for keyword matching
            top_k: Maximum number of results to return
            vector_weight: Weight for vector scores (keyword_weight = 1 - vector_weight)
            filters: Optional metadata filters
            similarity_threshold: Minimum combined score
            
        Returns:
            List of (chunk_id, combined_score) tuples
        """
        keyword_weight = 1 - vector_weight
        results: list[tuple[str, float]] = []
        
        for chunk_id, vector in self._vectors.items():
            # Apply filters
            if filters and not self._matches_filters(chunk_id, filters):
                continue
            
            # Calculate vector similarity
            vector_score = _cosine_similarity(query_vector, vector.vector)
            
            # Calculate keyword score
            chunk = self._chunks.get(chunk_id)
            keyword_score = 0.0
            if chunk:
                keyword_score = _keyword_score(query_text, chunk.content)
            
            # Combine scores
            combined_score = (vector_weight * vector_score) + (keyword_weight * keyword_score)
            
            if combined_score >= similarity_threshold:
                results.append((chunk_id, combined_score))
        
        # Sort by score descending and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    async def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: The chunk identifier
            
        Returns:
            The document chunk if found, None otherwise
        """
        return self._chunks.get(chunk_id)
    
    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]:
        """
        Retrieve multiple chunks by ID.
        
        Args:
            chunk_ids: Sequence of chunk identifiers
            
        Returns:
            List of found chunks
        """
        return [
            self._chunks[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self._chunks
        ]
    
    async def delete_by_document(self, document_id: str) -> int:
        """
        Delete all vectors for a document.
        
        Args:
            document_id: The document identifier
            
        Returns:
            Number of vectors deleted
        """
        chunk_ids = self._document_index.get(document_id, set())
        count = 0
        
        for chunk_id in list(chunk_ids):
            if chunk_id in self._vectors:
                del self._vectors[chunk_id]
                count += 1
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
        
        if document_id in self._document_index:
            del self._document_index[document_id]
        
        return count
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a single chunk/vector.
        
        Args:
            chunk_id: The chunk identifier
            
        Returns:
            True if deleted, False if it didn't exist
        """
        if chunk_id not in self._vectors:
            return False
        
        # Get document ID to update index
        chunk = self._chunks.get(chunk_id)
        if chunk:
            doc_id = chunk.document_id
            if doc_id in self._document_index:
                self._document_index[doc_id].discard(chunk_id)
                if not self._document_index[doc_id]:
                    del self._document_index[doc_id]
        
        del self._vectors[chunk_id]
        if chunk_id in self._chunks:
            del self._chunks[chunk_id]
        
        return True
    
    async def document_exists(self, document_id: str) -> bool:
        """
        Check if a document has vectors in the store.
        
        Args:
            document_id: The document identifier
            
        Returns:
            True if at least one chunk exists for this document
        """
        return (
            document_id in self._document_index
            and len(self._document_index[document_id]) > 0
        )
    
    async def get_chunk_count(self) -> int:
        """
        Get the total number of chunks in the store.
        
        Returns:
            Total chunk count
        """
        return len(self._chunks)
    
    def _matches_filters(
        self,
        chunk_id: str,
        filters: Mapping[str, Any],
    ) -> bool:
        """
        Check if a chunk matches the given filters.
        
        Filters are matched against custom_metadata in the chunk's metadata.
        
        Args:
            chunk_id: The chunk to check
            filters: Key-value pairs to match
            
        Returns:
            True if all filters match
        """
        chunk = self._chunks.get(chunk_id)
        if not chunk:
            return False
        
        custom_metadata = chunk.metadata.custom_metadata
        for key, value in filters.items():
            if key not in custom_metadata:
                return False
            if custom_metadata[key] != value:
                return False
        
        return True
    
    def clear(self) -> None:
        """Clear all stored data. For testing only."""
        self._vectors.clear()
        self._chunks.clear()
        self._document_index.clear()


__all__ = [
    "InMemoryVectorStore",
]
