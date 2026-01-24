"""
Retrieval service.

Orchestrates the document retrieval pipeline:
1. Accept search query
2. Generate query embedding
3. Search vector store
4. Assemble and rank results
5. Return grounding context
"""

import time
from typing import Literal

import structlog

from genai_rag_service.domain.embedding import EmbeddingConfig
from genai_rag_service.domain.retrieval import RetrievalContext, SearchQuery, SearchResult
from genai_rag_service.ports.embedding import EmbeddingProviderPort
from genai_rag_service.ports.vector_store import VectorStorePort


logger = structlog.get_logger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant document chunks.
    
    This service handles the complete retrieval pipeline:
    1. Accept and validate search queries
    2. Generate query embedding
    3. Execute vector or hybrid search
    4. Retrieve chunk content
    5. Assemble ranked results
    
    The service returns grounding context, NOT answers.
    It never generates natural language responses.
    
    Attributes:
        vector_store: Vector store adapter for search
        embedding_provider: Embedding provider for query vectors
        embedding_config: Configuration for embedding generation
    """
    
    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_provider: EmbeddingProviderPort,
        embedding_config: EmbeddingConfig,
    ) -> None:
        """
        Initialize the retrieval service.
        
        Args:
            vector_store: Vector store adapter
            embedding_provider: Embedding provider adapter
            embedding_config: Embedding configuration
        """
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._embedding_config = embedding_config
    
    async def search(self, query: SearchQuery) -> RetrievalContext:
        """
        Search for relevant document chunks.
        
        Args:
            query: The search query
            
        Returns:
            RetrievalContext with ranked results and metadata
        """
        start_time = time.monotonic()
        
        logger.info(
            "search_started",
            query_length=len(query.query_text),
            top_k=query.top_k,
            search_type=query.search_type,
        )
        
        # Generate query embedding
        model_id = self._embedding_config.model_id
        query_vector = await self._embedding_provider.embed_query(
            query.query_text,
            model_id,
        )
        
        # Execute search based on type
        if query.search_type == "hybrid":
            search_results = await self._vector_store.hybrid_search(
                query_vector=query_vector,
                query_text=query.query_text,
                top_k=query.top_k,
                vector_weight=1.0 - query.keyword_weight,
                filters=dict(query.filters) if query.filters else None,
                similarity_threshold=query.similarity_threshold,
            )
            match_type: Literal["vector", "keyword", "hybrid"] = "hybrid"
        else:
            search_results = await self._vector_store.search(
                query_vector=query_vector,
                top_k=query.top_k,
                filters=dict(query.filters) if query.filters else None,
                similarity_threshold=query.similarity_threshold,
            )
            match_type = "vector"
        
        # Retrieve chunk content
        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        chunks = await self._vector_store.get_chunks(chunk_ids)
        
        # Build chunk lookup for ordering
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Assemble results in score order
        results: list[SearchResult] = []
        for chunk_id, score in search_results:
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                # Chunk was deleted between search and retrieval
                logger.warning("chunk_not_found", chunk_id=chunk_id)
                continue
            
            result = SearchResult(
                chunk=chunk,
                score=score,
                match_type=match_type,
            )
            results.append(result)
        
        # Calculate latency
        latency_ms = (time.monotonic() - start_time) * 1000
        
        # Get total chunk count for stats
        total_chunks = await self._vector_store.get_chunk_count()
        
        context = RetrievalContext(
            results=tuple(results),
            query=query,
            latency_ms=latency_ms,
            total_chunks_searched=total_chunks,
        )
        
        logger.info(
            "search_completed",
            result_count=len(results),
            latency_ms=latency_ms,
            total_chunks_searched=total_chunks,
        )
        
        return context
    
    async def get_chunk(self, chunk_id: str) -> SearchResult | None:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: The chunk identifier
            
        Returns:
            SearchResult with the chunk, or None if not found
        """
        chunk = await self._vector_store.get_chunk(chunk_id)
        if chunk is None:
            return None
        
        return SearchResult(
            chunk=chunk,
            score=1.0,  # Direct retrieval has perfect score
            match_type="vector",
        )
    
    async def get_document_chunks(self, document_id: str) -> list[SearchResult]:
        """
        Retrieve all chunks for a document.
        
        Args:
            document_id: The document identifier
            
        Returns:
            List of SearchResults for all chunks in the document
        """
        # Note: This is a simplified implementation
        # A production implementation would have a more efficient
        # way to retrieve all chunks for a document
        if not await self._vector_store.document_exists(document_id):
            return []
        
        # For the in-memory adapter, we can iterate through all chunks
        # A production adapter would have an index for this
        all_chunks = await self._vector_store.get_chunks([])  # This won't work efficiently
        
        # Filter to document
        results = []
        for chunk in all_chunks:
            if chunk.document_id == document_id:
                result = SearchResult(
                    chunk=chunk,
                    score=1.0,
                    match_type="vector",
                )
                results.append(result)
        
        # Sort by chunk index
        results.sort(key=lambda r: r.chunk.chunk_index)
        return results


__all__ = [
    "RetrievalService",
]
