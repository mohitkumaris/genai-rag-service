"""
Search Service.

Handles vector and hybrid search orchestration.
"""

import time
import structlog
from rag.schemas import (
    SearchQuery, RetrievalContext, SearchResult, 
    VectorStorePort, EmbeddingProviderPort
)

logger = structlog.get_logger(__name__)

class SearchService:
    """
    Core retrieval logic.
    Dependent on Ports (interfaces), NOT concrete adapters.
    """

    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_provider: EmbeddingProviderPort,
        embedding_model_id: str = "text-embedding-ada-002"
    ):
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._embedding_model_id = embedding_model_id

    async def search(self, query: SearchQuery) -> RetrievalContext:
        start_t = time.monotonic()
        
        # 1. Embed Query
        query_vector = await self._embedding_provider.embed_query(
            query.query_text, 
            self._embedding_model_id
        )

        # 2. Execute Search (Vector or Hybrid)
        if query.search_type == "hybrid":
            raw_results = await self._vector_store.hybrid_search(
                query_vector=query_vector,
                query_text=query.query_text,
                top_k=query.top_k,
                vector_weight=(1.0 - query.keyword_weight),
                filters=query.filters,
                similarity_threshold=query.similarity_threshold
            )
            match_type = "hybrid"
        else:
            raw_results = await self._vector_store.search(
                query_vector=query_vector,
                top_k=query.top_k,
                filters=query.filters,
                similarity_threshold=query.similarity_threshold
            )
            match_type = "vector"

        # 3. Hydrate Results (Get full chunks)
        # Assuming vector store might return only IDs/scores, but port says get_chunks is separate?
        # The port definitions in schema.py for search/hybrid_search return list[tuple[str, float]] (id, score).
        # We need to fetch the chunks.
        
        chunk_ids = [r[0] for r in raw_results]
        chunks = await self._vector_store.get_chunks(chunk_ids)
        chunk_map = {c.chunk_id: c for c in chunks}
        
        results = []
        for cid, score in raw_results:
            if cid in chunk_map:
                results.append(SearchResult(
                    chunk=chunk_map[cid],
                    score=score,
                    match_type=match_type
                ))

        latency = (time.monotonic() - start_t) * 1000
        
        # Approximate stat
        total_chunks = await self._vector_store.get_chunk_count()

        return RetrievalContext(
            results=tuple(results),
            query=query,
            latency_ms=latency,
            total_chunks_searched=total_chunks
        )
