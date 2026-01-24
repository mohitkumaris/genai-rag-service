"""
In-memory vector store adapter for testing/dev.
"""

import math
from typing import Sequence, Any, Mapping
from collections import defaultdict
from rag.schemas import DocumentChunk, EmbeddingVector, VectorStorePort

class InMemoryVectorStore(VectorStorePort):
    """
    Brute-force vector search implementation.
    Not for production use with large datasets.
    """
    
    def __init__(self) -> None:
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        self._doc_to_chunks: dict[str, set[str]] = defaultdict(set)
        
    async def upsert_vectors(
        self,
        vectors: Sequence[EmbeddingVector],
        chunks: Sequence[DocumentChunk],
    ) -> int:
        count = 0
        for vec, chunk in zip(vectors, chunks):
            if vec.chunk_id != chunk.chunk_id:
                continue
                
            self._vectors[vec.chunk_id] = vec.vector
            self._chunks[chunk.chunk_id] = chunk
            self._doc_to_chunks[chunk.document_id].add(chunk.chunk_id)
            count += 1
        return count

    async def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        scores = []
        
        for chunk_id, vector in self._vectors.items():
            # Apply filters first
            if filters and not self._matches_filters(self._chunks[chunk_id], filters):
                continue
                
            score = self._cosine_similarity(query_vector, vector)
            if score >= similarity_threshold:
                scores.append((chunk_id, score))
                
        # Sort desc
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def hybrid_search(
        self,
        query_vector: tuple[float, ...],
        query_text: str,
        top_k: int,
        vector_weight: float = 0.7,
        filters: Mapping[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        # Naive hybrid: cosine similarity + keyword match in content
        scores = []
        keyword_weight = 1.0 - vector_weight
        query_terms = set(query_text.lower().split())
        
        for chunk_id, vector in self._vectors.items():
            chunk = self._chunks[chunk_id]
            
            if filters and not self._matches_filters(chunk, filters):
                continue

            vec_score = self._cosine_similarity(query_vector, vector)
            
            # Simple keyword score: % of query terms present
            content_lower = chunk.content.lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            kw_score = matches / len(query_terms) if query_terms else 0.0
            
            final_score = (vec_score * vector_weight) + (kw_score * keyword_weight)
            
            if final_score >= similarity_threshold:
                scores.append((chunk_id, final_score))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    async def document_exists(self, document_id: str) -> bool:
        return document_id in self._doc_to_chunks and len(self._doc_to_chunks[document_id]) > 0

    async def delete_by_document(self, document_id: str) -> int:
        if document_id not in self._doc_to_chunks:
            return 0
            
        chunk_ids = self._doc_to_chunks[document_id]
        count = len(chunk_ids)
        
        for cid in chunk_ids:
            if cid in self._vectors:
                del self._vectors[cid]
            if cid in self._chunks:
                del self._chunks[cid]
        
        del self._doc_to_chunks[document_id]
        return count

    async def get_chunk_count(self) -> int:
        return len(self._chunks)

    def _cosine_similarity(self, v1: tuple[float, ...], v2: tuple[float, ...]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _matches_filters(self, chunk: DocumentChunk, filters: Mapping[str, Any]) -> bool:
        meta = chunk.metadata.custom_metadata
        for key, value in filters.items():
            if key not in meta or meta[key] != value:
                return False
        return True
