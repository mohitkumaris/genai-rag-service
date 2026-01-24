"""
Mock embedding provider for testing.
"""

import hashlib
from typing import Sequence
from rag.schemas import EmbeddingProviderPort

class MockEmbeddingProvider(EmbeddingProviderPort):
    """generates deterministic embeddings based on content hash."""

    def __init__(self, dimension: int = 1536) -> None:
        self._dimension = dimension

    async def embed_texts(
        self,
        texts: Sequence[str],
        model_id: str,
    ) -> list[tuple[float, ...]]:
        return [self._generate_embedding(text) for text in texts]

    async def embed_query(
        self,
        query: str,
        model_id: str,
    ) -> tuple[float, ...]:
        return self._generate_embedding(query)

    def _generate_embedding(self, text: str) -> tuple[float, ...]:
        # Seed generator with hash of text
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        
        # Simple pseudo-random vector based on seed
        # This ensures same text = same vector (deterministic)
        vector = []
        curr = seed
        for _ in range(self._dimension):
            curr = (curr * 1103515245 + 12345) & 0x7FFFFFFF
            val = (curr / 0x7FFFFFFF) # 0.0 to 1.0
            vector.append(val)
            
        # Normalize
        norm = sum(x*x for x in vector) ** 0.5
        return tuple(x/norm for x in vector)
