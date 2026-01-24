"""
Mock embedding provider adapter.

A deterministic mock implementation of EmbeddingProviderPort for testing.
Generates consistent embeddings based on text hash.
"""

import hashlib
import math
from collections.abc import Sequence


class MockEmbeddingProvider:
    """
    Mock implementation of EmbeddingProviderPort.
    
    This adapter generates deterministic embeddings based on text content.
    The same text will always produce the same embedding, making tests
    reproducible.
    
    The embeddings are NOT semantically meaningful - they are just
    consistent random-looking vectors derived from text hashes.
    
    Supported model IDs:
    - "mock-small" (128 dimensions)
    - "mock-medium" (512 dimensions)
    - "mock-large" (1536 dimensions)
    """
    
    # Model configuration: model_id -> (dimension, max_batch_size)
    _MODELS = {
        "mock-small": (128, 100),
        "mock-medium": (512, 50),
        "mock-large": (1536, 32),
        "text-embedding-ada-002": (1536, 32),  # Alias for OpenAI compatibility
    }
    
    def __init__(self, default_model: str = "mock-large") -> None:
        """
        Initialize the mock embedding provider.
        
        Args:
            default_model: Default model ID to use
        """
        if default_model not in self._MODELS:
            raise ValueError(f"Unknown model: {default_model}")
        self._default_model = default_model
        self._call_count = 0  # For testing/observability
    
    async def embed_texts(
        self,
        texts: Sequence[str],
        model_id: str,
    ) -> list[tuple[float, ...]]:
        """
        Generate deterministic mock embeddings for multiple texts.
        
        Args:
            texts: Sequence of texts to embed
            model_id: Identifier of the embedding model to use
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If the model is not supported
        """
        if model_id not in self._MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        dimension, _ = self._MODELS[model_id]
        self._call_count += 1
        
        embeddings = []
        for text in texts:
            embedding = self._generate_embedding(text, dimension)
            embeddings.append(embedding)
        
        return embeddings
    
    async def embed_query(
        self,
        query: str,
        model_id: str,
    ) -> tuple[float, ...]:
        """
        Generate a deterministic mock embedding for a single query.
        
        Args:
            query: The query text to embed
            model_id: Identifier of the embedding model to use
            
        Returns:
            The embedding vector
        """
        if model_id not in self._MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        dimension, _ = self._MODELS[model_id]
        self._call_count += 1
        
        return self._generate_embedding(query, dimension)
    
    def get_dimension(self, model_id: str) -> int:
        """
        Get the embedding dimension for a model.
        
        Args:
            model_id: Identifier of the embedding model
            
        Returns:
            Dimension of the embedding vectors
            
        Raises:
            ValueError: If the model is not supported
        """
        if model_id not in self._MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        dimension, _ = self._MODELS[model_id]
        return dimension
    
    def get_max_batch_size(self, model_id: str) -> int:
        """
        Get the maximum batch size for a model.
        
        Args:
            model_id: Identifier of the embedding model
            
        Returns:
            Maximum number of texts per batch call
        """
        if model_id not in self._MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        _, batch_size = self._MODELS[model_id]
        return batch_size
    
    def _generate_embedding(self, text: str, dimension: int) -> tuple[float, ...]:
        """
        Generate a deterministic embedding from text.
        
        Uses SHA-256 hash to generate a sequence of bytes, then converts
        to floats and normalizes to unit length.
        
        Args:
            text: The text to embed
            dimension: Target dimension
            
        Returns:
            Normalized embedding vector
        """
        # Generate enough hash bytes for the dimension
        hash_bytes = b""
        hash_input = text.encode("utf-8")
        
        while len(hash_bytes) < dimension * 4:  # 4 bytes per float
            hasher = hashlib.sha256()
            hasher.update(hash_input)
            hasher.update(len(hash_bytes).to_bytes(4, "big"))
            hash_bytes += hasher.digest()
        
        # Convert bytes to floats in range [-1, 1]
        raw_values: list[float] = []
        for i in range(dimension):
            # Use 4 bytes for each value
            byte_slice = hash_bytes[i * 4 : (i + 1) * 4]
            int_value = int.from_bytes(byte_slice, "big")
            # Normalize to [-1, 1]
            float_value = (int_value / (2**32 - 1)) * 2 - 1
            raw_values.append(float_value)
        
        # Normalize to unit length
        magnitude = math.sqrt(sum(v * v for v in raw_values))
        if magnitude == 0:
            # Avoid division by zero
            return tuple([0.0] * dimension)
        
        normalized = tuple(v / magnitude for v in raw_values)
        return normalized
    
    @property
    def call_count(self) -> int:
        """Return the number of embedding calls made."""
        return self._call_count
    
    def reset_call_count(self) -> None:
        """Reset the call count. For testing only."""
        self._call_count = 0


__all__ = [
    "MockEmbeddingProvider",
]
