"""
Embedding provider port interface.

Defines the abstract interface for embedding generation.
Implementations can use OpenAI, Azure OpenAI, Cohere, Vertex AI, etc.
"""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProviderPort(Protocol):
    """
    Port for embedding generation.
    
    This interface abstracts embedding model providers. Implementations
    should handle authentication, rate limiting, and retries.
    
    All methods are async for non-blocking I/O.
    """
    
    async def embed_texts(
        self,
        texts: Sequence[str],
        model_id: str,
    ) -> list[tuple[float, ...]]:
        """
        Generate embeddings for multiple texts.
        
        Batch embedding is more efficient than individual calls.
        The implementation should handle batching internally if needed.
        
        Args:
            texts: Sequence of texts to embed
            model_id: Identifier of the embedding model to use
            
        Returns:
            List of embedding vectors (same order as input texts)
            
        Raises:
            EmbeddingError: If embedding generation fails
            RateLimitError: If rate limit is exceeded
        """
        ...
    
    async def embed_query(
        self,
        query: str,
        model_id: str,
    ) -> tuple[float, ...]:
        """
        Generate embedding for a single query.
        
        Some providers have different endpoints or handling for queries
        vs documents. Use this method for search queries.
        
        Args:
            query: The query text to embed
            model_id: Identifier of the embedding model to use
            
        Returns:
            The embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...
    
    def get_dimension(self, model_id: str) -> int:
        """
        Get the embedding dimension for a model.
        
        This is a synchronous method since dimensions are typically
        known upfront and don't require API calls.
        
        Args:
            model_id: Identifier of the embedding model
            
        Returns:
            Dimension of the embedding vectors
            
        Raises:
            ValueError: If the model is not supported
        """
        ...
    
    def get_max_batch_size(self, model_id: str) -> int:
        """
        Get the maximum batch size for a model.
        
        Args:
            model_id: Identifier of the embedding model
            
        Returns:
            Maximum number of texts per batch call
        """
        ...


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    
    def __init__(
        self,
        message: str,
        model_id: str | None = None,
        is_retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.is_retryable = is_retryable


class RateLimitError(EmbeddingError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        model_id: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message, model_id, is_retryable=True)
        self.retry_after_seconds = retry_after_seconds


__all__ = [
    "EmbeddingProviderPort",
    "EmbeddingError",
    "RateLimitError",
]
