"""
Embedding domain models.

These models represent embedding vectors and configuration for the RAG system.
Embeddings are treated as immutable artifacts once generated.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmbeddingVector:
    """
    An embedding vector with its source reference.
    
    Embedding vectors are immutable once generated. They are stored in the
    vector store and used for similarity search.
    
    Attributes:
        chunk_id: Reference to the source chunk
        vector: The embedding vector as an immutable tuple
        model_id: Identifier of the model that generated this embedding
        dimension: Dimension of the vector (for validation)
    """
    
    chunk_id: str
    vector: tuple[float, ...]
    model_id: str
    dimension: int
    
    def __post_init__(self) -> None:
        """Validate embedding on creation."""
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not self.vector:
            raise ValueError("vector cannot be empty")
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, "
                f"got {len(self.vector)}"
            )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage/logging."""
        return {
            "chunk_id": self.chunk_id,
            "vector": list(self.vector),
            "model_id": self.model_id,
            "dimension": self.dimension,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingVector":
        """Deserialize from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            vector=tuple(data["vector"]),
            model_id=data["model_id"],
            dimension=data["dimension"],
        )
    
    @classmethod
    def create(
        cls,
        chunk_id: str,
        vector: list[float] | tuple[float, ...],
        model_id: str,
    ) -> "EmbeddingVector":
        """
        Factory method to create an embedding vector.
        
        Automatically calculates the dimension from the vector length.
        """
        vector_tuple = tuple(vector)
        return cls(
            chunk_id=chunk_id,
            vector=vector_tuple,
            model_id=model_id,
            dimension=len(vector_tuple),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for embedding generation.
    
    This configuration is used to specify how embeddings should be generated
    and controls batching behavior for efficiency.
    
    Attributes:
        model_id: Identifier of the embedding model to use
        dimension: Expected dimension of output vectors
        batch_size: Number of texts to embed in a single API call
    """
    
    model_id: str
    dimension: int
    batch_size: int = 32
    
    def __post_init__(self) -> None:
        """Validate configuration on creation."""
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingConfig":
        """Deserialize from dictionary."""
        return cls(
            model_id=data["model_id"],
            dimension=data["dimension"],
            batch_size=data.get("batch_size", 32),
        )


__all__ = [
    "EmbeddingVector",
    "EmbeddingConfig",
]
