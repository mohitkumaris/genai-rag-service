"""
Chunking configuration.

Defines configuration options for chunking strategies.
All configuration is immutable for safety and predictability.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Configuration for document chunking behavior.
    
    This configuration controls how documents are split into chunks.
    All values are designed to work with token-based chunking.
    
    Attributes:
        strategy: The chunking strategy to use
            - "fixed_size": Split by token count with overlap
            - "recursive": Split by separators, then by size
        chunk_size: Target chunk size in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        min_chunk_size: Minimum chunk size (smaller chunks are merged)
        separators: Separator strings for recursive chunking (in priority order)
        model_name: Tokenizer model for accurate token counting
    """
    
    strategy: Literal["fixed_size", "recursive"] = "fixed_size"
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    separators: tuple[str, ...] = field(
        default_factory=lambda: ("\n\n", "\n", ". ", " ")
    )
    model_name: str = "cl100k_base"  # GPT-4 / text-embedding-ada-002 tokenizer
    
    def __post_init__(self) -> None:
        """Validate configuration on creation."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must be <= chunk_size")
        if not self.separators:
            raise ValueError("separators cannot be empty")
    
    @property
    def effective_chunk_size(self) -> int:
        """
        Return the effective chunk size after accounting for overlap.
        
        This is the number of new tokens in each chunk (excluding overlap).
        """
        return self.chunk_size - self.chunk_overlap
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "separators": list(self.separators),
            "model_name": self.model_name,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkingConfig":
        """Deserialize from dictionary."""
        separators = data.get("separators")
        if separators is not None:
            separators = tuple(separators)
        else:
            separators = ("\n\n", "\n", ". ", " ")
        
        return cls(
            strategy=data.get("strategy", "fixed_size"),
            chunk_size=data.get("chunk_size", 512),
            chunk_overlap=data.get("chunk_overlap", 50),
            min_chunk_size=data.get("min_chunk_size", 100),
            separators=separators,
            model_name=data.get("model_name", "cl100k_base"),
        )
    
    @classmethod
    def default(cls) -> "ChunkingConfig":
        """Create a default configuration."""
        return cls()
    
    @classmethod
    def for_large_documents(cls) -> "ChunkingConfig":
        """Create a configuration optimized for large documents."""
        return cls(
            strategy="recursive",
            chunk_size=1024,
            chunk_overlap=100,
            min_chunk_size=200,
        )
    
    @classmethod
    def for_small_documents(cls) -> "ChunkingConfig":
        """Create a configuration optimized for small documents."""
        return cls(
            strategy="fixed_size",
            chunk_size=256,
            chunk_overlap=25,
            min_chunk_size=50,
        )


__all__ = [
    "ChunkingConfig",
]
