"""
Chunking strategies for document processing.

This package provides configurable, deterministic chunking strategies
for splitting documents into retrievable chunks.

Key principles:
- Deterministic: Same input always produces same chunks
- Configurable: Chunk size, overlap, separators are customizable
- Auditable: Chunking decisions can be traced
"""

from genai_rag_service.chunking.base import ChunkingStrategy
from genai_rag_service.chunking.config import ChunkingConfig
from genai_rag_service.chunking.fixed_size import FixedSizeChunker
from genai_rag_service.chunking.recursive import RecursiveChunker

__all__ = [
    "ChunkingStrategy",
    "ChunkingConfig",
    "FixedSizeChunker",
    "RecursiveChunker",
]
