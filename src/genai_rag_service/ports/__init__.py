"""
Port interfaces for the RAG service.

Ports define abstract interfaces for external dependencies (adapters).
This enables testing with mocks and swapping implementations without
changing business logic.

Following the hexagonal architecture pattern:
- Ports are interfaces (Protocols)
- Adapters are concrete implementations
- The domain is isolated from infrastructure concerns
"""

from genai_rag_service.ports.embedding import EmbeddingProviderPort
from genai_rag_service.ports.storage import BlobStoragePort
from genai_rag_service.ports.vector_store import VectorStorePort

__all__ = [
    "BlobStoragePort",
    "VectorStorePort",
    "EmbeddingProviderPort",
]
