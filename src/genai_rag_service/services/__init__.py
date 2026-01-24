"""
Business services for the RAG service.

This package contains the core business logic for:
- Document ingestion (chunking, embedding, indexing)
- Retrieval (search, ranking, context assembly)

Services orchestrate the domain models and adapters.
"""

from genai_rag_service.services.ingestion import IngestionService
from genai_rag_service.services.retrieval import RetrievalService

__all__ = [
    "IngestionService",
    "RetrievalService",
]
