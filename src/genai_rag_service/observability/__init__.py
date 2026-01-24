"""
Observability utilities for the RAG service.
"""

from genai_rag_service.observability.logging import configure_logging, get_logger

__all__ = [
    "configure_logging",
    "get_logger",
]
