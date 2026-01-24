"""
FastAPI application layer.

This package provides the HTTP interface for the RAG service.
"""

from genai_rag_service.api.app import create_app

__all__ = [
    "create_app",
]
