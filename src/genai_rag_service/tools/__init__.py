"""
MCP tool implementations for the RAG service.

This package provides MCP-compatible tools for:
- rag_ingest: Document ingestion
- rag_search: Document retrieval

Tools follow the genai-mcp-core contracts and are designed
for consumption by LLM agents via the orchestrator service.
"""

from genai_rag_service.tools.rag_ingest import rag_ingest_handler, rag_ingest_tool
from genai_rag_service.tools.rag_search import rag_search_handler, rag_search_tool
from genai_rag_service.tools.registry import create_tool_registry

__all__ = [
    # Ingestion
    "rag_ingest_tool",
    "rag_ingest_handler",
    # Search
    "rag_search_tool",
    "rag_search_handler",
    # Registry
    "create_tool_registry",
]
