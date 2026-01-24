"""
MCP Tool Registry setup.

Creates and configures the ToolRegistry with all RAG tools.
"""

from genai_mcp_core import ToolRegistry

from genai_rag_service.services.ingestion import IngestionService
from genai_rag_service.services.retrieval import RetrievalService
from genai_rag_service.tools.rag_ingest import rag_ingest_handler, rag_ingest_tool
from genai_rag_service.tools.rag_search import rag_search_handler, rag_search_tool


def create_tool_registry(
    ingestion_service: IngestionService,
    retrieval_service: RetrievalService,
) -> ToolRegistry:
    """
    Create and configure the tool registry with all RAG tools.
    
    Args:
        ingestion_service: Ingestion service instance
        retrieval_service: Retrieval service instance
        
    Returns:
        Configured ToolRegistry with all tools registered
    """
    registry = ToolRegistry()
    
    # Register ingestion tool
    registry.register(
        rag_ingest_tool,
        rag_ingest_handler(ingestion_service),
    )
    
    # Register search tool
    registry.register(
        rag_search_tool,
        rag_search_handler(retrieval_service),
    )
    
    return registry


__all__ = [
    "create_tool_registry",
]
