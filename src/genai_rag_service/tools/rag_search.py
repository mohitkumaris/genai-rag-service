"""
RAG Search Tool.

MCP tool for searching the RAG system for relevant document chunks.
"""

from collections.abc import Mapping
from typing import Any

from genai_mcp_core import (
    MCPContext,
    ToolDefinition,
    ToolFailure,
    ToolResult,
    ToolSuccess,
)

from genai_rag_service.domain.retrieval import SearchQuery
from genai_rag_service.services.retrieval import RetrievalService


# Tool definition
rag_search_tool = ToolDefinition(
    name="rag_search",
    description=(
        "Search the RAG system for relevant document chunks. "
        "Returns ranked chunks with similarity scores for grounding LLM responses. "
        "Supports both vector similarity search and hybrid (vector + keyword) search."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "The search query in natural language",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
                "description": "Maximum number of results to return",
            },
            "search_type": {
                "type": "string",
                "enum": ["vector", "hybrid"],
                "default": "vector",
                "description": "Type of search to perform",
            },
            "similarity_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.0,
                "description": "Minimum similarity score for results",
            },
            "filters": {
                "type": "object",
                "description": "Metadata filters to apply (key-value matching)",
                "additionalProperties": True,
            },
            "keyword_weight": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.3,
                "description": "Weight for keyword matching in hybrid search",
            },
        },
        "required": ["query"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "description": "Ranked search results",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "Unique identifier for the chunk",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Parent document identifier",
                        },
                        "content": {
                            "type": "string",
                            "description": "The chunk text content",
                        },
                        "score": {
                            "type": "number",
                            "description": "Relevance score (0.0 to 1.0)",
                        },
                        "match_type": {
                            "type": "string",
                            "enum": ["vector", "keyword", "hybrid"],
                            "description": "How the match was found",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Chunk metadata",
                        },
                    },
                    "required": ["chunk_id", "content", "score"],
                },
            },
            "total_results": {
                "type": "integer",
                "description": "Number of results returned",
            },
            "latency_ms": {
                "type": "number",
                "description": "Search latency in milliseconds",
            },
            "total_chunks_searched": {
                "type": "integer",
                "description": "Total chunks in the index",
            },
        },
        "required": ["results", "total_results"],
    },
    version="1.0.0",
    category="rag",
    tags=frozenset(["retrieval", "search", "vector", "semantic"]),
    required_permissions=frozenset(["rag:search"]),
)


class RagSearchHandler:
    """
    Handler for the rag_search tool.
    
    This handler wraps the RetrievalService and provides the MCP
    tool interface for document search.
    """
    
    def __init__(self, retrieval_service: RetrievalService) -> None:
        """
        Initialize the handler.
        
        Args:
            retrieval_service: The retrieval service to use
        """
        self._retrieval_service = retrieval_service
    
    async def __call__(
        self,
        context: MCPContext,
        input_data: Mapping[str, Any],
    ) -> ToolResult:
        """
        Execute the rag_search tool.
        
        Args:
            context: MCP context with permissions and metadata
            input_data: Tool input matching the input_schema
            
        Returns:
            ToolSuccess with search results, or ToolFailure on error
        """
        try:
            # Parse query from input
            query_text = input_data.get("query", "")
            if not query_text:
                return ToolFailure(
                    error_code="INVALID_INPUT",
                    message="Query text is required",
                    is_retryable=False,
                )
            
            # Build search query
            query = SearchQuery(
                query_text=query_text,
                top_k=input_data.get("top_k", 10),
                similarity_threshold=input_data.get("similarity_threshold", 0.0),
                filters=input_data.get("filters", {}),
                search_type=input_data.get("search_type", "vector"),
                keyword_weight=input_data.get("keyword_weight", 0.3),
            )
            
            # Execute search
            context_result = await self._retrieval_service.search(query)
            
            # Build response
            results = []
            for result in context_result.results:
                results.append({
                    "chunk_id": result.chunk.chunk_id,
                    "document_id": result.chunk.document_id,
                    "content": result.chunk.content,
                    "score": result.score,
                    "match_type": result.match_type,
                    "metadata": {
                        "source_uri": result.chunk.metadata.source_uri,
                        "version": result.chunk.metadata.version,
                        "chunk_index": result.chunk.chunk_index,
                        "token_count": result.chunk.token_count,
                        **dict(result.chunk.metadata.custom_metadata),
                    },
                })
            
            return ToolSuccess(
                data={
                    "results": results,
                    "total_results": len(results),
                    "latency_ms": context_result.latency_ms,
                    "total_chunks_searched": context_result.total_chunks_searched,
                },
                metadata={
                    "request_id": context.request_id,
                    "trace_id": context.trace_id,
                    "search_type": query.search_type,
                },
            )
            
        except Exception as e:
            return ToolFailure(
                error_code="SEARCH_ERROR",
                message=str(e),
                details={"exception_type": type(e).__name__},
                is_retryable=True,
            )


def rag_search_handler(retrieval_service: RetrievalService) -> RagSearchHandler:
    """
    Factory function to create the rag_search handler.
    
    Args:
        retrieval_service: The retrieval service to use
        
    Returns:
        Configured handler instance
    """
    return RagSearchHandler(retrieval_service)


__all__ = [
    "rag_search_tool",
    "rag_search_handler",
    "RagSearchHandler",
]
