"""
rag_search MCP Tool.
"""

import structlog
from typing import Any
from genai_mcp_core import MCPContext, ToolDefinition, ToolHandler, ToolResult
from rag.retrieval.search import SearchService
from rag.schemas import SearchQuery

logger = structlog.get_logger(__name__)

class RagSearchHandler(ToolHandler):
    """Handler for the rag_search tool."""

    def __init__(self, search_service: SearchService) -> None:
        self._service = search_service

    async def execute(self, arguments: dict[str, Any], context: MCPContext) -> ToolResult:
        logger.info("rag_search_invoked", request_id=context.request_id)
        
        query = SearchQuery(
            query_text=arguments["query"],
            top_k=arguments.get("top_k", 5),
            search_type=arguments.get("search_type", "vector"),
            similarity_threshold=arguments.get("similarity_threshold", 0.0),
            filters=arguments.get("filters"),
            keyword_weight=arguments.get("keyword_weight", 0.3)
        )

        ctx = await self._service.search(query)

        results_data = []
        for r in ctx.results:
            results_data.append({
                "content": r.chunk.content,
                "score": r.score,
                "source_uri": r.chunk.metadata.source_uri,
                "metadata": r.chunk.metadata.custom_metadata,
                "chunk_id": r.chunk.chunk_id
            })

        return ToolResult.success({
            "results": results_data,
            "count": len(results_data),
            "latency_ms": ctx.latency_ms,
            "total_searched": ctx.total_chunks_searched
        })

rag_search_tool = ToolDefinition(
    name="rag_search",
    description="Search for relevant context in the RAG knowledge base. Returns grounding context only.",
    permissions=["rag:search"],
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "search_type": {"type": "string", "enum": ["vector", "hybrid"], "default": "vector"},
            "filters": {"type": "object", "description": "Metadata filters"},
            "similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["query"]
    }
)
