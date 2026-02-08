"""
rag_ingest MCP Tool.
"""

import structlog
from typing import Any
from genai_mcp_core import MCPContext, ToolDefinition, ToolHandler, ToolResult
from rag.ingestion.pipeline import IngestionPipeline, IngestionRequest
from rag.ingestion.chunker import ChunkingConfig

logger = structlog.get_logger(__name__)

class RagIngestHandler(ToolHandler):
    """Handler for the rag_ingest tool."""

    def __init__(self, pipeline: IngestionPipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, arguments: dict[str, Any], context: MCPContext) -> ToolResult:
        logger.info("rag_ingest_invoked", request_id=context.request_id, user_id=context.user_id)
        
        docs_data = arguments.get("documents", [])
        chunking = arguments.get("chunking", {})
        
        requests = [
            IngestionRequest(
                uri=d["uri"],
                content=d["content"],
                metadata=d.get("metadata", {})
            ) 
            for d in docs_data
        ]

        config = None
        if chunking:
            config = ChunkingConfig(
                strategy=chunking.get("strategy", "fixed_size"),
                chunk_size=chunking.get("chunk_size", 512),
                chunk_overlap=chunking.get("chunk_overlap", 50)
            )

        result = await self._pipeline.run(requests, config)

        if result.errors:
            # Partial success is mostly success, but let's report details
            pass

        return ToolResult.success({
            "ingested_count": result.ingested_count,
            "skipped_count": result.skipped_count,
            "chunk_count": result.chunk_count,
            "document_ids": list(result.document_ids),
            "errors": [dict(e) for e in result.errors]
        })

rag_ingest_tool = ToolDefinition(
    name="rag_ingest",
    description="Ingest documents into the RAG system with automatic chunking and embedding. Idempotent.",
    input_schema={
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string", "description": "Unique URI for the document"},
                        "content": {"type": "string", "description": "Text content"},
                        "metadata": {"type": "object", "description": "Custom metadata"}
                    },
                    "required": ["uri", "content"]
                }
            },
            "chunking": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": ["fixed_size", "recursive"]},
                    "chunk_size": {"type": "integer"},
                    "chunk_overlap": {"type": "integer"}
                }
            }
        },
        "required": ["documents"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "ingested_count": {"type": "integer"},
            "skipped_count": {"type": "integer"},
            "chunk_count": {"type": "integer"},
            "document_ids": {"type": "array", "items": {"type": "string"}},
            "errors": {"type": "array"}
        },
        "required": ["ingested_count", "chunk_count", "document_ids"]
    },
    required_permissions=frozenset(["rag:ingest"])
)
