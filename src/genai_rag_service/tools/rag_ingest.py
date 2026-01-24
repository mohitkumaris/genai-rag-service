"""
RAG Ingest Tool.

MCP tool for ingesting documents into the RAG system.
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

from genai_rag_service.chunking.config import ChunkingConfig
from genai_rag_service.services.ingestion import DocumentInput, IngestionService


# Tool definition
rag_ingest_tool = ToolDefinition(
    name="rag_ingest",
    description=(
        "Ingest documents into the RAG system for later retrieval. "
        "Documents are chunked, embedded, and indexed for semantic search. "
        "This operation is idempotent - re-ingesting the same document is a no-op."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "description": "List of documents to ingest",
                "items": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "Unique identifier/source URI for the document",
                        },
                        "content": {
                            "type": "string",
                            "description": "The text content of the document",
                        },
                        "version": {
                            "type": "string",
                            "description": "Optional version string for tracking updates",
                            "default": "1.0.0",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional custom metadata for filtering",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["uri", "content"],
                },
                "minItems": 1,
                "maxItems": 100,
            },
            "chunking_config": {
                "type": "object",
                "description": "Optional chunking configuration",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["fixed_size", "recursive"],
                        "description": "Chunking strategy to use",
                        "default": "fixed_size",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 2000,
                        "description": "Target chunk size in tokens",
                        "default": 512,
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 500,
                        "description": "Overlap between chunks in tokens",
                        "default": 50,
                    },
                },
            },
        },
        "required": ["documents"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "ingested_count": {
                "type": "integer",
                "description": "Number of new documents ingested",
            },
            "skipped_count": {
                "type": "integer",
                "description": "Number of documents skipped (already exist)",
            },
            "chunk_count": {
                "type": "integer",
                "description": "Total number of chunks created",
            },
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of successfully ingested documents",
            },
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "document_uri": {"type": "string"},
                        "error_code": {"type": "string"},
                        "message": {"type": "string"},
                        "is_retryable": {"type": "boolean"},
                    },
                },
                "description": "Errors encountered during ingestion",
            },
        },
        "required": ["ingested_count", "document_ids"],
    },
    version="1.0.0",
    category="rag",
    tags=frozenset(["retrieval", "ingestion", "documents", "embeddings"]),
    required_permissions=frozenset(["rag:ingest"]),
)


class RagIngestHandler:
    """
    Handler for the rag_ingest tool.
    
    This handler wraps the IngestionService and provides the MCP
    tool interface for document ingestion.
    """
    
    def __init__(self, ingestion_service: IngestionService) -> None:
        """
        Initialize the handler.
        
        Args:
            ingestion_service: The ingestion service to use
        """
        self._ingestion_service = ingestion_service
    
    async def __call__(
        self,
        context: MCPContext,
        input_data: Mapping[str, Any],
    ) -> ToolResult:
        """
        Execute the rag_ingest tool.
        
        Args:
            context: MCP context with permissions and metadata
            input_data: Tool input matching the input_schema
            
        Returns:
            ToolSuccess with ingestion results, or ToolFailure on error
        """
        try:
            # Parse documents from input
            raw_documents = input_data.get("documents", [])
            if not raw_documents:
                return ToolFailure(
                    error_code="INVALID_INPUT",
                    message="No documents provided",
                    is_retryable=False,
                )
            
            documents = [
                DocumentInput(
                    uri=doc["uri"],
                    content=doc["content"],
                    version=doc.get("version", "1.0.0"),
                    metadata=doc.get("metadata"),
                )
                for doc in raw_documents
            ]
            
            # Parse chunking config if provided
            chunking_config = None
            raw_config = input_data.get("chunking_config")
            if raw_config:
                chunking_config = ChunkingConfig(
                    strategy=raw_config.get("strategy", "fixed_size"),
                    chunk_size=raw_config.get("chunk_size", 512),
                    chunk_overlap=raw_config.get("chunk_overlap", 50),
                )
            
            # Execute ingestion
            result = await self._ingestion_service.ingest(
                documents=documents,
                chunking_config=chunking_config,
            )
            
            # Build response
            return ToolSuccess(
                data={
                    "ingested_count": result.ingested_count,
                    "skipped_count": result.skipped_count,
                    "chunk_count": result.chunk_count,
                    "document_ids": list(result.document_ids),
                    "errors": [error.to_dict() for error in result.errors],
                },
                metadata={
                    "request_id": context.request_id,
                    "trace_id": context.trace_id,
                },
            )
            
        except Exception as e:
            return ToolFailure(
                error_code="INGESTION_ERROR",
                message=str(e),
                details={"exception_type": type(e).__name__},
                is_retryable=True,
            )


def rag_ingest_handler(ingestion_service: IngestionService) -> RagIngestHandler:
    """
    Factory function to create the rag_ingest handler.
    
    Args:
        ingestion_service: The ingestion service to use
        
    Returns:
        Configured handler instance
    """
    return RagIngestHandler(ingestion_service)


__all__ = [
    "rag_ingest_tool",
    "rag_ingest_handler",
    "RagIngestHandler",
]
