"""
Dependency Injection wiring.

Connects concrete adapters to functionality.
"""

from functools import lru_cache
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.search import SearchService
from adapters.storage.memory import InMemoryBlobStorage
from adapters.vector_store.memory import InMemoryVectorStore
from adapters.embeddings.mock import MockEmbeddingProvider
from genai_mcp_core import ToolRegistry
from mcp_tools.rag_ingest import RagIngestHandler, rag_ingest_tool
from mcp_tools.rag_search import RagSearchHandler, rag_search_tool

# Singletons (In a real app, scope accordingly)
vector_store = InMemoryVectorStore()
embedding_provider = MockEmbeddingProvider()
storage = InMemoryBlobStorage()

@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        vector_store=vector_store,
        embedding_provider=embedding_provider
    )

@lru_cache
def get_search_service() -> SearchService:
    return SearchService(
        vector_store=vector_store,
        embedding_provider=embedding_provider
    )

@lru_cache
def get_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    
    # Register RAG Ingest
    registry.register_tool(
        tool=rag_ingest_tool,
        handler=RagIngestHandler(get_ingestion_pipeline())
    )
    
    # Register RAG Search
    registry.register_tool(
        tool=rag_search_tool,
        handler=RagSearchHandler(get_search_service())
    )
    
    return registry
