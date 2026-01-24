"""
FastAPI application factory.

Creates and configures the FastAPI application with all routes,
middleware, and dependency injection.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from genai_rag_service.adapters.embedding.mock import MockEmbeddingProvider
from genai_rag_service.adapters.vector.memory import InMemoryVectorStore
from genai_rag_service.api.middleware.logging import StructuredLoggingMiddleware
from genai_rag_service.api.middleware.tracing import RequestTracingMiddleware
from genai_rag_service.api.routes import health, tools
from genai_rag_service.config.settings import Settings
from genai_rag_service.domain.embedding import EmbeddingConfig
from genai_rag_service.observability.logging import configure_logging
from genai_rag_service.services.ingestion import IngestionService
from genai_rag_service.services.retrieval import RetrievalService
from genai_rag_service.tools.registry import create_tool_registry


def _create_adapters(settings: Settings) -> tuple:
    """
    Create adapters based on configuration.
    
    In production, this would create real adapters based on settings.
    For now, we use in-memory/mock adapters for all configurations.
    """
    # Vector store
    if settings.vector_store_type == "memory":
        vector_store = InMemoryVectorStore()
    else:
        # TODO: Implement production adapters
        vector_store = InMemoryVectorStore()
    
    # Embedding provider
    if settings.embedding_provider_type == "mock":
        embedding_provider = MockEmbeddingProvider()
    else:
        # TODO: Implement production adapters
        embedding_provider = MockEmbeddingProvider()
    
    return vector_store, embedding_provider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for resource management.
    """
    # Startup
    settings: Settings = app.state.settings
    configure_logging(
        level=settings.log_level,
        format=settings.log_format,
    )
    
    yield
    
    # Shutdown
    # Clean up resources here if needed


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Application factory for the RAG service.
    
    Creates a fully configured FastAPI application with:
    - Health check endpoints
    - MCP tool invocation endpoints
    - Structured logging middleware
    - Request tracing middleware
    - All necessary dependencies injected
    
    Args:
        settings: Optional settings override (uses defaults if None)
        
    Returns:
        Configured FastAPI application
    """
    settings = settings or Settings()
    
    app = FastAPI(
        title="GenAI RAG Service",
        description=(
            "Document retrieval and grounding service for the GenAI platform. "
            "Provides document ingestion, chunking, embedding, and semantic search "
            "via MCP-compatible tool interfaces."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Store settings on app state
    app.state.settings = settings
    
    # Create adapters
    vector_store, embedding_provider = _create_adapters(settings)
    app.state.vector_store = vector_store
    app.state.embedding_provider = embedding_provider
    
    # Create embedding config
    embedding_config = EmbeddingConfig(
        model_id=settings.embedding_model_id,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    app.state.embedding_config = embedding_config
    
    # Create services
    app.state.ingestion_service = IngestionService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        embedding_config=embedding_config,
    )
    app.state.retrieval_service = RetrievalService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        embedding_config=embedding_config,
    )
    
    # Create tool registry
    app.state.tool_registry = create_tool_registry(
        ingestion_service=app.state.ingestion_service,
        retrieval_service=app.state.retrieval_service,
    )
    
    # Add middleware (order matters - last added is first executed)
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(RequestTracingMiddleware)
    
    # Include routes
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(tools.router, prefix="/tools", tags=["Tools"])
    
    return app


# For running with uvicorn directly
def get_application() -> FastAPI:
    """Get the application instance for uvicorn."""
    return create_app()


__all__ = [
    "create_app",
    "get_application",
]
