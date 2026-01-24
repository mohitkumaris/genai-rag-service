"""
Application settings.

Centralized configuration management using Pydantic settings.
All settings can be overridden via environment variables prefixed with RAG_.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings have sensible defaults for development.
    Production deployments should override via environment variables.
    
    Environment variable prefix: RAG_
    
    Example:
        RAG_ENVIRONMENT=production
        RAG_EMBEDDING_MODEL_ID=text-embedding-3-large
        RAG_LOG_LEVEL=INFO
    """
    
    # Service identification
    service_name: str = "genai-rag-service"
    environment: Literal["development", "staging", "production"] = "development"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    
    # Embedding configuration
    embedding_model_id: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 32
    
    # Chunking defaults
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50
    default_chunking_strategy: Literal["fixed_size", "recursive"] = "fixed_size"
    
    # Vector store
    vector_store_type: Literal["memory", "azure_search", "pinecone"] = "memory"
    
    # Blob storage
    storage_type: Literal["memory", "azure_blob", "s3"] = "memory"
    
    # Embedding provider
    embedding_provider_type: Literal["mock", "openai", "azure_openai"] = "mock"
    
    # Azure configuration (when using Azure services)
    azure_search_endpoint: str | None = None
    azure_search_index_name: str = "rag-index"
    azure_storage_connection_string: str | None = None
    azure_storage_container: str = "documents"
    
    # OpenAI configuration (when using OpenAI embeddings)
    openai_api_key: str | None = None
    openai_organization: str | None = None
    
    # Request limits
    max_documents_per_request: int = 100
    max_search_results: int = 100
    request_timeout_seconds: int = 60
    
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"


def get_settings() -> Settings:
    """
    Get the application settings.
    
    This function creates a new Settings instance each time.
    For caching, use FastAPI's dependency injection with lru_cache.
    
    Returns:
        Configured Settings instance
    """
    return Settings()


__all__ = [
    "Settings",
    "get_settings",
]
