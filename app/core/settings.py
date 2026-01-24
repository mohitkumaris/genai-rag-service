"""
Application configuration.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings using Pydantic."""
    
    # Environment
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "console"
    
    # RAG
    EMBEDDING_MODEL_ID: str = "text-embedding-ada-002"
    
    # Infra config placeholders
    STORAGE_TYPE: Literal["memory", "azure"] = "memory"
    VECTOR_STORE_TYPE: Literal["memory", "azure"] = "memory"
    
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore"
    )

settings = Settings()
