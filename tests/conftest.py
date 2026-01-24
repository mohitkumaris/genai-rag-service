"""
Test configuration and fixtures.
"""

import pytest

from adapters.embeddings.mock import MockEmbeddingProvider
from adapters.vector_store.memory import InMemoryVectorStore
from rag.ingestion.chunker import ChunkingConfig, FixedSizeChunker
from app.core.settings import settings as app_settings
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.search import SearchService


@pytest.fixture
def settings():
    """Create test settings."""
    return app_settings


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Create an in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def embedding_provider() -> MockEmbeddingProvider:
    """Create a mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """Create chunking configuration."""
    return ChunkingConfig(
        strategy="fixed_size",
        chunk_size=50,  # Small for testing short documents
        chunk_overlap=5,
        min_chunk_size=10,  # Very small to allow test documents to produce chunks
    )


@pytest.fixture
def chunker(chunking_config: ChunkingConfig) -> FixedSizeChunker:
    """Create a fixed-size chunker."""
    return FixedSizeChunker(chunking_config)


@pytest.fixture
def ingestion_service(
    vector_store: InMemoryVectorStore,
    embedding_provider: MockEmbeddingProvider,
) -> IngestionPipeline:
    """Create an ingestion pipeline."""
    return IngestionPipeline(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        embedding_dimension=1536,
    )


@pytest.fixture
def retrieval_service(
    vector_store: InMemoryVectorStore,
    embedding_provider: MockEmbeddingProvider,
) -> SearchService:
    """Create a retrieval service."""
    return SearchService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )


# Sample documents for testing
@pytest.fixture
def sample_text() -> str:
    """Sample text for testing chunking."""
    return """
    Machine learning is a subset of artificial intelligence that enables systems 
    to learn and improve from experience without being explicitly programmed.
    
    Deep learning is a subset of machine learning that uses neural networks with 
    multiple layers to progressively extract higher-level features from raw input.
    
    Natural language processing is a field of AI that focuses on the interaction 
    between computers and humans using natural language.
    
    Computer vision is an interdisciplinary field that deals with how computers 
    can gain high-level understanding from digital images or videos.
    
    Reinforcement learning is an area of machine learning concerned with how 
    intelligent agents ought to take actions in an environment.
    """.strip()


@pytest.fixture
def sample_documents() -> list[dict]:
    """Sample documents for testing ingestion."""
    return [
        {
            "uri": "doc://test/ml-basics",
            "content": """
            Machine learning is a field of study that gives computers the ability 
            to learn without being explicitly programmed. It is a subset of 
            artificial intelligence based on the idea that systems can learn from 
            data, identify patterns and make decisions.
            """.strip(),
            "version": "1.0.0",
            "metadata": {"category": "ml", "author": "test"},
        },
        {
            "uri": "doc://test/deep-learning",
            "content": """
            Deep learning is part of a broader family of machine learning methods 
            based on artificial neural networks with representation learning. 
            Learning can be supervised, semi-supervised or unsupervised.
            """.strip(),
            "version": "1.0.0",
            "metadata": {"category": "dl", "author": "test"},
        },
    ]
