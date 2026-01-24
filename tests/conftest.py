"""
Test configuration and fixtures.
"""

import pytest

from genai_rag_service.adapters.embedding.mock import MockEmbeddingProvider
from genai_rag_service.adapters.vector.memory import InMemoryVectorStore
from genai_rag_service.chunking.config import ChunkingConfig
from genai_rag_service.chunking.fixed_size import FixedSizeChunker
from genai_rag_service.config.settings import Settings
from genai_rag_service.domain.embedding import EmbeddingConfig
from genai_rag_service.services.ingestion import IngestionService
from genai_rag_service.services.retrieval import RetrievalService


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        environment="development",
        embedding_model_id="mock-large",
        embedding_dimension=1536,
    )


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Create an in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def embedding_provider() -> MockEmbeddingProvider:
    """Create a mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def embedding_config() -> EmbeddingConfig:
    """Create embedding configuration."""
    return EmbeddingConfig(
        model_id="mock-large",
        dimension=1536,
        batch_size=32,
    )


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
    embedding_config: EmbeddingConfig,
) -> IngestionService:
    """Create an ingestion service."""
    return IngestionService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        embedding_config=embedding_config,
    )


@pytest.fixture
def retrieval_service(
    vector_store: InMemoryVectorStore,
    embedding_provider: MockEmbeddingProvider,
    embedding_config: EmbeddingConfig,
) -> RetrievalService:
    """Create a retrieval service."""
    return RetrievalService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        embedding_config=embedding_config,
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
