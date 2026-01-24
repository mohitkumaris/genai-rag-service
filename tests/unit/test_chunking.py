"""
Unit tests for chunking strategies.
"""

import pytest
from rag.ingestion.chunker import (
    ChunkingConfig,
    FixedSizeChunker,
    RecursiveChunker,
)
from rag.schemas import DocumentMetadata


class TestChunkingConfig:
    """Tests for ChunkingConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ChunkingConfig()
        
        assert config.strategy == "fixed_size"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
    
    def test_effective_chunk_size(self) -> None:
        """Test effective chunk size calculation."""
        config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
        assert config.effective_chunk_size == 462


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""
    
    @pytest.fixture
    def chunker(self) -> FixedSizeChunker:
        """Create a chunker with small chunks for testing."""
        config = ChunkingConfig(
            strategy="fixed_size",
            chunk_size=50,  # Small for testing
            chunk_overlap=10,
            min_chunk_size=20,
        )
        return FixedSizeChunker(config)
    
    @pytest.fixture
    def metadata(self) -> DocumentMetadata:
        """Create test metadata."""
        return DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
        )
    
    def test_chunk_text(
        self,
        chunker: FixedSizeChunker,
        metadata: DocumentMetadata,
    ) -> None:
        """Test chunking text into multiple chunks."""
        text = """
        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks with multiple layers.
        Natural language processing focuses on human-computer interaction.
        """.strip()
        
        chunks = chunker.chunk(text, "doc-123", metadata)
        
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == "doc-123"
            assert chunk.chunk_index == i
            assert chunk.content
            assert chunk.token_count > 0
    
    def test_chunk_deterministic(
        self,
        chunker: FixedSizeChunker,
        metadata: DocumentMetadata,
    ) -> None:
        """Test that chunking is deterministic."""
        text = "This is a sample text for chunking that should produce consistent results."
        
        chunks1 = chunker.chunk(text, "doc-123", metadata)
        chunks2 = chunker.chunk(text, "doc-123", metadata)
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.content == c2.content
    
    def test_empty_text(
        self,
        chunker: FixedSizeChunker,
        metadata: DocumentMetadata,
    ) -> None:
        """Test chunking empty text."""
        chunks = chunker.chunk("", "doc-123", metadata)
        assert chunks == []
        
        chunks = chunker.chunk("   ", "doc-123", metadata)
        assert chunks == []
    
    def test_short_text(
        self,
        chunker: FixedSizeChunker,
        metadata: DocumentMetadata,
    ) -> None:
        """Test chunking text shorter than min_chunk_size."""
        short_text = "Hi"
        chunks = chunker.chunk(short_text, "doc-123", metadata)
        
        # Current implementation preserves the only chunk even if small
        assert len(chunks) == 1
        assert chunks[0].content == "Hi"
    
    def test_token_counting(self, chunker: FixedSizeChunker) -> None:
        """Test token counting."""
        text = "Hello world, this is a test."
        token_count = chunker.count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""
    
    @pytest.fixture
    def chunker(self) -> RecursiveChunker:
        """Create a chunker with small chunks for testing."""
        config = ChunkingConfig(
            strategy="recursive",
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=30,
            separators=("\n\n", "\n", ". ", " "),
        )
        return RecursiveChunker(config)
    
    @pytest.fixture
    def metadata(self) -> DocumentMetadata:
        """Create test metadata."""
        return DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
        )
    
    def test_chunk_by_paragraphs(
        self,
        chunker: RecursiveChunker,
        metadata: DocumentMetadata,
    ) -> None:
        """Test chunking by paragraph boundaries."""
        text = """
First paragraph about machine learning and its applications in modern technology.

Second paragraph about deep learning and neural networks with multiple layers.

Third paragraph about natural language processing and human computer interaction.
        """.strip()
        
        chunks = chunker.chunk(text, "doc-123", metadata)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content
    
    def test_chunk_deterministic(
        self,
        chunker: RecursiveChunker,
        metadata: DocumentMetadata,
    ) -> None:
        """Test that recursive chunking is deterministic."""
        text = """
Paragraph one with some content.

Paragraph two with more content.
        """.strip()
        
        chunks1 = chunker.chunk(text, "doc-123", metadata)
        chunks2 = chunker.chunk(text, "doc-123", metadata)
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
