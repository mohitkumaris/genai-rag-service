"""
Unit tests for domain models.
"""

from datetime import datetime, timezone
import pytest
from rag.schemas import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingVector,
    SearchQuery,
    SearchResult,
    RetrievalContext,
)
from rag.ingestion.pipeline import IngestionResult

class TestDocumentMetadata:
    """Tests for DocumentMetadata."""
    
    def test_create_metadata(self) -> None:
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
            version="1.0.0",
        )
        
        assert metadata.source_uri == "doc://test/sample"
        assert metadata.content_hash == "abc123"
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.ingested_at, datetime)
    
    def test_metadata_immutable(self) -> None:
        """Test that metadata is immutable."""
        metadata = DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
        )
        
        with pytest.raises(AttributeError):  # FrozenInstanceError is a subclass
            metadata.source_uri = "new_uri"
    
    def test_metadata_validation(self) -> None:
        """Test metadata validation."""
        with pytest.raises(ValueError, match="source_uri cannot be empty"):
            DocumentMetadata(source_uri="", content_hash="abc123")
        
        with pytest.raises(ValueError, match="content_hash cannot be empty"):
            DocumentMetadata(source_uri="doc://test", content_hash="")
    
    def test_metadata_serialization(self) -> None:
        """Test metadata serialization."""
        metadata = DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
            version="1.0.0",
            custom_metadata={"key": "value"},
        )
        
        data = metadata.to_dict()
        restored = DocumentMetadata.from_dict(data)
        
        assert restored.source_uri == metadata.source_uri
        assert restored.content_hash == metadata.content_hash
        assert restored.version == metadata.version
        assert dict(restored.custom_metadata) == {"key": "value"}


class TestDocumentChunk:
    """Tests for DocumentChunk."""
    
    def test_create_chunk(self) -> None:
        """Test creating a document chunk."""
        metadata = DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
        )
        
        chunk = DocumentChunk.create(
            document_id="doc-123",
            content="This is the chunk content.",
            chunk_index=0,
            token_count=10,
            metadata=metadata,
        )
        
        assert chunk.document_id == "doc-123"
        assert chunk.content == "This is the chunk content."
        assert chunk.chunk_index == 0
        assert chunk.token_count == 10
        assert len(chunk.chunk_id) == 32  # Deterministic ID
    
    def test_chunk_deterministic_id(self) -> None:
        """Test that chunk IDs are deterministic."""
        metadata = DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
        )
        
        chunk1 = DocumentChunk.create(
            document_id="doc-123",
            content="Same content",
            chunk_index=0,
            token_count=5,
            metadata=metadata,
        )
        
        chunk2 = DocumentChunk.create(
            document_id="doc-123",
            content="Same content",
            chunk_index=0,
            token_count=5,
            metadata=metadata,
        )
        
        assert chunk1.chunk_id == chunk2.chunk_id


class TestDocument:
    """Tests for Document."""
    
    def test_create_document(self) -> None:
        """Test creating a document."""
        metadata = DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
        )
        
        doc = Document(
            document_id="doc-123",
            chunks=(),
            metadata=metadata
        )
        
        assert doc.document_id == "doc-123"
        assert doc.chunk_count == 0
        assert doc.metadata.source_uri == "doc://test/sample"
    
    def test_document_with_chunks(self) -> None:
        """Test document with chunks."""
        metadata = DocumentMetadata(
            source_uri="doc://test/sample",
            content_hash="abc123",
        )
        
        chunks = tuple(
            DocumentChunk.create(
                document_id="doc-123",
                content=f"Chunk {i}",
                chunk_index=i,
                token_count=3,
                metadata=metadata,
            )
            for i in range(3)
        )
        
        doc = Document(
            document_id="doc-123",
            chunks=chunks,
            metadata=metadata,
        )
        
        assert doc.chunk_count == 3


class TestEmbeddingVector:
    """Tests for EmbeddingVector."""
    
    def test_create_embedding(self) -> None:
        """Test creating an embedding vector."""
        vector = EmbeddingVector(
            chunk_id="chunk-123",
            vector=(0.1, 0.2, 0.3),
            model_id="test-model",
        )
        
        assert vector.chunk_id == "chunk-123"
        assert vector.dimension == 3
        assert len(vector.vector) == 3
    

class TestSearchQuery:
    """Tests for SearchQuery."""
    
    def test_create_query(self) -> None:
        """Test creating a search query."""
        query = SearchQuery(
            query_text="machine learning",
            top_k=5,
            search_type="vector",
        )
        
        assert query.query_text == "machine learning"
        assert query.top_k == 5
        assert query.search_type == "vector"

class TestIngestionResult:
    """Tests for IngestionResult."""
    
    def test_create_result(self) -> None:
        """Test creating an ingestion result."""
        result = IngestionResult(
            ingested_count=5,
            skipped_count=2,
            chunk_count=25,
            document_ids=("doc-1", "doc-2", "doc-3", "doc-4", "doc-5"),
            errors=()
        )
        
        assert result.ingested_count == 5
        assert len(result.errors) == 0
    
    def test_result_with_errors(self) -> None:
        """Test result with errors."""
        result = IngestionResult(
            ingested_count=3,
            skipped_count=0,
            chunk_count=15,
            document_ids=("doc-1", "doc-2", "doc-3"),
            errors=({"uri": "doc://failed", "error": "failed"},)
        )
        
        assert len(result.errors) == 1
