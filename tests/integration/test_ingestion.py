"""
Integration tests for the ingestion pipeline.
"""

import pytest

from genai_rag_service.chunking.config import ChunkingConfig
from genai_rag_service.services.ingestion import DocumentInput, IngestionService


class TestIngestionPipeline:
    """Integration tests for the ingestion pipeline."""
    
    @pytest.mark.asyncio
    async def test_ingest_single_document(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test ingesting a single document."""
        documents = [
            DocumentInput(
                uri="doc://test/single",
                content="""
                Machine learning is a subset of artificial intelligence that 
                enables systems to learn and improve from experience without 
                being explicitly programmed. It focuses on developing programs 
                that can access data and use it to learn for themselves.
                """.strip(),
                version="1.0.0",
                metadata={"category": "ml"},
            ),
        ]
        
        # Use smaller chunking config for test documents
        config = ChunkingConfig(
            strategy="fixed_size",
            chunk_size=50,
            chunk_overlap=5,
            min_chunk_size=10,
        )
        
        result = await ingestion_service.ingest(documents, config)
        
        assert result.ingested_count == 1
        assert result.skipped_count == 0
        assert len(result.document_ids) == 1
        assert result.chunk_count > 0
        assert not result.has_errors
    
    @pytest.mark.asyncio
    async def test_ingest_batch_documents(
        self,
        ingestion_service: IngestionService,
        sample_documents: list[dict],
    ) -> None:
        """Test ingesting multiple documents."""
        documents = [
            DocumentInput(
                uri=doc["uri"],
                content=doc["content"],
                version=doc.get("version", "1.0.0"),
                metadata=doc.get("metadata"),
            )
            for doc in sample_documents
        ]
        
        config = ChunkingConfig(
            strategy="fixed_size",
            chunk_size=50,
            chunk_overlap=5,
            min_chunk_size=10,
        )
        
        result = await ingestion_service.ingest(documents, config)
        
        assert result.ingested_count == len(sample_documents)
        assert len(result.document_ids) == len(sample_documents)
    
    @pytest.mark.asyncio
    async def test_ingest_idempotent(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test that ingestion is idempotent."""
        documents = [
            DocumentInput(
                uri="doc://test/idempotent",
                content="Same content for testing idempotency in the system which needs to be long enough to chunk.",
            ),
        ]
        
        config = ChunkingConfig(
            strategy="fixed_size",
            chunk_size=50,
            chunk_overlap=5,
            min_chunk_size=10,
        )
        
        # First ingestion
        result1 = await ingestion_service.ingest(documents, config)
        assert result1.ingested_count == 1
        assert result1.chunk_count > 0  # Ensure chunks were created
        
        # Second ingestion (same document)
        result2 = await ingestion_service.ingest(documents, config)
        assert result2.ingested_count == 0
        assert result2.skipped_count == 1
    
    @pytest.mark.asyncio
    async def test_ingest_with_custom_chunking(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test ingestion with custom chunking config."""
        documents = [
            DocumentInput(
                uri="doc://test/custom-chunking",
                content="""
                This is a longer document that should be split into multiple 
                chunks based on the custom chunking configuration. The content
                continues with more information about various topics including
                machine learning, deep learning, and artificial intelligence.
                """.strip(),
            ),
        ]
        
        config = ChunkingConfig(
            strategy="fixed_size",
            chunk_size=100,  # Smaller chunks
            chunk_overlap=10,
            min_chunk_size=30,
        )
        
        result = await ingestion_service.ingest(documents, config)
        
        assert result.ingested_count == 1
        assert result.chunk_count > 0


class TestIngestionWithDeletion:
    """Tests for ingestion and deletion."""
    
    @pytest.mark.asyncio
    async def test_delete_document(
        self,
        ingestion_service: IngestionService,
        vector_store,  # From fixture
    ) -> None:
        """Test deleting an ingested document."""
        # Ingest a document
        documents = [
            DocumentInput(
                uri="doc://test/to-delete",
                content="Content that will be deleted from the system. This needs to be long enough for chunking.",
            ),
        ]
        
        config = ChunkingConfig(
            strategy="fixed_size",
            chunk_size=50,
            chunk_overlap=5,
            min_chunk_size=10,
        )
        
        result = await ingestion_service.ingest(documents, config)
        assert result.ingested_count == 1
        assert result.chunk_count > 0  # Ensure chunks were created
        document_id = result.document_ids[0]
        
        # Verify document exists
        exists = await vector_store.document_exists(document_id)
        assert exists
        
        # Delete document
        deleted_count = await ingestion_service.delete_document(document_id)
        assert deleted_count > 0
        
        # Verify document no longer exists
        exists = await vector_store.document_exists(document_id)
        assert not exists
