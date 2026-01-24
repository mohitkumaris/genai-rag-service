"""
Integration tests for the retrieval pipeline.
"""

import pytest

from genai_rag_service.chunking.config import ChunkingConfig
from genai_rag_service.domain.retrieval import SearchQuery
from genai_rag_service.services.ingestion import DocumentInput, IngestionService
from genai_rag_service.services.retrieval import RetrievalService


class TestRetrievalPipeline:
    """Integration tests for the retrieval pipeline."""
    
    @pytest.fixture
    async def indexed_documents(
        self,
        ingestion_service: IngestionService,
    ) -> list[str]:
        """Ingest test documents and return their IDs."""
        documents = [
            DocumentInput(
                uri="doc://test/machine-learning",
                content="""
                Machine learning is a type of artificial intelligence that 
                allows software applications to become more accurate at 
                predicting outcomes without being explicitly programmed.
                Machine learning algorithms use historical data as input 
                to predict new output values.
                """.strip(),
                metadata={"topic": "ml"},
            ),
            DocumentInput(
                uri="doc://test/deep-learning",
                content="""
                Deep learning is a subset of machine learning that uses 
                artificial neural networks inspired by the human brain.
                Deep learning models can automatically learn representations 
                from data without manual feature engineering.
                """.strip(),
                metadata={"topic": "dl"},
            ),
            DocumentInput(
                uri="doc://test/nlp",
                content="""
                Natural language processing enables computers to understand,
                interpret, and generate human language. NLP combines 
                computational linguistics with machine learning to process
                text and speech data.
                """.strip(),
                metadata={"topic": "nlp"},
            ),
        ]
        
        config = ChunkingConfig(
            strategy="fixed_size",
            chunk_size=50,
            chunk_overlap=5,
            min_chunk_size=10,
        )
        
        result = await ingestion_service.ingest(documents, config)
        return list(result.document_ids)
    
    @pytest.mark.asyncio
    async def test_vector_search(
        self,
        retrieval_service: RetrievalService,
        indexed_documents: list[str],
    ) -> None:
        """Test vector similarity search."""
        query = SearchQuery(
            query_text="machine learning prediction",
            top_k=5,
            search_type="vector",
        )
        
        context = await retrieval_service.search(query)
        
        assert context.has_results
        assert len(context.results) <= 5
        assert context.latency_ms >= 0
        
        # Results should be sorted by score (descending)
        scores = [r.score for r in context.results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_hybrid_search(
        self,
        retrieval_service: RetrievalService,
        indexed_documents: list[str],
    ) -> None:
        """Test hybrid (vector + keyword) search."""
        query = SearchQuery(
            query_text="neural networks deep learning",
            top_k=5,
            search_type="hybrid",
            keyword_weight=0.3,
        )
        
        context = await retrieval_service.search(query)
        
        assert context.has_results
        for result in context.results:
            assert result.match_type == "hybrid"
    
    @pytest.mark.asyncio
    async def test_search_with_threshold(
        self,
        retrieval_service: RetrievalService,
        indexed_documents: list[str],
    ) -> None:
        """Test search with similarity threshold."""
        query = SearchQuery(
            query_text="machine learning",
            top_k=10,
            search_type="vector",
            similarity_threshold=0.5,  # Only highly relevant results
        )
        
        context = await retrieval_service.search(query)
        
        # All results should meet threshold
        for result in context.results:
            assert result.score >= 0.5
    
    @pytest.mark.asyncio
    async def test_search_empty_index(
        self,
        retrieval_service: RetrievalService,
    ) -> None:
        """Test search on empty index."""
        query = SearchQuery(
            query_text="some random query",
            top_k=5,
        )
        
        context = await retrieval_service.search(query)
        
        assert not context.has_results
        assert context.result_count == 0


class TestRetrievalMetadata:
    """Tests for retrieval with metadata filtering."""
    
    @pytest.fixture
    async def indexed_with_metadata(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Ingest documents with different metadata."""
        documents = [
            DocumentInput(
                uri="doc://test/cat1-doc1",
                content="First document in category one with detailed content about topic A.",
                metadata={"category": "cat1"},
            ),
            DocumentInput(
                uri="doc://test/cat2-doc1",
                content="First document in category two with detailed content about topic B.",
                metadata={"category": "cat2"},
            ),
        ]
        
        config = ChunkingConfig(
            strategy="fixed_size",
            chunk_size=50,
            chunk_overlap=5,
            min_chunk_size=10,
        )
        
        await ingestion_service.ingest(documents, config)
    
    @pytest.mark.asyncio
    async def test_search_with_filter(
        self,
        retrieval_service: RetrievalService,
        indexed_with_metadata: None,
    ) -> None:
        """Test search with metadata filter."""
        query = SearchQuery(
            query_text="document content topic",
            top_k=10,
            filters={"category": "cat1"},
        )
        
        context = await retrieval_service.search(query)
        
        # All results should have category=cat1
        for result in context.results:
            assert result.chunk.metadata.custom_metadata.get("category") == "cat1"
