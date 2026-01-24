"""
Ingestion service.

Orchestrates the document ingestion pipeline:
1. Accept documents
2. Check for duplicates (idempotency)
3. Chunk documents
4. Generate embeddings
5. Store in vector database
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from genai_rag_service.chunking.base import ChunkingStrategy
from genai_rag_service.chunking.config import ChunkingConfig
from genai_rag_service.chunking.fixed_size import FixedSizeChunker
from genai_rag_service.chunking.recursive import RecursiveChunker
from genai_rag_service.domain.document import Document, DocumentChunk, DocumentMetadata
from genai_rag_service.domain.embedding import EmbeddingConfig, EmbeddingVector
from genai_rag_service.domain.ingestion import IngestionError, IngestionResult
from genai_rag_service.ports.embedding import EmbeddingProviderPort
from genai_rag_service.ports.vector_store import VectorStorePort


logger = structlog.get_logger(__name__)


def _generate_document_id(source_uri: str, content_hash: str) -> str:
    """Generate a deterministic document ID."""
    composite = f"{source_uri}:{content_hash}"
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class DocumentInput:
    """Input representation for a document to ingest."""
    
    uri: str
    content: str
    version: str = "1.0.0"
    metadata: Mapping[str, Any] | None = None


class IngestionService:
    """
    Service for ingesting documents into the RAG system.
    
    This service handles the complete ingestion pipeline:
    1. Parse and validate input documents
    2. Check for existing documents (idempotency)
    3. Chunk documents using configurable strategy
    4. Generate embeddings for chunks
    5. Store chunks and vectors in the vector store
    
    The service is stateless and can be horizontally scaled.
    All state is managed by the injected adapters.
    
    Attributes:
        vector_store: Vector store adapter for persistence
        embedding_provider: Embedding provider for vector generation
        embedding_config: Configuration for embedding generation
    """
    
    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_provider: EmbeddingProviderPort,
        embedding_config: EmbeddingConfig,
    ) -> None:
        """
        Initialize the ingestion service.
        
        Args:
            vector_store: Vector store adapter
            embedding_provider: Embedding provider adapter
            embedding_config: Embedding configuration
        """
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._embedding_config = embedding_config
    
    async def ingest(
        self,
        documents: Sequence[DocumentInput],
        chunking_config: ChunkingConfig | None = None,
    ) -> IngestionResult:
        """
        Ingest a batch of documents.
        
        This operation is idempotent: re-ingesting the same document
        (same URI and content) will skip it and not create duplicates.
        
        Args:
            documents: Sequence of documents to ingest
            chunking_config: Optional chunking configuration
            
        Returns:
            IngestionResult with counts and any errors
        """
        config = chunking_config or ChunkingConfig.default()
        chunker = self._create_chunker(config)
        
        ingested_count = 0
        skipped_count = 0
        total_chunk_count = 0
        document_ids: list[str] = []
        errors: list[IngestionError] = []
        
        for doc_input in documents:
            try:
                result = await self._ingest_single(doc_input, chunker)
                
                if result is None:
                    # Document already exists
                    skipped_count += 1
                    logger.info(
                        "document_skipped",
                        source_uri=doc_input.uri,
                        reason="already_exists",
                    )
                else:
                    ingested_count += 1
                    total_chunk_count += result.chunk_count
                    document_ids.append(result.document_id)
                    logger.info(
                        "document_ingested",
                        document_id=result.document_id,
                        source_uri=doc_input.uri,
                        chunk_count=result.chunk_count,
                    )
            except Exception as e:
                error = IngestionError(
                    document_uri=doc_input.uri,
                    error_code="INGESTION_FAILED",
                    message=str(e),
                    is_retryable=True,
                )
                errors.append(error)
                logger.error(
                    "document_ingestion_failed",
                    source_uri=doc_input.uri,
                    error=str(e),
                )
        
        return IngestionResult(
            ingested_count=ingested_count,
            skipped_count=skipped_count,
            chunk_count=total_chunk_count,
            document_ids=tuple(document_ids),
            errors=tuple(errors),
        )
    
    async def _ingest_single(
        self,
        doc_input: DocumentInput,
        chunker: ChunkingStrategy,
    ) -> Document | None:
        """
        Ingest a single document.
        
        Returns None if the document already exists (idempotency check).
        """
        # Generate content hash for idempotency
        content_hash = hashlib.sha256(doc_input.content.encode("utf-8")).hexdigest()
        document_id = _generate_document_id(doc_input.uri, content_hash)
        
        # Check if document already exists
        if await self._vector_store.document_exists(document_id):
            return None
        
        # Create metadata
        metadata = DocumentMetadata(
            source_uri=doc_input.uri,
            content_hash=content_hash,
            version=doc_input.version,
            custom_metadata=doc_input.metadata or {},
        )
        
        # Chunk the document
        chunks = chunker.chunk(doc_input.content, document_id, metadata)
        
        if not chunks:
            # Document too short to chunk
            logger.warning(
                "document_too_short",
                source_uri=doc_input.uri,
                content_length=len(doc_input.content),
            )
            return Document(
                document_id=document_id,
                chunks=(),
                metadata=metadata,
            )
        
        # Generate embeddings in batches
        vectors = await self._generate_embeddings(chunks)
        
        # Store in vector store
        await self._vector_store.upsert_vectors(vectors, chunks)
        
        return Document(
            document_id=document_id,
            chunks=tuple(chunks),
            metadata=metadata,
        )
    
    async def _generate_embeddings(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddingVector]:
        """Generate embeddings for chunks in batches."""
        model_id = self._embedding_config.model_id
        batch_size = self._embedding_config.batch_size
        
        vectors: list[EmbeddingVector] = []
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            embeddings = await self._embedding_provider.embed_texts(texts, model_id)
            
            for chunk, embedding in zip(batch, embeddings):
                vector = EmbeddingVector.create(
                    chunk_id=chunk.chunk_id,
                    vector=embedding,
                    model_id=model_id,
                )
                vectors.append(vector)
        
        return vectors
    
    def _create_chunker(self, config: ChunkingConfig) -> ChunkingStrategy:
        """Create a chunker based on configuration."""
        if config.strategy == "fixed_size":
            return FixedSizeChunker(config)
        elif config.strategy == "recursive":
            return RecursiveChunker(config)
        else:
            raise ValueError(f"Unknown chunking strategy: {config.strategy}")
    
    async def delete_document(self, document_id: str) -> int:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: The document to delete
            
        Returns:
            Number of chunks deleted
        """
        count = await self._vector_store.delete_by_document(document_id)
        logger.info("document_deleted", document_id=document_id, chunk_count=count)
        return count


__all__ = [
    "IngestionService",
    "DocumentInput",
]
