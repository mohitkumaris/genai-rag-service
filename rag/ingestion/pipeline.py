"""
Ingestion Pipeline.

Orchestrates the loading, chunking, embedding, and storage of documents.
"""

import hashlib
import structlog
from dataclasses import dataclass, field
from typing import Sequence, Any
from rag.schemas import (
    Document, DocumentMetadata, EmbeddingVector, 
    VectorStorePort, EmbeddingProviderPort
)
from rag.ingestion.chunker import (
    ChunkingStrategy, ChunkingConfig, FixedSizeChunker, RecursiveChunker
)

logger = structlog.get_logger(__name__)

@dataclass(frozen=True)
class IngestionRequest:
    uri: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class IngestionResult:
    ingested_count: int
    skipped_count: int
    chunk_count: int
    document_ids: tuple[str, ...]
    errors: tuple[dict[str, Any], ...]

class IngestionPipeline:
    """
    Core ingestion logic.
    Dependent on Ports (interfaces), NOT concrete adapters.
    """
    
    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_provider: EmbeddingProviderPort,
        embedding_model_id: str = "text-embedding-ada-002",
        embedding_dimension: int = 1536,
        batch_size: int = 32,
    ):
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._embedding_model_id = embedding_model_id
        self._embedding_dimension = embedding_dimension
        self._batch_size = batch_size

    async def run(
        self, 
        requests: Sequence[IngestionRequest],
        chunking_config: ChunkingConfig | None = None
    ) -> IngestionResult:
        """
        Process a batch of documents idempotently.
        """
        config = chunking_config or ChunkingConfig()
        chunker = self._create_chunker(config)
        
        ingested = 0
        skipped = 0
        total_chunks = 0
        doc_ids = []
        errors = []

        for req in requests:
            try:
                # 1. Deterministic ID Generation
                content_hash = hashlib.sha256(req.content.encode("utf-8")).hexdigest()
                doc_id = hashlib.sha256(f"{req.uri}:{content_hash}".encode("utf-8")).hexdigest()[:32]
                
                # 2. Idempotency Check
                if await self._vector_store.document_exists(doc_id):
                    skipped += 1
                    logger.info("document_skipped", uri=req.uri, reason="exists")
                    continue

                # 3. Create Metadata
                meta = DocumentMetadata(
                    source_uri=req.uri,
                    content_hash=content_hash,
                    custom_metadata=req.metadata
                )

                # 4. Chunk
                chunks = chunker.chunk(req.content, doc_id, meta)
                if not chunks:
                    logger.warning("document_too_short", uri=req.uri)
                    # We accept it but store nothing? Or store empty doc placeholder?
                    # For now just log and continue, technically ingested 0 chunks.
                    skipped += 1 # Treated as skip/noop
                    continue

                # 5. Embed
                vectors = []
                for i in range(0, len(chunks), self._batch_size):
                    batch = chunks[i : i + self._batch_size]
                    texts = [c.content for c in batch]
                    embeddings = await self._embedding_provider.embed_texts(texts, self._embedding_model_id)
                    
                    for chunk, emb in zip(batch, embeddings):
                        vectors.append(EmbeddingVector(
                            chunk_id=chunk.chunk_id,
                            vector=emb,
                            model_id=self._embedding_model_id
                        ))

                # 6. Store
                await self._vector_store.upsert_vectors(vectors, chunks)
                
                ingested += 1
                total_chunks += len(chunks)
                doc_ids.append(doc_id)
                
                logger.info("document_ingested", uri=req.uri, chunks=len(chunks))

            except Exception as e:
                logger.error("ingestion_failed", uri=req.uri, error=str(e))
                errors.append({"uri": req.uri, "error": str(e)})

        return IngestionResult(
            ingested_count=ingested,
            skipped_count=skipped,
            chunk_count=total_chunks,
            document_ids=tuple(doc_ids),
            errors=tuple(errors)
        )

    def _create_chunker(self, config: ChunkingConfig) -> ChunkingStrategy:
        if config.strategy == "recursive":
            return RecursiveChunker(config)
        return FixedSizeChunker(config)

    async def delete_document(self, document_id: str) -> int:
        """Delete a document by ID."""
        return await self._vector_store.delete_by_document(document_id)
