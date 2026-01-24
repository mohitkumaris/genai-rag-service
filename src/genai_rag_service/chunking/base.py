"""
Base chunking strategy interface.

Defines the protocol that all chunking implementations must follow.
"""

from typing import Protocol

from genai_rag_service.domain.document import DocumentChunk, DocumentMetadata


class ChunkingStrategy(Protocol):
    """
    Protocol for chunking implementations.
    
    All chunking strategies must be deterministic: the same input text
    with the same configuration must always produce identical chunks.
    This is critical for idempotent ingestion.
    
    Implementations must:
    - Generate consistent chunk IDs
    - Preserve token counts accurately
    - Handle edge cases (empty text, very short text)
    """
    
    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """
        Split text into chunks.
        
        The chunking must be deterministic: calling this method with the
        same arguments must always produce the same output.
        
        Args:
            text: The full document text to chunk
            document_id: Identifier for the parent document
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of document chunks in order
            
        Note:
            An empty list may be returned if the text is too short
            to produce any valid chunks.
        """
        ...
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        This should use the same tokenizer that will be used by the
        embedding model to ensure accurate chunk sizing.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        ...


__all__ = [
    "ChunkingStrategy",
]
