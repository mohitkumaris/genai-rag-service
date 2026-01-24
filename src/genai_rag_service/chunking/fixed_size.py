"""
Fixed-size chunking strategy.

Splits text into chunks of a fixed token size with configurable overlap.
This is the simplest and most predictable chunking strategy.
"""

import tiktoken

from genai_rag_service.chunking.config import ChunkingConfig
from genai_rag_service.domain.document import DocumentChunk, DocumentMetadata


class FixedSizeChunker:
    """
    Fixed-size token-based chunker with overlap.
    
    This chunker splits text into chunks of approximately equal token
    count, with configurable overlap between consecutive chunks.
    
    The chunking is deterministic: the same text with the same config
    will always produce identical chunks.
    
    Attributes:
        config: Chunking configuration
        _encoding: The tiktoken encoding for token counting
    """
    
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """
        Initialize the chunker.
        
        Args:
            config: Chunking configuration (uses defaults if None)
        """
        self.config = config or ChunkingConfig.default()
        self._encoding = tiktoken.get_encoding(self.config.model_name)
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self._encoding.encode(text))
    
    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        The algorithm:
        1. Tokenize the entire text
        2. Split into chunks of chunk_size tokens
        3. Each chunk starts (chunk_size - overlap) tokens after the previous
        4. Decode tokens back to text for each chunk
        
        Args:
            text: The full document text to chunk
            document_id: Identifier for the parent document
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of document chunks in order
        """
        if not text or not text.strip():
            return []
        
        # Tokenize the text
        tokens = self._encoding.encode(text)
        total_tokens = len(tokens)
        
        # Handle case where text is shorter than min_chunk_size
        if total_tokens < self.config.min_chunk_size:
            return []
        
        chunks: list[DocumentChunk] = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = chunk_size - overlap  # How many tokens to advance each iteration
        
        chunk_index = 0
        start = 0
        
        while start < total_tokens:
            # Calculate end position for this chunk
            end = min(start + chunk_size, total_tokens)
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start:end]
            chunk_token_count = len(chunk_tokens)
            
            # Skip if chunk is too small (except for the last chunk)
            if chunk_token_count < self.config.min_chunk_size:
                if start > 0:
                    # This is a trailing chunk that's too small, skip it
                    break
                else:
                    # This is the only chunk, but it's too small
                    return []
            
            # Decode tokens back to text
            chunk_text = self._encoding.decode(chunk_tokens)
            
            # Create the chunk using the factory method
            chunk = DocumentChunk.create(
                document_id=document_id,
                content=chunk_text,
                chunk_index=chunk_index,
                token_count=chunk_token_count,
                metadata=metadata,
            )
            chunks.append(chunk)
            
            chunk_index += 1
            start += step
            
            # Don't create overlapping chunks at the end if we've covered everything
            if end >= total_tokens:
                break
        
        return chunks
    
    def estimate_chunk_count(self, text: str) -> int:
        """
        Estimate the number of chunks for a text.
        
        This is useful for progress reporting without actually chunking.
        
        Args:
            text: The text to estimate chunks for
            
        Returns:
            Estimated number of chunks
        """
        total_tokens = self.count_tokens(text)
        if total_tokens < self.config.min_chunk_size:
            return 0
        
        step = self.config.chunk_size - self.config.chunk_overlap
        # Add 1 because we start at 0, then add 1 for any remaining tokens
        return max(1, (total_tokens - self.config.chunk_overlap + step - 1) // step)


__all__ = [
    "FixedSizeChunker",
]
