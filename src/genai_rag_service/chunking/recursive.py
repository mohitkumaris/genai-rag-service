"""
Recursive chunking strategy.

Splits text by natural boundaries (paragraphs, sentences, etc.) and only
falls back to fixed-size splitting when necessary.
"""

import tiktoken

from genai_rag_service.chunking.config import ChunkingConfig
from genai_rag_service.domain.document import DocumentChunk, DocumentMetadata


class RecursiveChunker:
    """
    Recursive text splitter that respects natural boundaries.
    
    This chunker attempts to split text at natural boundaries (paragraphs,
    sentences, etc.) while staying within the configured chunk size.
    
    The algorithm:
    1. Try to split by the first separator (e.g., paragraphs)
    2. If chunks are still too large, recursively split with next separator
    3. Continue until all chunks are within size limits
    4. Merge small chunks when possible
    
    This produces more semantically coherent chunks than fixed-size splitting.
    
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
        Split text into chunks using recursive splitting.
        
        Args:
            text: The full document text to chunk
            document_id: Identifier for the parent document
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of document chunks in order
        """
        if not text or not text.strip():
            return []
        
        # Get raw text chunks
        text_chunks = self._split_text(text, list(self.config.separators))
        
        # Filter out empty chunks and merge small ones
        text_chunks = self._merge_small_chunks(text_chunks)
        
        if not text_chunks:
            return []
        
        # Convert to DocumentChunk objects
        chunks: list[DocumentChunk] = []
        for i, chunk_text in enumerate(text_chunks):
            token_count = self.count_tokens(chunk_text)
            
            chunk = DocumentChunk.create(
                document_id=document_id,
                content=chunk_text,
                chunk_index=i,
                token_count=token_count,
                metadata=metadata,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text by separators.
        
        Args:
            text: Text to split
            separators: Ordered list of separators to try
            
        Returns:
            List of text segments
        """
        if not separators:
            # No more separators, return text as-is (may be oversized)
            return [text] if text.strip() else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by the current separator
        splits = text.split(separator)
        
        result: list[str] = []
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
            
            split_tokens = self.count_tokens(split)
            
            if split_tokens <= self.config.chunk_size:
                # This split fits, keep it
                result.append(split)
            else:
                # Still too large, try next separator
                sub_splits = self._split_text(split, remaining_separators)
                result.extend(sub_splits)
        
        return result
    
    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """
        Merge consecutive small chunks to improve chunk quality.
        
        Small chunks (below min_chunk_size) are merged with their neighbors
        when the combined size is still within chunk_size.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Merged list of chunks
        """
        if not chunks:
            return []
        
        merged: list[str] = []
        current = chunks[0]
        current_tokens = self.count_tokens(current)
        
        for chunk in chunks[1:]:
            chunk_tokens = self.count_tokens(chunk)
            combined = current + "\n\n" + chunk
            combined_tokens = self.count_tokens(combined)
            
            if (
                current_tokens < self.config.min_chunk_size
                and combined_tokens <= self.config.chunk_size
            ):
                # Merge small chunk with next
                current = combined
                current_tokens = combined_tokens
            elif (
                chunk_tokens < self.config.min_chunk_size
                and combined_tokens <= self.config.chunk_size
            ):
                # Merge next small chunk into current
                current = combined
                current_tokens = combined_tokens
            else:
                # Can't merge, emit current and start new
                if current_tokens >= self.config.min_chunk_size:
                    merged.append(current)
                current = chunk
                current_tokens = chunk_tokens
        
        # Don't forget the last chunk
        if current and current_tokens >= self.config.min_chunk_size:
            merged.append(current)
        elif current and merged:
            # Try to merge with last chunk
            last = merged[-1]
            combined = last + "\n\n" + current
            combined_tokens = self.count_tokens(combined)
            if combined_tokens <= self.config.chunk_size:
                merged[-1] = combined
        
        return merged
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """
        Add overlap between consecutive chunks.
        
        This version preserves natural boundaries but adds context from
        the previous chunk.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Chunks with overlap added
        """
        if len(chunks) <= 1 or self.config.chunk_overlap == 0:
            return chunks
        
        result: list[str] = [chunks[0]]
        overlap_tokens = self.config.chunk_overlap
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # Extract overlap from end of previous chunk
            prev_tokens = self._encoding.encode(prev_chunk)
            overlap_part = ""
            if len(prev_tokens) > overlap_tokens:
                overlap_tokens_list = prev_tokens[-overlap_tokens:]
                overlap_part = self._encoding.decode(overlap_tokens_list)
            
            # Add overlap to beginning of current chunk
            if overlap_part:
                current_chunk = overlap_part + "\n" + current_chunk
            
            result.append(current_chunk)
        
        return result


__all__ = [
    "RecursiveChunker",
]
