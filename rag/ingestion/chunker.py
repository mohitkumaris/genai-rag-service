"""
Document chunking strategies.

Implements deterministic splitting logic.
"""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import tiktoken
from rag.schemas import DocumentMetadata, DocumentChunk

@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for chunking strategies."""
    strategy: str = "fixed_size"  # fixed_size, recursive
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    model_name: str = "gpt-4"
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    @property
    def effective_chunk_size(self) -> int:
        return self.chunk_size - self.chunk_overlap


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies."""
    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]: ...
    def count_tokens(self, text: str) -> int: ...


class BaseChunker:
    """Base class for chunkers using tiktoken."""
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


class FixedSizeChunker(BaseChunker):
    """Splits text into fixed-size token chunks with overlap."""
    
    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        if not text or not text.strip():
            return []
            
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return []
            
        chunks = []
        step = self.config.chunk_size - self.config.chunk_overlap
        
        for i, start_idx in enumerate(range(0, len(tokens), step)):
            end_idx = min(start_idx + self.config.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Skip chunks that are too small unless it's the only one
            if len(chunk_tokens) < self.config.min_chunk_size and len(chunks) > 0:
                continue
                
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append(DocumentChunk.create(
                document_id=document_id,
                content=chunk_text,
                chunk_index=i,
                token_count=len(chunk_tokens),
                metadata=metadata,
            ))
            
        return chunks


class RecursiveChunker(BaseChunker):
    """Splits text by natural boundaries recursively."""
    
    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        raw_chunks = self._recursive_split(
            text, 
            list(self.config.separators),
            self.config.chunk_size
        )
        
        return [
            DocumentChunk.create(
                document_id=document_id,
                content=content,
                chunk_index=i,
                token_count=self.count_tokens(content),
                metadata=metadata,
            )
            for i, content in enumerate(raw_chunks)
        ]

    def _recursive_split(self, text: str, separators: list[str], max_size: int) -> list[str]:
        final_chunks = []
        current_step_chunks = []
        
        separator = separators[0] if separators else ""
        next_separators = separators[1:] if len(separators) > 1 else []
        
        if not separator:
            # Base case: no more separators, just split by chars if needed? 
            # Or just return text as is if it fits, or force split.
            # Simplified: just return text.
            return [text] if text.strip() else []

        # Split using current separator
        splits = text.split(separator)
        
        current_chunk = ""
        
        for split in splits:
            # Re-attach separator for semantic coherence (heuristic)
            # Generally we assume separator is consumed, but for sentence splitting ". " we might want to keep it?
            # Keeping it simple: separator is lost in split.
            
            # If the split itself is too big, recurse on it
            if self.count_tokens(split) > max_size and next_separators:
                sub_chunks = self._recursive_split(split, next_separators, max_size)
                # Try to accumulate sub_chunks
                for sub in sub_chunks:
                    if self.count_tokens(current_chunk + (separator if current_chunk else "") + sub) <= max_size:
                        current_chunk += (separator if current_chunk else "") + sub
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk)
                        current_chunk = sub
            else:
                # Add to current chunk if it fits
                candidate = current_chunk + (separator if current_chunk else "") + split
                if self.count_tokens(candidate) <= max_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    current_chunk = split
                    
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks
