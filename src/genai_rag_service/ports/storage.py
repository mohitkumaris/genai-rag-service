"""
Blob storage port interface.

Defines the abstract interface for document storage operations.
Implementations can use Azure Blob Storage, S3, local filesystem, etc.
"""

from typing import Protocol


class BlobStoragePort(Protocol):
    """
    Port for document storage operations.
    
    This interface abstracts blob/object storage for raw documents.
    Implementations should handle connection management and retries.
    
    All methods are async for non-blocking I/O.
    """
    
    async def get_document(self, uri: str) -> bytes:
        """
        Retrieve raw document content by URI.
        
        Args:
            uri: The storage URI/path of the document
            
        Returns:
            Raw document content as bytes
            
        Raises:
            DocumentNotFoundError: If the document doesn't exist
            StorageError: If there's a storage access error
        """
        ...
    
    async def put_document(self, uri: str, content: bytes) -> None:
        """
        Store document content.
        
        Args:
            uri: The storage URI/path to store the document
            content: Raw document content as bytes
            
        Raises:
            StorageError: If there's a storage access error
        """
        ...
    
    async def exists(self, uri: str) -> bool:
        """
        Check if a document exists at the given URI.
        
        Args:
            uri: The storage URI/path to check
            
        Returns:
            True if the document exists, False otherwise
        """
        ...
    
    async def delete(self, uri: str) -> bool:
        """
        Delete a document from storage.
        
        Args:
            uri: The storage URI/path to delete
            
        Returns:
            True if deleted, False if it didn't exist
        """
        ...
    
    async def list_documents(self, prefix: str = "") -> list[str]:
        """
        List document URIs with the given prefix.
        
        Args:
            prefix: URI prefix to filter by
            
        Returns:
            List of document URIs matching the prefix
        """
        ...


class StorageError(Exception):
    """Base exception for storage errors."""
    
    def __init__(self, message: str, uri: str | None = None) -> None:
        super().__init__(message)
        self.uri = uri


class DocumentNotFoundError(StorageError):
    """Exception raised when a document is not found."""
    pass


__all__ = [
    "BlobStoragePort",
    "StorageError",
    "DocumentNotFoundError",
]
