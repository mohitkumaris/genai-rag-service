"""
In-memory blob storage adapter.

A simple in-memory implementation of the BlobStoragePort for testing
and development. All data is lost when the process terminates.
"""

from genai_rag_service.ports.storage import DocumentNotFoundError


class InMemoryBlobStorage:
    """
    In-memory implementation of BlobStoragePort.
    
    This adapter stores documents in a dictionary for testing purposes.
    It is not suitable for production use.
    
    Thread-safety: Not thread-safe. Use appropriate synchronization
    in multi-threaded contexts.
    """
    
    def __init__(self) -> None:
        """Initialize the in-memory storage."""
        self._storage: dict[str, bytes] = {}
    
    async def get_document(self, uri: str) -> bytes:
        """
        Retrieve raw document content by URI.
        
        Args:
            uri: The storage URI/path of the document
            
        Returns:
            Raw document content as bytes
            
        Raises:
            DocumentNotFoundError: If the document doesn't exist
        """
        if uri not in self._storage:
            raise DocumentNotFoundError(f"Document not found: {uri}", uri=uri)
        return self._storage[uri]
    
    async def put_document(self, uri: str, content: bytes) -> None:
        """
        Store document content.
        
        Args:
            uri: The storage URI/path to store the document
            content: Raw document content as bytes
        """
        self._storage[uri] = content
    
    async def exists(self, uri: str) -> bool:
        """
        Check if a document exists at the given URI.
        
        Args:
            uri: The storage URI/path to check
            
        Returns:
            True if the document exists, False otherwise
        """
        return uri in self._storage
    
    async def delete(self, uri: str) -> bool:
        """
        Delete a document from storage.
        
        Args:
            uri: The storage URI/path to delete
            
        Returns:
            True if deleted, False if it didn't exist
        """
        if uri in self._storage:
            del self._storage[uri]
            return True
        return False
    
    async def list_documents(self, prefix: str = "") -> list[str]:
        """
        List document URIs with the given prefix.
        
        Args:
            prefix: URI prefix to filter by
            
        Returns:
            List of document URIs matching the prefix
        """
        return [uri for uri in self._storage if uri.startswith(prefix)]
    
    def clear(self) -> None:
        """Clear all stored documents. For testing only."""
        self._storage.clear()
    
    @property
    def document_count(self) -> int:
        """Return the number of stored documents."""
        return len(self._storage)


__all__ = [
    "InMemoryBlobStorage",
]
