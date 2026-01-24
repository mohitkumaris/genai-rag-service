"""
In-memory blob storage adapter for testing.
"""

from rag.schemas import BlobStoragePort

class InMemoryBlobStorage(BlobStoragePort):
    """Simple dictionary-based storage."""
    
    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}

    async def get_document(self, uri: str) -> bytes:
        if uri not in self._storage:
            raise FileNotFoundError(f"Document not found: {uri}")
        return self._storage[uri]

    async def put_document(self, uri: str, content: bytes) -> None:
        self._storage[uri] = content

    async def exists(self, uri: str) -> bool:
        return uri in self._storage

    async def list_documents(self, prefix: str = "") -> list[str]:
        return [k for k in self._storage.keys() if k.startswith(prefix)]

    async def delete(self, uri: str) -> None:
        if uri in self._storage:
            del self._storage[uri]
