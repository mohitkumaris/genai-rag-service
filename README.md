# genai-rag-service

**The knowledge backbone of the GenAI platform.**

`genai-rag-service` is a production-grade Retrieval-Augmented Generation (RAG) service that provides document ingestion, chunking, embedding, and vector/hybrid search capabilities for LLM agents.

## What This Service IS

- ✅ A **pure retrieval system** — returns grounding context, not answers
- ✅ A **document ingestion pipeline** — chunk, embed, and index documents
- ✅ A **semantic search engine** — vector and hybrid (vector + keyword) search
- ✅ **MCP-compatible** — exposed as tools for LLM agents
- ✅ **Horizontally scalable** — stateless at runtime

## What This Service is NOT

- ❌ Not an answer generation service
- ❌ Not an agent or reasoning system
- ❌ Not a conversation memory store
- ❌ Not a user-facing API

---

## Quick Start

### Installation

```bash
# Clone the repository
cd genai-rag-service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Run the Development Server

```bash
uvicorn app.main:app, --host 0.0.0.0 --port 8000 --reload
```

### Health Check

```bash
curl http://localhost:8000/health
```

### List Available Tools

```bash
curl http://localhost:8000/tools
```

---

## Architecture

This service uses a **flat layout** structure to ensure separation of concerns:

```
genai-rag-service/
├── app/            # FastAPI delivery layer (API, core routes, config)
├── rag/            # Pure domain logic (Schemas, Ingestion, Retrieval)
├── adapters/       # Infrastructure adapters (Memory, Azure AI Search, etc.)
├── mcp_tools/      # MCP tool definitions and handlers
├── tests/          # Test suite (Unit & Integration)
├── Dockerfile      # Container definition
├── pyproject.toml  # Project configuration
└── .env.example    # Environment variable template
```

### Design Principles

1. **Hexagonal Architecture** — Business logic isolated from infrastructure
2. **Ports & Adapters** — External dependencies abstracted behind interfaces in `rag/schemas.py`
3. **Deterministic Operations** — Same input always produces same output (chunk IDs, etc.)
4. **Idempotent Ingestion** — Re-ingesting same document is a no-op
5. **Separation of Concerns** — API layer (`app/`) is separate from Domain (`rag/`)

---

## MCP Tools

### rag_ingest

Ingest documents into the RAG system.

```python
# Input
{
    "documents": [
        {
            "uri": "doc://example/intro",
            "content": "Document text content...",
            "metadata": {"category": "tutorials"}
        }
    ],
    "chunking": {
        "strategy": "fixed_size",
        "chunk_size": 512,
        "chunk_overlap": 50
    }
}

# Output
{
    "ingested_count": 1,
    "skipped_count": 0,
    "chunk_count": 5,
    "document_ids": ["abc123..."],
    "errors": []
}
```

Required permission: `rag:ingest`

### rag_search

Search for relevant document chunks.

```python
# Input
{
    "query": "What is machine learning?",
    "top_k": 10,
    "search_type": "vector",  # or "hybrid"
    "similarity_threshold": 0.5,
    "filters": {"category": "tutorials"}
}

# Output
{
    "results": [
        {
            "chunk_id": "xyz789...",
            "content": "Machine learning is...",
            "score": 0.92,
            "source_uri": "doc://example/intro",
            "metadata": {...}
        }
    ],
    "count": 1,
    "latency_ms": 45.2
}
```

Required permission: `rag:search`

---

## Configuration

All settings can be configured via environment variables with the `RAG_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ENVIRONMENT` | `development` | Environment (development, staging, production) |
| `RAG_LOG_LEVEL` | `INFO` | Log level |
| `RAG_EMBEDDING_MODEL_ID` | `text-embedding-ada-002` | Embedding model identifier |
| `RAG_DEFAULT_CHUNK_SIZE` | `512` | Default chunk size in tokens |
| `RAG_VECTOR_STORE_TYPE` | `memory` | Vector store type (memory, azure) |
| `RAG_STORAGE_TYPE` | `memory` | Blob storage type (memory, azure) |

Create a `.env` file for local development:

```bash
RAG_ENVIRONMENT=development
RAG_LOG_LEVEL=DEBUG
RAG_LOG_FORMAT=console
```

---

## Development

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v
```

### Type Checking

```bash
mypy rag adapters mcp_tools app
```

### Linting

```bash
ruff check .
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health summary |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/tools` | List available tools |
| `POST` | `/tools/invoke/{name}` | Invoke a tool |

---

## Integration with genai-mcp-core

This service depends on `genai-mcp-core` for:

- `MCPContext` — Request envelope with tracing and permissions
- `ToolDefinition` — Tool contracts
- `ToolRegistry` — Tool discovery and invocation
- `ToolResult` — Explicit Success/Failure return types

---

## License

MIT
