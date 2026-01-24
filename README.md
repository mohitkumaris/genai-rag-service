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
uvicorn genai_rag_service.api.app:get_application --reload
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

```
genai-rag-service/
├── src/genai_rag_service/
│   ├── domain/           # Pure domain models (Document, Chunk, Embedding, etc.)
│   ├── ports/            # Abstract interfaces for adapters
│   ├── adapters/         # Concrete implementations (in-memory, Azure, etc.)
│   ├── chunking/         # Document chunking strategies
│   ├── services/         # Business logic (Ingestion, Retrieval)
│   ├── tools/            # MCP tool definitions and handlers
│   ├── api/              # FastAPI application
│   ├── config/           # Settings and configuration
│   └── observability/    # Logging and metrics
└── tests/
    ├── unit/             # Unit tests
    └── integration/      # Integration tests
```

### Design Principles

1. **Hexagonal Architecture** — Business logic isolated from infrastructure
2. **Ports & Adapters** — External dependencies abstracted behind interfaces
3. **Deterministic Operations** — Same input always produces same output
4. **Idempotent Ingestion** — Re-ingesting same document is a no-op
5. **Explicit Results** — ToolSuccess/ToolFailure over exceptions

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
            "version": "1.0.0",
            "metadata": {"category": "tutorials"}
        }
    ],
    "chunking_config": {
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
            "document_id": "abc123...",
            "content": "Machine learning is...",
            "score": 0.92,
            "match_type": "vector",
            "metadata": {...}
        }
    ],
    "total_results": 1,
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
| `RAG_EMBEDDING_DIMENSION` | `1536` | Embedding vector dimension |
| `RAG_DEFAULT_CHUNK_SIZE` | `512` | Default chunk size in tokens |
| `RAG_VECTOR_STORE_TYPE` | `memory` | Vector store type (memory, azure_search) |
| `RAG_EMBEDDING_PROVIDER_TYPE` | `mock` | Embedding provider (mock, openai) |

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

# With coverage
pytest tests/ -v --cov=src/genai_rag_service
```

### Type Checking

```bash
mypy src/genai_rag_service --strict
```

### Linting

```bash
ruff check src/
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health summary |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/tools` | List available tools |
| `GET` | `/tools/{name}` | Get tool details |
| `POST` | `/tools/invoke/{name}` | Invoke a tool |

---

## Integration with genai-mcp-core

This service depends on `genai-mcp-core` for:

- `MCPContext` — Request envelope with tracing and permissions
- `ToolDefinition` — Tool contracts
- `ToolRegistry` — Tool discovery and invocation
- `ToolSuccess/ToolFailure` — Explicit result types

```python
from genai_mcp_core import MCPContext, ToolRegistry

# Create context with permissions
context = MCPContext.create(
    permissions=["rag:ingest", "rag:search"],
    metadata={"tenant_id": "acme-corp"},
)

# Invoke tool through registry
result = await registry.invoke("rag_search", context, {"query": "..."})
```

---

## License

MIT
