"""
genai-rag-service: Document Retrieval and Grounding for GenAI Platform

This service is the knowledge backbone of the GenAI platform. It provides:

- Document ingestion with configurable chunking
- Embedding generation (via external providers)
- Vector and hybrid search for retrieval
- Grounding context for LLM agents

Design Principles:
- Pure retrieval system - returns context, not answers
- Stateless at runtime
- Horizontally scalable
- Idempotent ingestion
- MCP-compatible tool interface

This service must NEVER:
- Generate natural language answers
- Contain agent logic
- Perform reasoning or planning
- Store conversation memory
"""

__version__ = "0.1.0"
