"""
Health check endpoints.

Provides liveness and readiness probes for orchestration platforms.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/live")
async def liveness() -> dict[str, Any]:
    """
    Liveness probe.
    
    Returns 200 if the service is running.
    Used by orchestrators to detect crashed containers.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
async def readiness(request: Request) -> dict[str, Any]:
    """
    Readiness probe.
    
    Returns 200 if the service is ready to handle requests.
    Checks that all dependencies are initialized.
    """
    # Check that core components are available
    checks = {
        "vector_store": hasattr(request.app.state, "vector_store"),
        "embedding_provider": hasattr(request.app.state, "embedding_provider"),
        "ingestion_service": hasattr(request.app.state, "ingestion_service"),
        "retrieval_service": hasattr(request.app.state, "retrieval_service"),
        "tool_registry": hasattr(request.app.state, "tool_registry"),
    }
    
    all_ready = all(checks.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


@router.get("")
async def health_summary(request: Request) -> dict[str, Any]:
    """
    Health summary.
    
    Returns overall health status and basic metrics.
    """
    settings = request.app.state.settings
    vector_store = request.app.state.vector_store
    
    # Get chunk count if available
    chunk_count = 0
    try:
        chunk_count = await vector_store.get_chunk_count()
    except Exception:
        pass
    
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": "0.1.0",
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "total_chunks": chunk_count,
        },
    }


__all__ = [
    "router",
]
