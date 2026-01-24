"""
Structured logging middleware.

Logs request/response information in a structured format.
"""

import time
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from genai_rag_service.observability.logging import (
    bind_request_context,
    clear_request_context,
)


logger = structlog.get_logger(__name__)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs request/response information.
    
    Logs:
    - Request start with method, path, and tracing context
    - Request completion with status code and latency
    - Request errors with exception details
    
    Uses structlog for structured, machine-readable output.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Process the request with structured logging."""
        # Get tracing context (set by RequestTracingMiddleware)
        request_id = getattr(request.state, "request_id", "unknown")
        trace_id = getattr(request.state, "trace_id", "unknown")
        
        # Bind context for all log calls in this request
        bind_request_context(
            request_id=request_id,
            trace_id=trace_id,
        )
        
        start_time = time.monotonic()
        
        # Log request start
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
            client_ip=request.client.host if request.client else None,
        )
        
        try:
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.monotonic() - start_time) * 1000
            
            # Log request completion
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=round(latency_ms, 2),
            )
            
            # Add latency header
            response.headers["X-Response-Time-Ms"] = str(round(latency_ms, 2))
            
            return response
            
        except Exception as e:
            # Calculate latency
            latency_ms = (time.monotonic() - start_time) * 1000
            
            # Log error
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                latency_ms=round(latency_ms, 2),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            # Clear context
            clear_request_context()


__all__ = [
    "StructuredLoggingMiddleware",
]
