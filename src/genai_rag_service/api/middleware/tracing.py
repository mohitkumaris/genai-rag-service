"""
Request tracing middleware.

Adds trace context to every request for distributed tracing.
"""

import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds tracing headers to requests and responses.
    
    Incoming headers:
    - X-Request-ID: Client-provided request ID (or generated if missing)
    - X-Trace-ID: Distributed trace ID (or generated if missing)
    - X-Span-ID: Parent span ID (optional)
    
    Response headers:
    - X-Request-ID: Echo back the request ID
    - X-Trace-ID: Echo back the trace ID
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Process the request with tracing context."""
        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Extract or generate trace ID
        trace_id = request.headers.get("X-Trace-ID")
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        # Extract parent span ID (optional)
        parent_span_id = request.headers.get("X-Span-ID")
        
        # Generate span ID for this request
        span_id = str(uuid.uuid4())
        
        # Store in request state for access in handlers
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        request.state.span_id = span_id
        request.state.parent_span_id = parent_span_id
        
        # Process request
        response = await call_next(request)
        
        # Add tracing headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Span-ID"] = span_id
        
        return response


__all__ = [
    "RequestTracingMiddleware",
]
