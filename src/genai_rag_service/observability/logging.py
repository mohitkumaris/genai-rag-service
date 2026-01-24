"""
Structured logging configuration.

Configures structlog for machine-readable JSON logging in production
and human-readable console output in development.
"""

import logging
import sys
from typing import Any, Literal

import structlog


def configure_logging(
    level: str = "INFO",
    format: Literal["json", "console"] = "json",
) -> None:
    """
    Configure structured logging for the application.
    
    In production (json format):
    - Outputs JSON for machine parsing
    - Includes timestamps, log levels, and structured fields
    
    In development (console format):
    - Outputs colored, human-readable logs
    - Easier to read during development
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format (json for production, console for dev)
    """
    # Set up the stdlib logging as a fallback
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Common processors for both formats
    common_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if format == "json":
        # Production: JSON output
        processors = common_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colored console output
        processors = common_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def bind_request_context(
    request_id: str,
    trace_id: str | None = None,
    **extra: Any,
) -> None:
    """
    Bind request context to the current context vars.
    
    This makes the request_id and trace_id available in all
    subsequent log calls within the same async context.
    
    Args:
        request_id: Unique request identifier
        trace_id: Distributed trace identifier
        **extra: Additional context to bind
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        trace_id=trace_id,
        **extra,
    )


def clear_request_context() -> None:
    """Clear the request context at the end of a request."""
    structlog.contextvars.clear_contextvars()


__all__ = [
    "configure_logging",
    "get_logger",
    "bind_request_context",
    "clear_request_context",
]
