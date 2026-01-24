"""
MCP tool invocation endpoints.

Provides HTTP endpoints for invoking MCP tools.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from genai_mcp_core import MCPContext, ToolNotFoundError, ToolSuccess


router = APIRouter()


class ToolInvocationRequest(BaseModel):
    """Request body for tool invocation."""
    
    input_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Input data for the tool",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional request ID (generated if not provided)",
    )
    trace_id: str | None = Field(
        default=None,
        description="Optional distributed trace ID",
    )
    permissions: list[str] = Field(
        default_factory=list,
        description="Permissions granted for this invocation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the context",
    )


class ToolInvocationResponse(BaseModel):
    """Response body for tool invocation."""
    
    success: bool
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def list_tools(request: Request) -> dict[str, Any]:
    """
    List all available tools.
    
    Returns tool definitions in a format suitable for LLM consumption.
    """
    registry = request.app.state.tool_registry
    tools = registry.list_tools()
    
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "tags": list(tool.tags),
                "required_permissions": list(tool.required_permissions),
                "input_schema": dict(tool.input_schema),
                "output_schema": dict(tool.output_schema),
            }
            for tool in tools
        ],
        "count": len(tools),
    }


@router.post("/invoke/{tool_name}")
async def invoke_tool(
    tool_name: str,
    body: ToolInvocationRequest,
    request: Request,
) -> ToolInvocationResponse:
    """
    Invoke an MCP tool by name.
    
    This endpoint provides HTTP access to MCP tools, translating
    HTTP requests into MCPContext and tool invocations.
    
    Args:
        tool_name: Name of the tool to invoke
        body: Tool invocation request with input data and context
        request: FastAPI request for accessing app state
        
    Returns:
        Tool execution result
    """
    registry = request.app.state.tool_registry
    
    # Get tracing context from middleware
    request_id = body.request_id or getattr(request.state, "request_id", None)
    trace_id = body.trace_id or getattr(request.state, "trace_id", None)
    
    # Create MCP context
    context = MCPContext.create(
        request_id=request_id,
        trace_id=trace_id,
        permissions=body.permissions,
        metadata=body.metadata,
    )
    
    try:
        # Invoke the tool
        result = await registry.invoke(tool_name, context, body.input_data)
        
        if isinstance(result, ToolSuccess):
            return ToolInvocationResponse(
                success=True,
                data=result.data,
                metadata=dict(result.metadata),
            )
        else:
            # ToolFailure
            return ToolInvocationResponse(
                success=False,
                error={
                    "error_code": result.error_code,
                    "message": result.message,
                    "details": dict(result.details),
                    "is_retryable": result.is_retryable,
                },
            )
            
    except ToolNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Tool not found: {tool_name}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tool invocation failed: {str(e)}",
        )


@router.get("/{tool_name}")
async def get_tool(tool_name: str, request: Request) -> dict[str, Any]:
    """
    Get details for a specific tool.
    
    Args:
        tool_name: Name of the tool
        request: FastAPI request for accessing app state
        
    Returns:
        Tool definition details
    """
    registry = request.app.state.tool_registry
    
    tool = registry.get_tool(tool_name)
    if tool is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tool not found: {tool_name}",
        )
    
    return {
        "name": tool.name,
        "description": tool.description,
        "version": tool.version,
        "category": tool.category,
        "tags": list(tool.tags),
        "required_permissions": list(tool.required_permissions),
        "input_schema": dict(tool.input_schema),
        "output_schema": dict(tool.output_schema),
    }


__all__ = [
    "router",
]
