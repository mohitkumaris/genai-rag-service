"""
RAG Tool invocation endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from genai_mcp_core import ToolRegistry, MCPContext
from app.dependencies import get_tool_registry

router = APIRouter()

@router.get("/tools")
async def list_tools(
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """List available tools."""
    return [t.model_dump() for t in registry.get_tools()]

@router.post("/tools/invoke/{tool_name}")
async def invoke_tool(
    tool_name: str,
    request: Request,
    registry: ToolRegistry = Depends(get_tool_registry)
):
    """Invoke a tool."""
    body = await request.json()
    args = body.get("arguments", {})
    
    # Basic context creation from headers (simplified)
    # In prod, extract from auth/tracing headers
    context = MCPContext(
        request_id=request.headers.get("x-request-id", "req-unknown"),
        user_id=request.headers.get("x-user-id", "user-unknown"),
        permissions=set(request.headers.get("x-permissions", "").split(",")) 
        if request.headers.get("x-permissions") else {"rag:ingest", "rag:search"} # Default permissive for dev
    )
    
    try:
        result = await registry.invoke(tool_name, args, context)
        return result.model_dump()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
