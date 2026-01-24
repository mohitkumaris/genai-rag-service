"""
Health check endpoints.
"""

from fastapi import APIRouter
from typing import Dict

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}

@router.get("/health/live")
async def liveness() -> Dict[str, str]:
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness() -> Dict[str, str]:
    return {"status": "ready"}
