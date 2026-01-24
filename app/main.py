"""
Application Entrypoint.
"""

from fastapi import FastAPI
from app.core.settings import settings
from app.core.logging import configure_logging
from app.api import health, rag

# Configure logging at startup
configure_logging()

def create_app() -> FastAPI:
    app = FastAPI(
        title="GenAI RAG Service",
        description="RAG service for GenAI platform",
        version="0.1.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url=None
    )
    
    app.include_router(health.router)
    app.include_router(rag.router)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.ENVIRONMENT == "development"
    )
