"""FastAPI application for pest detection using YOLOv8.

This module defines the main FastAPI application, wires together the
clean architecture layers, and handles application lifecycle events.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.infrastructure.db import database as db_module
from app.infrastructure.db.models import ScoutingSessionModel  # noqa: F401 — ensures table creation
from app.infrastructure.external.api_client import api_client
from app.infrastructure.utils import init_temp_directory, cleanup_temp_directory
from app.presentation.rest.routes import router as api_router, _detector
from app.presentation.websocket.stream import ws_router, get_stream_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables from backend/.env during local development.
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Pest Detection API",
    description="YOLOv8-based pest detection system for fall armyworms and maize diseases",
    version="2.0.0",
)

# Add CORS Middleware to support web frontend queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Include modular routes
app.include_router(api_router)
app.include_router(ws_router)


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup.

    - Initializes temporary directory for file uploads
    - Loads YOLOv8 model into memory (for WebSocket stream; Celery workers load their own)
    - Initializes database tables
    - Initializes external API client
    """
    logger.info("Starting up Pest Detection API (v2 — Clean Architecture)...")

    # Initialize temp directory
    init_temp_directory()

    # Initialize API client
    api_client.init_client()

    # Initialize database tables
    logger.info("Initializing database tables...")
    db_module.Base.metadata.create_all(bind=db_module.engine)

    # Load model for the WebSocket live stream (sync inference in-process)
    # The Celery worker loads its own model instance separately.
    try:
        _detector.load_model("model/best.pt", device=None)
        # The stream detector is the same class but a different instance;
        # load it too so the WebSocket handler works.
        stream_det = get_stream_detector()
        stream_det.load_model("model/best.pt", device=None)
        logger.info("Model loaded successfully at startup (for live stream)")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
        logger.warning("Continuing without loaded model; /health will report model_loaded=False")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down Pest Detection API...")
    cleanup_temp_directory()
    await api_client.close()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
