"""
FastAPI application for pest detection using YOLOv8.

This module defines the main FastAPI application with endpoints for
pest detection, health checks, and class information retrieval.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app import detector, utils
from app import history_store
from app.routes import router as api_router

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
    version="1.0.0",
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

    # Add API key security schema for x-api-key header
    components = openapi_schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
    }

    # Mark protected endpoints with the security requirement so Swagger shows them
    paths = openapi_schema.get("paths", {})
    for p in ["/detect", "/history"]:
        if p in paths:
            for method in paths[p].keys():
                paths[p][method]["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Include modular routes
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup.
    
    - Initializes temporary directory for file uploads
    - Loads YOLOv8 model into memory
    """
    logger.info("Starting up Pest Detection API...")
    
    # Initialize temp directory
    utils.init_temp_directory()
    try:
        await history_store.init_db()
        logger.info("MongoDB initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB at startup: {e}")
        logger.warning("Continuing without history persistence; history endpoints may be unavailable")
    
    # Load model at startup (do not crash the app if model fails to load)
    try:
        # Let detector decide device (defaults to CPU if CUDA unavailable)
        detector.load_model("model/best.pt", device=None)
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
        logger.warning("Continuing without loaded model; /health will report model_loaded=False")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown.
    
    - Removes temporary files
    - Performs cleanup operations
    """
    logger.info("Shutting down Pest Detection API...")
    utils.cleanup_temp_directory()





@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors.
    
    Args:
        request: The request that caused the error
        exc: The exception that was raised
        
    Returns:
        JSON error response with 500 status
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# All route handlers are implemented in `app.routes` and registered above.


if __name__ == "__main__":
    import uvicorn
    
    # Run development server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
