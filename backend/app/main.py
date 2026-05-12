"""
FastAPI application for pest detection using YOLOv8.

This module defines the main FastAPI application with endpoints for
pest detection, health checks, and class information retrieval.
"""

import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app import detector, utils
from app.schemas import DetectionResponse, Detection, ClassesResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pest Detection API",
    description="YOLOv8-based pest detection system for fall armyworms and maize diseases",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup.
    
    - Initializes temporary directory for file uploads
    - Loads YOLOv8 model into memory
    """
    logger.info("Starting up Pest Detection API...")
    
    # Initialize temp directory
    utils.init_temp_directory()
    
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


@app.get("/", tags=["Status"])
async def root():
    """Root endpoint - confirms API is running.
    
    Returns:
        Simple status message
    """
    logger.info("GET / - API status check")
    return {"message": "Pest Detection API running"}


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    """Health check endpoint for monitoring.
    
    Returns:
        Health status and model load state
    """
    logger.info("GET /health - Health check")
    return HealthResponse(
        status="healthy",
        model_loaded=detector.is_model_loaded(),
    )


@app.get("/classes", response_model=ClassesResponse, tags=["Information"])
async def get_classes():
    """Get all supported pest and disease classes.
    
    Returns:
        Mapping of class IDs to class names
    """
    logger.info("GET /classes - Retrieving supported classes")
    classes = detector.get_all_classes()
    
    # Convert int keys to strings for JSON serialization
    classes_str = {str(k): v for k, v in classes.items()}
    
    return ClassesResponse(classes=classes_str)


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_pests(file: UploadFile = File(...)):
    """Detect pests in uploaded image.
    
    This endpoint accepts an image file, runs YOLOv8 inference,
    and returns detected pests with confidence scores, bounding boxes,
    and management recommendations.
    
    Args:
        file: Image file upload (multipart/form-data)
        
    Returns:
        DetectionResponse with list of detections
        
    Raises:
        400: Bad Request - No file uploaded or empty file
        415: Unsupported Media Type - Invalid image format
        500: Internal Server Error - Model inference failure
    """
    logger.info(f"POST /detect - Processing image: {file.filename}")
    
    # Validate file was uploaded
    if not file:
        logger.warning("No file uploaded")
        raise HTTPException(
            status_code=400,
            detail="No file uploaded",
        )
    
    # Validate file extension
    if not utils.validate_file_extension(file.filename):
        logger.warning(f"Invalid file extension: {file.filename}")
        raise HTTPException(
            status_code=415,
            detail=f"Invalid file type. Allowed types: {', '.join(utils.ALLOWED_EXTENSIONS)}",
        )
    
    temp_file_path = None
    
    try:
        # Read uploaded file
        file_content = await file.read()
        
        # Validate file size
        if not utils.validate_file_size(len(file_content)):
            logger.warning(f"File size too large: {len(file_content)} bytes")
            raise HTTPException(
                status_code=400,
                detail="File size exceeds maximum limit (10MB)",
            )
        
        # Save uploaded file temporarily
        unique_filename, temp_file_path = utils.save_uploaded_file(
            file_content, file.filename
        )
        logger.info(f"Saved temporary file: {temp_file_path}")
        
        # Run inference
        detections_list = detector.detect(str(temp_file_path))
        
        # Convert to response model
        detections = [
            Detection(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d["bbox"],
                recommendation=d["recommendation"],
            )
            for d in detections_list
        ]
        
        logger.info(f"Returning {len(detections)} detections")
        return DetectionResponse(detections=detections)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Handle file validation errors
        logger.error(f"File validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except RuntimeError as e:
        # Handle inference errors
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error processing image",
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )
    finally:
        # Always clean up temporary file
        if temp_file_path:
            utils.delete_file(temp_file_path)


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


if __name__ == "__main__":
    import uvicorn
    
    # Run development server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
