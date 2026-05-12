"""
Pydantic response schemas for the pest detection API.

This module defines the data models used for API responses,
ensuring consistent and validated responses across all endpoints.
"""

from typing import List
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected object.
    
    Attributes:
        x1: Left x-coordinate
        y1: Top y-coordinate
        x2: Right x-coordinate
        y2: Bottom y-coordinate
    """
    x1: float = Field(..., description="Left x-coordinate")
    y1: float = Field(..., description="Top y-coordinate")
    x2: float = Field(..., description="Right x-coordinate")
    y2: float = Field(..., description="Bottom y-coordinate")


class Detection(BaseModel):
    """Single pest detection result.
    
    Attributes:
        class_id: YOLO class identifier
        class_name: Human-readable class name
        confidence: Detection confidence score (0-1)
        bbox: Bounding box coordinates
        recommendation: Pest management recommendation
    """
    class_id: int = Field(..., description="YOLO class ID")
    class_name: str = Field(..., description="Class name (pest/disease type)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    recommendation: str = Field(..., description="Pest management recommendation")


class DetectionResponse(BaseModel):
    """API response for detection request.
    
    Attributes:
        detections: List of detected pests and diseases
    """
    detections: List[Detection] = Field(default_factory=list, description="List of detections")


class ClassesResponse(BaseModel):
    """API response for available classes.
    
    Attributes:
        classes: Mapping of class ID to class name
    """
    classes: dict = Field(..., description="Mapping of class ID to class name")


class HealthResponse(BaseModel):
    """Health check response.
    
    Attributes:
        status: Health status
        model_loaded: Whether the YOLO model is loaded
    """
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
