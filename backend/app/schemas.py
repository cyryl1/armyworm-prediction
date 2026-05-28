"""
Pydantic response schemas for the pest detection API.

This module defines the data models used for API responses,
ensuring consistent and validated responses across all endpoints.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field


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
    recommendation_details: Optional[Dict[str, str]] = Field(
        default=None,
        description="Structured recommendation details",
    )
    gps_latitude: Optional[float] = Field(default=None, description="GPS latitude from image metadata")
    gps_longitude: Optional[float] = Field(default=None, description="GPS longitude from image metadata")
    detection_timestamp: Optional[str] = Field(default=None, description="UTC timestamp of detection")
    image_url: Optional[str] = Field(default=None, description="Annotated image URL or path")
    farm_id: Optional[str] = Field(default=None, description="Associated farm identifier")
    user_id: Optional[str] = Field(default=None, description="Associated user identifier")


class DetectionResponse(BaseModel):
    """API response for detection request.
    
    Attributes:
        detections: List of detected pests and diseases
        annotated_image: Base64-encoded JPEG of the image with bounding boxes drawn
    """
    detections: List[Detection] = Field(default_factory=list, description="List of detections")
    annotated_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded JPEG of the image with bounding boxes drawn",
    )


class DetectionHistoryItem(BaseModel):
    """Stored detection entry returned from the history endpoint."""

    id: str = Field(..., description="Database record ID")
    class_id: int = Field(..., description="YOLO class ID")
    class_name: str = Field(..., description="Class name (pest/disease type)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    recommendation: str = Field(..., description="Pest management recommendation")
    recommendation_details: Optional[Dict[str, str]] = Field(default=None, description="Structured recommendation details")
    gps_latitude: Optional[float] = Field(default=None, description="GPS latitude from image metadata")
    gps_longitude: Optional[float] = Field(default=None, description="GPS longitude from image metadata")
    detection_timestamp: str = Field(..., description="UTC timestamp of detection")
    image_url: Optional[str] = Field(default=None, description="Annotated image URL or path")
    farm_id: Optional[str] = Field(default=None, description="Associated farm identifier")
    user_id: Optional[str] = Field(default=None, description="Associated user identifier")


class HistoryResponse(BaseModel):
    """API response for stored detection history."""

    records: List[DetectionHistoryItem] = Field(default_factory=list, description="Stored detection records")


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
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
