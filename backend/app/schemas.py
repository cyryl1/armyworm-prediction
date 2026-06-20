"""
Pydantic response schemas for the pest detection API.

This module defines the data models used for API responses,
ensuring consistent and validated responses across all endpoints.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    """Single pest detection result.

    Attributes:
        class_id: YOLO class identifier
        class_name: Human-readable class name
        confidence: Detection confidence score (0-1)
        bbox: Bounding box coordinates
        recommendation: Compact recommendation string
        recommendation_details: Full structured management protocol
        gps_latitude: GPS latitude from image EXIF metadata
        gps_longitude: GPS longitude from image EXIF metadata
        detection_timestamp: UTC timestamp of detection
    """
    class_id: int = Field(..., description="YOLO class ID")
    class_name: str = Field(..., description="Class name (pest/disease type)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    bbox: List[float] = Field(..., description="Normalized bounding box [x1, y1, x2, y2] in 0-1 range (multiply by image width/height to get pixel coords)")
    recommendation: str = Field(..., description="Compact pest management recommendation")
    recommendation_details: Optional[dict] = Field(
        default=None,
        description="Full structured management protocol with cultural, biological, chemical controls and sources",
    )
    gps_latitude: Optional[float] = Field(default=None, description="GPS latitude from image metadata")
    gps_longitude: Optional[float] = Field(default=None, description="GPS longitude from image metadata")
    detection_timestamp: Optional[str] = Field(default=None, description="UTC timestamp of detection")


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
