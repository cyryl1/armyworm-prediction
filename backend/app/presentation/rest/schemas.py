"""Pydantic response schemas for the pest detection API.

This module defines the data models used for API responses,
ensuring consistent and validated responses across all endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Detection schemas
# ---------------------------------------------------------------------------

class Detection(BaseModel):
    """Single pest detection result."""

    class_id: int = Field(..., description="YOLO class ID")
    class_name: str = Field(..., description="Class name (pest/disease type)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    bbox: List[float] = Field(
        ...,
        description="Normalized bounding box [x1, y1, x2, y2] in 0-1 range",
    )
    recommendation: str = Field(..., description="Compact pest management recommendation")
    recommendation_details: Optional[dict] = Field(
        default=None,
        description="Full structured management protocol",
    )
    gps_latitude: Optional[float] = Field(default=None, description="GPS latitude from image metadata")
    gps_longitude: Optional[float] = Field(default=None, description="GPS longitude from image metadata")
    detection_timestamp: Optional[str] = Field(default=None, description="UTC timestamp of detection")
    data_source: Optional[str] = Field(default="offline", description="Source of recommendation data")
    recommendation_last_updated: Optional[str] = Field(default=None, description="ISO timestamp of last fetch")


class DetectionResponse(BaseModel):
    """API response for a completed detection."""

    detections: List[Detection] = Field(default_factory=list, description="List of detections")
    annotated_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded JPEG of the image with bounding boxes drawn",
    )


class DetectionTaskSubmitted(BaseModel):
    """API response returned immediately after enqueuing a detection task."""

    task_id: str = Field(..., description="Celery task ID for polling")
    status: str = Field(default="pending", description="Initial task status")


class DetectionTaskStatus(BaseModel):
    """API response for polling a detection task's progress."""

    task_id: str = Field(..., description="Celery task ID")
    status: str = Field(..., description="pending | processing | completed | failed")
    result: Optional[DetectionResponse] = Field(
        default=None,
        description="Detection result (present only when status is 'completed')",
    )
    error: Optional[str] = Field(default=None, description="Error message if task failed")


# ---------------------------------------------------------------------------
# Information / status schemas
# ---------------------------------------------------------------------------

class ClassesResponse(BaseModel):
    """API response for available classes."""

    classes: dict = Field(..., description="Mapping of class ID to class name")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")


# ---------------------------------------------------------------------------
# Scouting schemas
# ---------------------------------------------------------------------------

class ScoutingSessionCreate(BaseModel):
    """Schema for creating a scouting session."""

    total_plants: int
    healthy_plants: int
    infested_plants: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    primary_recommendation: Optional[str] = None


class ScoutingSession(ScoutingSessionCreate):
    """Schema for a persisted scouting session."""

    id: int
    created_at: str

    model_config = ConfigDict(from_attributes=True)
