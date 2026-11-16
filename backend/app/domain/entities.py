"""Domain entities — pure data structures with no framework dependencies.

These dataclasses represent the core concepts of the application.
They are used across all layers as the canonical data format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Detection domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    """Normalised bounding box (0-1 range relative to image dimensions)."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class BoundingBoxPixel:
    """Absolute pixel bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class DetectionResult:
    """A single pest/disease detection from the ML model."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    bbox_pixel: BoundingBoxPixel
    recommendation: str
    recommendation_details: Optional[Dict] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    detection_timestamp: Optional[str] = None
    data_source: Optional[str] = "offline"
    recommendation_last_updated: Optional[str] = None


# ---------------------------------------------------------------------------
# Scouting domain
# ---------------------------------------------------------------------------

@dataclass
class ScoutingSessionEntity:
    """A scouting session recorded by the user."""
    id: Optional[int] = None
    total_plants: int = 0
    healthy_plants: int = 0
    infested_plants: int = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    primary_recommendation: Optional[str] = None
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Class mapping (domain constant)
# ---------------------------------------------------------------------------

CLASS_NAMES: Dict[int, str] = {
    0: "fall-armyworm-egg",
    1: "fall-armyworm-frass",
    2: "fall-armyworm-larva",
    3: "fall-armyworm-larval-damage",
    4: "healthy-maize",
    5: "maize-streak-disease",
}
