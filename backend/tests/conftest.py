"""Pytest fixtures for backend tests."""

import os
import pytest
from pathlib import Path
import tempfile
from PIL import Image


@pytest.fixture
def temp_image_jpg():
    """Create a temporary JPEG image for testing."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        # Create a simple 100x100 RGB image
        img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        img.save(f.name, format="JPEG")
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_image_png():
    """Create a temporary PNG image for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        img.save(f.name, format="PNG")
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_detections():
    """Return sample detection results matching the detector format."""
    return [
        {
            "class_id": 2,
            "class_name": "fall-armyworm-larva",
            "confidence": 0.92,
            "bbox": [50.0, 30.0, 120.0, 90.0],
            "recommendation": "Chemical | Apply targeted control promptly using label-approved products.",
            "recommendation_details": {
                "severity": "high",
                "management_tier": "chemical",
                "primary_action": "Apply targeted control promptly using label-approved products.",
                "secondary_action": "Rotate modes of action to reduce resistance pressure.",
            },
        },
        {
            "class_id": 4,
            "class_name": "healthy-maize",
            "confidence": 0.78,
            "bbox": [10.0, 10.0, 40.0, 50.0],
            "recommendation": "Monitoring | No treatment required.",
            "recommendation_details": {
                "severity": "none",
                "management_tier": "monitoring",
                "primary_action": "No treatment required.",
                "secondary_action": "Continue routine scouting.",
            },
        },
    ]
