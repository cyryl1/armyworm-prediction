"""Use case: Detect pests in an uploaded image.

Orchestrates file validation, temporary storage, GPS extraction,
and synchronous YOLO inference.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any

from app.infrastructure.ml.yolo_detector import YOLOPestDetector
from app.infrastructure.external.recommendations import get_recommendation_details
from app.infrastructure.utils import (
    validate_file_extension,
    validate_file_size,
    save_uploaded_file,
    extract_gps_from_exif,
    ALLOWED_EXTENSIONS,
    annotate_detections_to_base64,
    delete_file,
)

logger = logging.getLogger(__name__)


class DetectPestsUseCase:
    """Accepts an image upload, validates it, and runs synchronous detection."""

    def __init__(self, detector: YOLOPestDetector) -> None:
        self._detector = detector

    def execute(
        self,
        file_content: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Validate, save, run detection, and enrich with recommendations."""
        # Validate file extension
        if not validate_file_extension(filename):
            raise ValueError(
                f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Validate file size
        if not validate_file_size(len(file_content)):
            raise ValueError("File size exceeds maximum limit (10MB)")

        # Save to temp directory
        unique_filename, temp_file_path = save_uploaded_file(file_content, filename)
        logger.info(f"Saved temporary file for detection: {temp_file_path}")

        try:
            # Extract GPS from EXIF
            gps_latitude, gps_longitude = extract_gps_from_exif(temp_file_path)

            # Detect
            detections_list = self._detector.detect(str(temp_file_path))

            # Annotate
            annotated_image_b64 = None
            if detections_list:
                annotated_image_b64 = annotate_detections_to_base64(temp_file_path, detections_list)

            # Enrich
            for d in detections_list:
                live_rec = get_recommendation_details(
                    class_name=d["class_name"],
                    latitude=gps_latitude,
                    longitude=gps_longitude,
                )
                d["recommendation_details"] = live_rec
                d["data_source"] = live_rec.get("data_source", "offline")
                d["recommendation_last_updated"] = live_rec.get("recommendation_last_updated")
                d["gps_latitude"] = gps_latitude
                d["gps_longitude"] = gps_longitude
                d["detection_timestamp"] = datetime.now(timezone.utc).isoformat()
            
            return {
                "detections": detections_list,
                "annotated_image": annotated_image_b64,
            }
        finally:
            if temp_file_path.exists():
                delete_file(temp_file_path)
