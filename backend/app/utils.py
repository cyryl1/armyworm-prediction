"""
Utility functions for file handling and image processing.

This module provides helpers for managing uploaded files including
validation, temporary storage, and cleanup operations.
"""

import os
import base64
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ExifTags

# Configure logging
logger = logging.getLogger(__name__)

# Allowed image file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# Temporary directory for storing uploaded files
TEMP_DIR = Path("./temp")


def init_temp_directory() -> None:
    """Initialize temporary directory for file uploads.
    
    Creates temp directory if it doesn't exist.
    """
    TEMP_DIR.mkdir(exist_ok=True)
    logger.info(f"Initialized temp directory: {TEMP_DIR}")


def validate_file_extension(filename: str) -> bool:
    """Validate if file has allowed image extension.
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    file_ext = Path(filename).suffix.lower()
    is_valid = file_ext in ALLOWED_EXTENSIONS
    
    if not is_valid:
        logger.warning(f"Invalid file extension: {file_ext}")
    
    return is_valid


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """Validate if file size is within limits.
    
    Args:
        file_size: Size of file in bytes
        max_size_mb: Maximum allowed size in megabytes
        
    Returns:
        True if file size is acceptable, False otherwise
    """
    max_bytes = max_size_mb * 1024 * 1024
    is_valid = file_size <= max_bytes
    
    if not is_valid:
        logger.warning(f"File size {file_size} bytes exceeds limit {max_bytes}")
    
    return is_valid


def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename to avoid collisions.
    
    Args:
        original_filename: Original name of the uploaded file
        
    Returns:
        Unique filename with UUID prefix
    """
    file_ext = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())[:8]
    unique_filename = f"{unique_id}_{original_filename}"
    
    return unique_filename


def get_temp_file_path(filename: str) -> Path:
    """Get full path for temporary file.
    
    Args:
        filename: Name of the temporary file
        
    Returns:
        Full path to temporary file
    """
    return TEMP_DIR / filename


def save_uploaded_file(file_content: bytes, original_filename: str) -> Tuple[str, Path]:
    """Save uploaded file to temporary directory.
    
    Args:
        file_content: Raw bytes of the uploaded file
        original_filename: Original filename from upload
        
    Returns:
        Tuple of (unique_filename, full_file_path)
        
    Raises:
        ValueError: If file validation fails
    """
    # Validate extension
    if not validate_file_extension(original_filename):
        raise ValueError(f"Invalid file type: {original_filename}")
    
    # Validate size
    if not validate_file_size(len(file_content)):
        raise ValueError(f"File size exceeds maximum limit")
    
    # Generate unique filename
    unique_filename = generate_unique_filename(original_filename)
    file_path = get_temp_file_path(unique_filename)
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved uploaded file: {file_path}")
        return unique_filename, file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise


def delete_file(file_path: Path) -> bool:
    """Delete temporary file after processing.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if file_path.exists():
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
            return True
        else:
            logger.warning(f"File not found for deletion: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


def _convert_to_degrees(value) -> float:
    """Convert EXIF GPS coordinates to decimal degrees."""
    degrees = value[0][0] / value[0][1]
    minutes = value[1][0] / value[1][1]
    seconds = value[2][0] / value[2][1]
    return degrees + (minutes / 60.0) + (seconds / 3600.0)


def extract_gps_from_exif(image_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Extract GPS coordinates from image EXIF metadata if present."""
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            if not exif:
                return None, None

            gps_tag = None
            for tag_id, tag_name in ExifTags.TAGS.items():
                if tag_name == "GPSInfo":
                    gps_tag = tag_id
                    break

            if gps_tag is None or gps_tag not in exif:
                return None, None

            gps_data = exif.get(gps_tag)
            if not gps_data:
                return None, None

            gps_info = {}
            for key, value in gps_data.items():
                gps_info[ExifTags.GPSTAGS.get(key, key)] = value

            lat = gps_info.get("GPSLatitude")
            lat_ref = gps_info.get("GPSLatitudeRef")
            lon = gps_info.get("GPSLongitude")
            lon_ref = gps_info.get("GPSLongitudeRef")

            if not lat or not lon or not lat_ref or not lon_ref:
                return None, None

            latitude = _convert_to_degrees(lat)
            longitude = _convert_to_degrees(lon)

            if str(lat_ref).upper() == "S":
                latitude = -latitude
            if str(lon_ref).upper() == "W":
                longitude = -longitude

            return latitude, longitude
    except Exception as e:
        logger.warning(f"Failed to extract EXIF GPS data from {image_path}: {e}")
        return None, None


_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "fall-armyworm-egg": (0, 180, 255),       # orange (BGR)
    "fall-armyworm-frass": (0, 140, 220),      # dark orange
    "fall-armyworm-larva": (60, 60, 255),      # red
    "fall-armyworm-larval-damage": (0, 165, 255),  # orange
    "healthy-maize": (80, 220, 80),            # green
    "maize-streak-disease": (200, 80, 180),    # purple
}

_DEFAULT_COLOR = (0, 200, 200)  # yellow fallback


def _draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on an image array (in-place).

    Uses class-specific colours. Boxes that cover >90% of the image area
    are treated as whole-image classifications — a label banner is drawn
    at the top instead of a full rectangle, so the image stays clean.

    Args:
        image: BGR image as a numpy array.
        detections: List of detection dicts with 'bbox', 'class_name', 'confidence'.

    Returns:
        The same image array with annotations drawn.
    """
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    banner_offset = 0  # stacks banners if multiple full-image detections

    for detection in detections:
        bbox = detection.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

        # Clamp to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        class_name = detection.get("class_name", "object")
        confidence = detection.get("confidence", 0)
        color = _CLASS_COLORS.get(class_name, _DEFAULT_COLOR)
        label = f"{class_name.replace('-', ' ').title()}  {confidence:.0%}"

        box_area = abs(x2 - x1) * abs(y2 - y1)

        if img_area > 0 and box_area / img_area > 0.90:
            # Full-image classification → draw a label banner at the top
            font_scale = max(0.5, min(img_w / 800, 1.2))
            thickness = max(1, int(font_scale * 2))
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            bar_h = th + baseline + 20
            y_pos = banner_offset

            # Semi-transparent banner background
            overlay = image.copy()
            cv2.rectangle(overlay, (0, y_pos), (img_w, y_pos + bar_h), color, -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # White text on banner
            cv2.putText(
                image, label,
                (10, y_pos + th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA,
            )
            banner_offset += bar_h + 4
        else:
            # Normal tight bounding box around the detection
            line_thickness = max(2, int(min(img_w, img_h) / 300))

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

            # Draw corner accents for a cleaner look
            corner_len = max(10, int(min(abs(x2 - x1), abs(y2 - y1)) * 0.15))
            ct = line_thickness + 1
            # Top-left
            cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, ct)
            cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, ct)
            # Top-right
            cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, ct)
            cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, ct)
            # Bottom-left
            cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, ct)
            cv2.line(image, (x1, y2), (x1, y2 - corner_len), color, ct)
            # Bottom-right
            cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, ct)
            cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, ct)

            # Label with filled background
            font_scale = max(0.4, min(img_w / 1000, 0.8))
            thickness = max(1, int(font_scale * 2))
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y = max(y1 - 6, th + 10)
            label_x = x1

            # Background rectangle for label
            overlay = image.copy()
            cv2.rectangle(overlay, (label_x, label_y - th - 6), (label_x + tw + 10, label_y + 4), color, -1)
            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

            # White text
            cv2.putText(
                image, label,
                (label_x + 5, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA,
            )

    return image


def annotate_detections(image_path: Path, detections: list, output_filename: Optional[str] = None) -> Path:
    """Draw detection boxes on an image and save the annotated copy."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image for annotation: {image_path}")

    _draw_detections(image, detections)

    if output_filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        output_filename = f"{image_path.stem}_annotated_{timestamp}{image_path.suffix}"

    output_path = get_temp_file_path(output_filename)
    cv2.imwrite(str(output_path), image)
    logger.info(f"Saved annotated image: {output_path}")
    return output_path


def annotate_detections_to_base64(
    image_path: Path,
    detections: list,
    jpeg_quality: int = 70,
) -> str:
    """Draw detection boxes on an image and return as a base64-encoded JPEG.

    Args:
        image_path: Path to the source image file.
        detections: List of detection dicts with 'bbox', 'class_name', 'confidence'.
        jpeg_quality: JPEG compression quality (1-100).

    Returns:
        Base64-encoded JPEG string of the annotated image.

    Raises:
        ValueError: If the image cannot be read.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image for annotation: {image_path}")

    _draw_detections(image, detections)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, buffer = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise RuntimeError("Failed to encode annotated image to JPEG")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    logger.info(f"Encoded annotated image to base64 ({len(b64_str)} chars)")
    return b64_str


def annotate_image_array_to_base64(
    image: np.ndarray,
    detections: list,
    jpeg_quality: int = 70,
) -> str:
    """Draw detection boxes on an in-memory image array and return as base64 JPEG.

    Args:
        image: BGR image as a numpy array.
        detections: List of detection dicts with 'bbox', 'class_name', 'confidence'.
        jpeg_quality: JPEG compression quality (1-100).

    Returns:
        Base64-encoded JPEG string of the annotated image.

    Raises:
        ValueError: If the image is None or empty.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot annotate empty or None image")

    _draw_detections(image, detections)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, buffer = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise RuntimeError("Failed to encode annotated image to JPEG")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    logger.info(f"Encoded annotated frame to base64 ({len(b64_str)} chars)")
    return b64_str


def cleanup_temp_directory() -> None:
    """Clean up temporary directory by removing all files.
    
    Useful for cleanup on shutdown or maintenance.
    """
    try:
        if TEMP_DIR.exists():
            for file in TEMP_DIR.glob("*"):
                if file.is_file():
                    os.remove(file)
            logger.info("Cleaned up temp directory")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {e}")
