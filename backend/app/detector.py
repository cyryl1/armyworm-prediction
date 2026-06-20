"""
YOLOv8 model loading and pest detection inference logic.

This module handles model initialization, inference execution,
and detection result processing. The model is loaded once at startup
for optimal performance.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
import cv2
from ultralytics import YOLO

from app.recommendations import format_recommendation, get_recommendation_details


# Configure logging
logger = logging.getLogger(__name__)

# Class ID to class name mapping
CLASS_NAMES = {
    0: "fall-armyworm-egg",
    1: "fall-armyworm-frass",
    2: "fall-armyworm-larva",
    3: "fall-armyworm-larval-damage",
    4: "healthy-maize",
    5: "maize-streak-disease",
}

# Global model instance (singleton pattern)
_model: Optional[YOLO] = None


def _normalize_device(device: Optional[str]) -> str:
    """Normalize device input.

    Accepts None, numeric strings like '0' or integers and returns a
    device string suitable for PyTorch/Ultralytics (e.g. 'cuda:0' or 'cpu').
    """
    if device is None:
        return "cpu"

    d = str(device).strip().lower()
    # Numeric device => cuda:index if available
    if d.isdigit():
        idx = int(d)
        return f"cuda:{idx}" if torch.cuda.is_available() else "cpu"

    if d in ("cpu", "none"):
        return "cpu"

    if d in ("gpu", "cuda"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if d.startswith("cuda"):
        return d if torch.cuda.is_available() else "cpu"

    return "cpu"


def load_model(model_path: str = "model/best.pt", device: Optional[str] = None) -> YOLO:
    """Load YOLOv8 model from disk.
    
    The model is loaded once and cached globally to avoid reloading
    on each request, ensuring fast inference performance.
    
    Args:
        model_path: Path to the .pt model file
        device: Device to use ("0" for GPU, "cpu" for CPU)
        
    Returns:
        Loaded YOLO model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    global _model

    if _model is not None:
        logger.info("Model already loaded, using cached instance")
        return _model

    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device_str = _normalize_device(device)

    try:
        logger.info(f"Loading YOLOv8 model from {model_path} on device {device_str}")
        _model = YOLO(model_path)

        # try moving model to device; fall back to CPU on failure
        try:
            _model.to(device_str)
        except Exception as move_err:
            logger.warning(f"Could not move model to {device_str}: {move_err}. Falling back to cpu.")
            _model.to("cpu")

        logger.info("Model loaded successfully")
        return _model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def get_model() -> Optional[YOLO]:
    """Get the cached model instance.
    
    Returns:
        Cached YOLO model or None if not loaded
    """
    return _model


def get_class_name(class_id: int) -> str:
    """Get class name from class ID.
    
    Args:
        class_id: YOLO class identifier
        
    Returns:
        Class name string
    """
    return CLASS_NAMES.get(class_id, f"unknown_class_{class_id}")


def get_recommendation(class_name: str) -> str:
    """Get a compact pest management recommendation for a class."""
    return format_recommendation(class_name)


def get_recommendation_payload(class_name: str) -> Dict[str, str]:
    """Get structured pest management guidance for a class."""
    return get_recommendation_details(class_name)


def detect(image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
    """Run YOLO inference on an image and extract detection results.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score for detections (0-1)
        
    Returns:
        List of detection dictionaries containing:
            - class_id: int
            - class_name: str
            - confidence: float
            - bbox: [x1, y1, x2, y2]
            - recommendation: str
            
    Raises:
        RuntimeError: If model is not loaded or inference fails
        ValueError: If image cannot be read
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Verify image exists
    image_file = Path(image_path)
    if not image_file.exists():
        raise ValueError(f"Image file not found: {image_path}")
    
    # Verify image is readable
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image file: {image_path}")

    img_h, img_w = image.shape[:2]
    
    try:
        logger.info(f"Running inference on image: {image_path} (size: {img_w}x{img_h})")
        
        # Run YOLO inference
        results = _model.predict(image_path, conf=confidence_threshold, verbose=False)
        
        detections = []
        
        # Process each result
        for result in results:
            # Extract boxes information
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get absolute pixel coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get class information
                class_id = int(box.cls[0].item())
                class_name = get_class_name(class_id)
                
                # Get confidence score
                confidence = float(box.conf[0].item())
                
                # Get recommendation
                recommendation = get_recommendation(class_name)
                recommendation_details = get_recommendation_payload(class_name)

                # Normalize bbox to 0-1 range for the API response
                # (frontend multiplies by display dimensions to render)
                bbox_normalized = [
                    x1 / img_w,
                    y1 / img_h,
                    x2 / img_w,
                    y2 / img_h,
                ]

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox_normalized,
                    # Keep pixel coords for server-side annotation drawing
                    "bbox_pixel": [x1, y1, x2, y2],
                    "recommendation": recommendation,
                    "recommendation_details": recommendation_details,
                }
                
                detections.append(detection)
                logger.info(
                    f"Detection: {class_name} ({confidence:.2%}) "
                    f"at bbox_pixel [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                    f"normalized [{bbox_normalized[0]:.4f}, {bbox_normalized[1]:.4f}, "
                    f"{bbox_normalized[2]:.4f}, {bbox_normalized[3]:.4f}]"
                )
        
        logger.info(f"Found {len(detections)} detections")
        return detections
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}")


def detect_batch(image_paths: List[str], confidence_threshold: float = 0.5) -> List[List[Dict]]:
    """Run YOLO inference on a batch of images and extract detection results for each.
    
    Args:
        image_paths: List of paths to the image files
        confidence_threshold: Minimum confidence score for detections (0-1)
        
    Returns:
        List of lists of detection dictionaries, matching the input image_paths order.
        
    Raises:
        RuntimeError: If model is not loaded or inference fails
        ValueError: If any image cannot be read or is missing
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    if not image_paths:
        return []
        
    # Verify all images exist and are readable; cache dimensions
    image_dims = []
    for path in image_paths:
        image_file = Path(path)
        if not image_file.exists():
            raise ValueError(f"Image file not found: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image file: {path}")
        img_h, img_w = image.shape[:2]
        image_dims.append((img_w, img_h))
            
    try:
        logger.info(f"Running batch inference on {len(image_paths)} images")
        
        # Run YOLO inference in batch mode
        results = _model.predict(image_paths, conf=confidence_threshold, verbose=False)
        
        batch_detections = []
        
        # Process each image's results
        for idx, result in enumerate(results):
            detections = []
            boxes = result.boxes
            img_w, img_h = image_dims[idx]
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                class_name = get_class_name(class_id)
                confidence = float(box.conf[0].item())
                
                recommendation = get_recommendation(class_name)
                recommendation_details = get_recommendation_payload(class_name)

                bbox_normalized = [
                    x1 / img_w,
                    y1 / img_h,
                    x2 / img_w,
                    y2 / img_h,
                ]
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox_normalized,
                    "bbox_pixel": [x1, y1, x2, y2],
                    "recommendation": recommendation,
                    "recommendation_details": recommendation_details,
                }
                detections.append(detection)
                
            batch_detections.append(detections)
            logger.info(f"Image {image_paths[idx]}: Found {len(detections)} detections")
            
        return batch_detections
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise RuntimeError(f"Batch inference failed: {e}")


def get_all_classes() -> Dict[int, str]:
    """Get all supported class mappings.
    
    Returns:
        Dictionary mapping class IDs to class names
    """
    return CLASS_NAMES.copy()


def is_model_loaded() -> bool:
    """Check if model is currently loaded.
    
    Returns:
        True if model is loaded, False otherwise
    """
    return _model is not None
