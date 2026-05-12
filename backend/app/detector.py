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

# Pest management recommendations
RECOMMENDATIONS = {
    "fall-armyworm-egg": "Monitor egg clusters, consider removal or targeted spraying",
    "fall-armyworm-frass": "Inspect surrounding leaves for larvae",
    "fall-armyworm-larva": "Apply neem-based pesticide immediately",
    "fall-armyworm-larval-damage": "Monitor crop and apply targeted control",
    "healthy-maize": "No action needed",
    "maize-streak-disease": "Use resistant maize varieties, control vector (leafhopper)",
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
    """Get pest management recommendation for a class.
    
    Args:
        class_name: Name of the detected class
        
    Returns:
        Recommendation string
    """
    return RECOMMENDATIONS.get(class_name, "No specific recommendation available")


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
    
    try:
        logger.info(f"Running inference on image: {image_path}")
        
        # Run YOLO inference
        results = _model.predict(image_path, conf=confidence_threshold, verbose=False)
        
        detections = []
        
        # Process each result
        for result in results:
            # Extract boxes information
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get class information
                class_id = int(box.cls[0].item())
                class_name = get_class_name(class_id)
                
                # Get confidence score
                confidence = float(box.conf[0].item())
                
                # Get recommendation
                recommendation = get_recommendation(class_name)
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "recommendation": recommendation,
                }
                
                detections.append(detection)
                logger.info(
                    f"Detection: {class_name} ({confidence:.2%}) "
                    f"at bbox [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                )
        
        logger.info(f"Found {len(detections)} detections")
        return detections
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}")


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
