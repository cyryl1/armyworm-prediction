"""Concrete adapter: YOLOv8 pest detector.

Implements ``PestDetectorInterface`` using the Ultralytics YOLO library.
The model is loaded once and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
from ultralytics import YOLO

from app.domain.entities import CLASS_NAMES
from app.domain.interfaces.ml_model import PestDetectorInterface
from app.infrastructure.external.recommendations import (
    format_recommendation,
    get_recommendation_details,
)

logger = logging.getLogger(__name__)


def _normalize_device(device: Optional[str]) -> str:
    """Normalize device input to a PyTorch device string."""
    if device is None:
        return "cpu"

    d = str(device).strip().lower()
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


class YOLOPestDetector(PestDetectorInterface):
    """Adapter wrapping the Ultralytics YOLO model."""

    def __init__(self) -> None:
        self._model: Optional[YOLO] = None

    # ---- interface implementation ----

    def load_model(self, model_path: str, device: Optional[str] = None) -> None:
        if self._model is not None:
            logger.info("Model already loaded, using cached instance")
            return

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        device_str = _normalize_device(device)

        try:
            logger.info(f"Loading YOLOv8 model from {model_path} on device {device_str}")
            self._model = YOLO(model_path)
            try:
                self._model.to(device_str)
            except Exception as move_err:
                logger.warning(f"Could not move model to {device_str}: {move_err}. Falling back to cpu.")
                self._model.to("cpu")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image_file = Path(image_path)
        if not image_file.exists():
            raise ValueError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image file: {image_path}")

        img_h, img_w = image.shape[:2]

        try:
            logger.info(f"Running inference on image: {image_path} (size: {img_w}x{img_h})")
            results = self._model.predict(image_path, conf=confidence_threshold, verbose=False)

            detections: List[Dict] = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    class_name = CLASS_NAMES.get(class_id, f"unknown_class_{class_id}")
                    confidence = float(box.conf[0].item())

                    recommendation = format_recommendation(class_name)
                    recommendation_details = get_recommendation_details(class_name)

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

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def get_all_classes(self) -> Dict[int, str]:
        return CLASS_NAMES.copy()
