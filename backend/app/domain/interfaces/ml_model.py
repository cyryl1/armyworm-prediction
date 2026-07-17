"""Port: ML model interface.

Any pest-detection model adapter (YOLO, TensorFlow, ONNX, etc.) must
implement this interface to be used by the application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class PestDetectorInterface(ABC):
    """Contract for a pest detection ML model."""

    @abstractmethod
    def load_model(self, model_path: str, device: Optional[str] = None) -> None:
        """Load model weights from disk into memory."""
        ...

    @abstractmethod
    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Run inference on a single image and return raw detection dicts.

        Each dict must contain at minimum:
            class_id, class_name, confidence, bbox, bbox_pixel, recommendation,
            recommendation_details
        """
        ...

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Return True if the model is ready for inference."""
        ...

    @abstractmethod
    def get_all_classes(self) -> Dict[int, str]:
        """Return the mapping of class-ID → class-name."""
        ...
