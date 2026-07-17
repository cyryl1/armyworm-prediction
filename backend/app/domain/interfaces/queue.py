"""Port: Task queue interface.

Defines the contract for offloading heavy work (ML inference) to
background workers.  Concrete implementations may use Celery, RQ,
or an in-process fallback for development.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class TaskQueueInterface(ABC):
    """Contract for an async task queue."""

    @abstractmethod
    def enqueue_detection(
        self,
        image_path: str,
        gps_latitude: Optional[float] = None,
        gps_longitude: Optional[float] = None,
    ) -> str:
        """Submit a detection job and return a task ID."""
        ...

    @abstractmethod
    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Fetch the current status and result of a queued task.

        Returns a dict with at minimum:
            status: "pending" | "processing" | "completed" | "failed"
            result: <detection payload or None>
            error:  <error message or None>
        """
        ...
