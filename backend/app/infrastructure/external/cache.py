"""In-memory TTL cache for recommendation data.

Provides a simple dictionary-based cache with per-entry expiration
and LRU-style eviction when capacity is reached.
"""

import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RecommendationCache:
    """In-memory TTL cache for recommendation data."""

    def __init__(self, ttl_hours: float = 24.0, max_entries: int = 50) -> None:
        self.ttl_seconds = ttl_hours * 3600
        self.max_entries = max_entries
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entry from the cache if it exists and is not expired."""
        if class_name not in self._cache:
            return None

        entry = self._cache[class_name]
        current_time = time.time()

        if current_time - entry["timestamp"] > self.ttl_seconds:
            logger.debug(f"Cache expired for {class_name}")
            del self._cache[class_name]
            return None

        logger.debug(f"Cache hit for {class_name}")
        return entry["data"]

    def set(self, class_name: str, data: Dict[str, Any]) -> None:
        """Store an entry in the cache."""
        if len(self._cache) >= self.max_entries and class_name not in self._cache:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

        self._cache[class_name] = {
            "timestamp": time.time(),
            "data": data,
        }
        logger.debug(f"Cached recommendation for {class_name}")

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()


# Global cache instance
recommendation_cache = RecommendationCache()
