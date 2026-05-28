"""
MongoDB-backed persistence for detection history.

This module uses `motor` and reads `MONGO_URI` from environment variables.
If no `MONGO_URI` is provided it defaults to a local Mongo instance.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from bson.objectid import ObjectId


_client: Optional[AsyncIOMotorClient] = None
_collection: Optional[AsyncIOMotorCollection] = None


async def init_db() -> None:
    """Initialize the MongoDB client and ensure indexes exist."""
    global _client, _collection
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "pest_detection")
    coll_name = os.getenv("MONGO_COLLECTION", "detections")
    timeout_ms = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "3000"))

    _client = AsyncIOMotorClient(
        mongo_uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
        socketTimeoutMS=timeout_ms,
    )
    db = _client[db_name]
    _collection = db[coll_name]

    # Create useful indexes asynchronously
    await _collection.create_index([("detection_timestamp", -1)])
    await _collection.create_index("class_name")


def _ensure_initialized():
    if _collection is None:
        raise RuntimeError("MongoDB client not initialized. Call init_db() first.")


async def save_detection(record: Dict[str, Any]) -> str:
    """Insert a detection record and return the created document id as string."""
    _ensure_initialized()
    result = await _collection.insert_one(record)
    return str(result.inserted_id)


async def list_detections(limit: int = 100) -> List[Dict[str, Any]]:
    """Return the most recent detection records (most recent first)."""
    _ensure_initialized()
    cursor = _collection.find().sort("detection_timestamp", -1)
    docs = await cursor.to_list(length=limit)
    results: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = doc.get("_id")
        doc["id"] = str(doc_id) if isinstance(doc_id, ObjectId) else doc_id
        # remove internal _id
        doc.pop("_id", None)
        results.append(doc)
    return results
