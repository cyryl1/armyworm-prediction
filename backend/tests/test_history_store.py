"""Tests for MongoDB-backed history storage."""

import os
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock, MagicMock

from app import history_store


class TestHistoryStore:
    """Test detection history persistence."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset the module-level client for each test."""
        history_store._client = None
        history_store._collection = None
        yield
        history_store._client = None
        history_store._collection = None

    @pytest.mark.asyncio
    async def test_init_db_creates_indexes(self):
        """Should initialize MongoDB and create indexes."""
        with patch("app.history_store.AsyncIOMotorClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_collection.create_index = AsyncMock()

            mock_db.__getitem__.return_value = mock_collection
            mock_client.__getitem__.return_value = mock_db
            mock_client_cls.return_value = mock_client

            os.environ["MONGO_URI"] = "mongodb://localhost:27017"
            await history_store.init_db()

            # Verify indexes were created
            assert mock_collection.create_index.call_count == 2
            mock_collection.create_index.assert_any_call([("detection_timestamp", -1)])
            mock_collection.create_index.assert_any_call("class_name")

    @pytest.mark.asyncio
    async def test_save_detection_returns_id(self):
        """Should return a string ID after saving."""
        with patch("app.history_store.AsyncIOMotorClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_collection.create_index = AsyncMock()
            mock_result = MagicMock()
            mock_result.inserted_id = "507f1f77bcf86cd799439011"
            mock_collection.insert_one = AsyncMock(return_value=mock_result)

            mock_db.__getitem__.return_value = mock_collection
            mock_client.__getitem__.return_value = mock_db
            mock_client_cls.return_value = mock_client

            await history_store.init_db()
            
            record = {
                "class_id": 2,
                "class_name": "fall-armyworm-larva",
                "confidence": 0.92,
                "bbox": [50, 30, 120, 90],
                "recommendation": "Apply control",
                "recommendation_details": {"severity": "high"},
                "detection_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            doc_id = await history_store.save_detection(record)
            assert doc_id == "507f1f77bcf86cd799439011"
            mock_collection.insert_one.assert_called_once_with(record)

    @pytest.mark.asyncio
    async def test_list_detections_returns_records(self):
        """Should return recent detection records in order."""
        with patch("app.history_store.AsyncIOMotorClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_collection.create_index = AsyncMock()
            mock_cursor = MagicMock()

            test_docs = [
                {
                    "_id": "507f1f77bcf86cd799439011",
                    "class_name": "fall-armyworm-larva",
                    "confidence": 0.92,
                },
                {
                    "_id": "507f1f77bcf86cd799439012",
                    "class_name": "healthy-maize",
                    "confidence": 0.78,
                },
            ]
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.to_list = AsyncMock(return_value=test_docs)
            mock_collection.find.return_value = mock_cursor

            mock_db.__getitem__.return_value = mock_collection
            mock_client.__getitem__.return_value = mock_db
            mock_client_cls.return_value = mock_client

            await history_store.init_db()

            records = await history_store.list_detections(limit=10)
            assert len(records) == 2
            assert records[0]["id"] == "507f1f77bcf86cd799439011"
            assert "_id" not in records[0]  # Should be renamed to "id"
            mock_collection.find.assert_called_once()
            mock_cursor.sort.assert_called_once_with("detection_timestamp", -1)
            mock_cursor.to_list.assert_called_once_with(length=10)

    def test_ensure_initialized_raises_if_not_init(self):
        """Should raise if MongoDB not initialized."""
        history_store._collection = None
        with pytest.raises(RuntimeError, match="not initialized"):
            history_store._ensure_initialized()
