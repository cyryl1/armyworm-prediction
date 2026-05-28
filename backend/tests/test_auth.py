"""Tests for API-key authentication."""

import os
import pytest
from fastapi import HTTPException
from unittest.mock import patch

from app import auth


class TestAPIKeyAuth:
    """Test API key authentication."""

    @pytest.fixture(autouse=True)
    def reset_env(self):
        """Reset environment variables for each test."""
        original_key = os.environ.get("API_KEY")
        yield
        if original_key:
            os.environ["API_KEY"] = original_key
        else:
            os.environ.pop("API_KEY", None)

    @pytest.mark.asyncio
    async def test_api_key_header_valid(self):
        """Should accept valid API key in header."""
        os.environ["API_KEY"] = "test-secret-key"
        result = await auth.api_key_header(x_api_key="test-secret-key")
        assert result == "test-secret-key"

    @pytest.mark.asyncio
    async def test_api_key_header_invalid(self):
        """Should reject invalid API key."""
        os.environ["API_KEY"] = "test-secret-key"
        with pytest.raises(HTTPException) as exc_info:
            await auth.api_key_header(x_api_key="wrong-key")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_api_key_header_missing(self):
        """Should reject missing API key."""
        os.environ["API_KEY"] = "test-secret-key"
        with pytest.raises(HTTPException) as exc_info:
            await auth.api_key_header(x_api_key=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_api_key_header_not_configured(self):
        """Should reject if API_KEY env var not set."""
        os.environ.pop("API_KEY", None)
        with pytest.raises(HTTPException) as exc_info:
            await auth.api_key_header(x_api_key="any-key")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_verify_ws_api_key_valid(self):
        """Should return True for valid WebSocket API key."""
        os.environ["API_KEY"] = "test-secret-key"
        
        mock_ws = type('MockWS', (), {
            'query_params': {'api_key': 'test-secret-key'},
            'headers': {},
            'close': None,
        })()
        
        result = await auth.verify_ws_api_key(mock_ws)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_ws_api_key_header_fallback(self):
        """Should use header if query param not provided."""
        os.environ["API_KEY"] = "test-secret-key"
        
        mock_ws = type('MockWS', (), {
            'query_params': {},
            'headers': {'x-api-key': 'test-secret-key'},
            'close': None,
        })()
        
        result = await auth.verify_ws_api_key(mock_ws)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_ws_api_key_invalid(self):
        """Should close WebSocket and return False for invalid key."""
        os.environ["API_KEY"] = "test-secret-key"
        
        close_called = False
        
        async def mock_close(*args, **kwargs):
            nonlocal close_called
            close_called = True
        
        mock_ws = type('MockWS', (), {
            'query_params': {'api_key': 'wrong-key'},
            'headers': {},
            'close': mock_close,
        })()
        
        result = await auth.verify_ws_api_key(mock_ws)
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_ws_api_key_not_configured(self):
        """Should close WebSocket if API_KEY not configured."""
        os.environ.pop("API_KEY", None)
        
        close_called = False
        
        async def mock_close(*args, **kwargs):
            nonlocal close_called
            close_called = True
        
        mock_ws = type('MockWS', (), {
            'query_params': {},
            'headers': {},
            'close': mock_close,
        })()
        
        result = await auth.verify_ws_api_key(mock_ws)
        assert result is False
