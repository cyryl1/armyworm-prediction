"""Simple API-key authentication helpers.

Use `API_KEY` env var to configure a single valid key. HTTP endpoints use
the `x-api-key` header. WebSocket clients must include `api_key` as a query
parameter or `x-api-key` header.
"""

from __future__ import annotations

import os
from fastapi import Header, HTTPException
from fastapi import WebSocket


def _expected_key() -> str | None:
    return os.getenv("API_KEY")


async def api_key_header(x_api_key: str | None = Header(None)) -> str:
    expected = _expected_key()
    if not expected:
        raise HTTPException(status_code=500, detail="API key not configured")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key


async def verify_ws_api_key(websocket: WebSocket) -> bool:
    """Verify API key for WebSocket; closes connection on failure.

    Returns True if valid, otherwise closes the websocket and returns False.
    """
    expected = _expected_key()
    if not expected:
        await websocket.close(code=1011)
        return False

    # accept either query param or header
    api_key = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")
    if api_key != expected:
        await websocket.close(code=1008)
        return False

    return True
