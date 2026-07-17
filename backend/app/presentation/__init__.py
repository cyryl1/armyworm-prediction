"""Presentation layer — HTTP/WebSocket handlers.

This layer adapts external requests (REST, WebSocket) into calls to
the use-case layer.  It is the only layer that knows about FastAPI.
"""
