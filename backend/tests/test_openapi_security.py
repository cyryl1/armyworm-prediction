"""Tests to ensure OpenAPI is public and does not contain API key security requirements."""

from fastapi.testclient import TestClient
from app.main import app


def test_openapi_public_access():
    client = TestClient(app)
    res = client.get('/openapi.json')
    assert res.status_code == 200
    doc = res.json()

    components = doc.get('components', {})
    security = components.get('securitySchemes', {})
    # No ApiKeyAuth should exist
    assert 'ApiKeyAuth' not in security

    paths = doc.get('paths', {})
    for p in ['/detect', '/history']:
        if p in paths:
            methods = paths[p]
            # None of the methods should require security schema
            for m in methods:
                assert 'security' not in methods[m]
