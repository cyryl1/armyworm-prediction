"""Tests to ensure OpenAPI includes API key security for protected endpoints."""

import os
from fastapi.testclient import TestClient
from app.main import app


def test_openapi_contains_api_key_security():
    client = TestClient(app)
    res = client.get('/openapi.json')
    assert res.status_code == 200
    doc = res.json()

    # securitySchemes.ApiKeyAuth should exist
    components = doc.get('components', {})
    security = components.get('securitySchemes', {})
    assert 'ApiKeyAuth' in security
    scheme = security['ApiKeyAuth']
    assert scheme.get('type') == 'apiKey'
    assert scheme.get('name') == 'x-api-key'

    # /detect and /history should be marked as requiring security
    paths = doc.get('paths', {})
    for p in ['/detect', '/history']:
        assert p in paths
        methods = paths[p]
        # check at least one method has security requirement
        has_sec = any((methods[m].get('security') for m in methods))
        assert has_sec
