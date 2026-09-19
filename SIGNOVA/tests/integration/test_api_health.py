"""
Integration tests for FastAPI endpoints (/health and /version).
"""

from fastapi.testclient import TestClient
import pytest
from apps.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "signova-api"
    assert data["version"] == "0.1.0"
    assert data["phase"] == 0
    assert "system" in data


def test_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "0.1.0"
    assert data["api_version"] == "v1"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["phase"] == 0
