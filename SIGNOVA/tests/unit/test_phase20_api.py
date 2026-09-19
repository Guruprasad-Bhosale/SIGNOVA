"""
Unit tests for Phase 20 FastAPI Endpoints and WebSocket live stream.
"""

from fastapi.testclient import TestClient
import numpy as np
import pytest

from apps.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_live_status(client):
    response = client.get("/api/live/status")
    assert response.status_code == 200
    data = response.json()
    assert "scientific_state" in data
    assert "buffer_frames" in data
    assert "model_status" in data
    assert data["scientific_state"] == "STATE_B"


def test_api_live_model_info(client):
    response = client.get("/api/live/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_status" in data
    assert "hardware" in data
    assert "input_spec" in data
    assert data["input_spec"]["temporal_window"] == 64


def test_api_live_session_lifecycle(client):
    # Start session
    res_start = client.post("/api/live/session/start")
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "SESSION_STARTED"

    # Reset session
    res_reset = client.post("/api/live/session/reset", json={"reason": "test"})
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "SESSION_RESET"

    # Stop session
    res_stop = client.post("/api/live/session/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "SESSION_STOPPED"


def test_websocket_live_stream_ping(client):
    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_json({"type": "PING", "timestamp": 12345.0})
        data = websocket.receive_json()
        assert data["type"] == "PONG"
        assert data["timestamp"] == 12345.0


def test_websocket_live_stream_signal_reset(client):
    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_json({"type": "SIGNAL_RESET"})
        data = websocket.receive_json()
        assert data["type"] == "SNAPSHOT"
        assert data["payload"]["buffer_frames"] == 0
