"""
FastAPI Live Streaming Routes & WebSocket Endpoint for SIGNOVA.
"""

import json
from typing import Any, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from signova.live.runtime import SignovaLiveRuntime
from signova.live.status import CameraStatus

router = APIRouter(tags=["Live Streaming"])

# Shared singleton runtime instance
live_runtime = SignovaLiveRuntime()


class ResetRequest(BaseModel):
    reason: Optional[str] = "user_requested"


@router.get("/api/live/status")
async def get_live_status():
    """Returns current live perception and model availability snapshot."""
    return live_runtime.get_state_snapshot().to_dict()


@router.post("/api/live/session/start")
async def start_live_session():
    """Starts a new live inference session."""
    session = live_runtime.session_manager.start_session()
    return {"status": "SESSION_STARTED", "session": session.to_dict()}


@router.post("/api/live/session/stop")
async def stop_live_session():
    """Stops the active live inference session."""
    session = live_runtime.session_manager.end_session()
    live_runtime.set_camera_status(CameraStatus.DISCONNECTED)
    return {"status": "SESSION_STOPPED", "session": session.to_dict() if session else {}}


@router.post("/api/live/session/reset")
async def reset_live_session(payload: Optional[ResetRequest] = None):
    """Sends SIGNAL_RESET: Clears buffer, resets token commitment and translation state."""
    snapshot = live_runtime.signal_reset()
    return {"status": "SESSION_RESET", "snapshot": snapshot.to_dict()}


@router.get("/api/live/model")
async def get_live_model_info():
    """Returns model registry hardware status, spec, and Phase 19 gate status."""
    status, meta, reason = live_runtime.model_registry.inspect_model_availability()
    hardware = live_runtime.model_registry.get_hardware_info()
    default_spec = live_runtime.model_registry.default_spec.to_dict()

    return {
        "model_status": status.value,
        "reason": reason,
        "hardware": hardware,
        "model_metadata": meta.to_dict() if meta else None,
        "input_spec": meta.input_spec.to_dict() if meta else default_spec,
    }


@router.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """
    Real-time bi-directional streaming WebSocket endpoint:
    - Ingests camera frames (base64 string or binary bytes)
    - Processes perception, features, buffer, and CTC/Translation pipeline
    - Returns real-time JSON snapshots with FPS, latencies, and recognition state
    """
    await websocket.accept()
    live_runtime.set_camera_status(CameraStatus.CONNECTED)

    try:
        while True:
            # Handle text (JSON) or binary messages
            message = await websocket.receive()

            if "text" in message:
                raw_text = message["text"]
                try:
                    msg_data = json.loads(raw_text)
                except Exception:
                    continue

                msg_type = msg_data.get("type", "FRAME")
                if msg_type == "FRAME":
                    frame_data = msg_data.get("data", "")
                    ts = msg_data.get("timestamp")
                    if frame_data:
                        snapshot = live_runtime.process_frame(frame_data, capture_timestamp_ms=ts)
                        await websocket.send_json({"type": "SNAPSHOT", "payload": snapshot.to_dict()})
                elif msg_type == "SIGNAL_RESET":
                    snapshot = live_runtime.signal_reset()
                    await websocket.send_json({"type": "SNAPSHOT", "payload": snapshot.to_dict()})
                elif msg_type == "PING":
                    await websocket.send_json({"type": "PONG", "timestamp": msg_data.get("timestamp")})

            elif "bytes" in message:
                byte_data = message["bytes"]
                if byte_data:
                    snapshot = live_runtime.process_frame(byte_data)
                    await websocket.send_json({"type": "SNAPSHOT", "payload": snapshot.to_dict()})

    except WebSocketDisconnect:
        live_runtime.set_camera_status(CameraStatus.DISCONNECTED)
    except Exception as e:
        live_runtime.set_camera_status(CameraStatus.ERROR)
        try:
            await websocket.close()
        except Exception:
            pass
