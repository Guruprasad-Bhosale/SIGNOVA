# SIGNOVA API Contract

## Overview

The SIGNOVA API provides RESTful endpoints for system health, dataset validation, batch video translation, and WebSocket streaming for low-latency live webcam inference.

---

## 1. System Endpoints (Implemented in Phase 0)

### `GET /health`
Returns current service operational health and hardware diagnostics.

**Response `200 OK`**:
```json
{
  "status": "ok",
  "service": "signova-api",
  "version": "0.1.0",
  "phase": 0,
  "system": {
    "python_version": "3.12.10",
    "platform": "Windows-11-10.0.26100-SP0",
    "processor": "Intel64 Family 6 Model 140",
    "pytorch": "2.3.0+cu121",
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU"
  }
}
```

### `GET /version`
Returns API metadata and supported translation capabilities.

**Response `200 OK`**:
```json
{
  "name": "SIGNOVA API",
  "version": "0.1.0",
  "api_version": "v1",
  "description": "Continuous Indian Sign Language to English Translation Pipeline",
  "supported_tasks": ["health_check", "dataset_validation"]
}
```

---

## 2. Planned Inference Endpoints (Phase 8/9)

### `POST /api/v1/inference/video`
Uploads a pre-recorded sign language video file for continuous recognition and translation.

**Request**:
- `multipart/form-data`: `file: video.mp4`
- `query_params`: `return_landmarks: bool` (default `false`)

**Planned Response `200 OK`**:
```json
{
  "glosses": ["HELLO", "MY", "NAME", "ISL"],
  "translation": "Hello, my name is in Indian Sign Language.",
  "confidence": 0.942,
  "latency_ms": 138.4,
  "num_frames": 96,
  "fps": 30.0
}
```

---

### `WebSocket /api/v1/inference/live`
Real-time bi-directional streaming for live webcam frames.

**Client Message (Frame Stream)**:
```json
{
  "type": "frame",
  "timestamp_ms": 1726581200000,
  "data": "<base64_encoded_jpeg_or_landmarks>"
}
```

**Server Message (Streaming Translation Delta)**:
```json
{
  "type": "prediction",
  "timestamp_ms": 1726581200032,
  "gloss_delta": "NAMASTE",
  "current_gloss_sequence": ["NAMASTE", "EVERYONE"],
  "partial_translation": "Namaste everyone.",
  "confidence": 0.96,
  "latency_ms": 28.5
}
```
