# SIGNOVA Phase 20 — Live Camera Inference Runtime & Temporal Streaming Architecture

## 1. Executive Summary

**SIGNOVA Phase 20** implements a real-time, low-latency live camera streaming and perception architecture for continuous Indian Sign Language (ISL) recognition and translation.

The architecture enforces the core scientific principle:
$$\text{GENUINE HUMAN SUPERVISION} > \text{DATA QUALITY} > \text{SCIENTIFIC VALIDITY} > \text{MODEL TRAINING}$$

The live streaming engine operates in two well-defined states:
1. **Real Model Available**: When authorized by the canonical Phase 19 supervision gate (`STATE_A` or `STATE_A_DATA_LIMITED`), runs genuine CTC model inference, token decoding, and gloss-to-English translation.
2. **Diagnostic Mode (Current `STATE_B`)**: Executes local camera capture, MediaPipe Holistic landmark extraction (543 landmarks), feature normalization, and temporal buffering (64 frames), while explicitly and honestly displaying that genuine continuous ISL recognition is unavailable (zero synthetic checkpoint substitution, zero random guesses, zero LLM-generated translations).

---

## 2. End-to-End Streaming Architecture

```text
                    ┌─────────────────────────┐
                    │ Browser Web Application │
                    │ (getUserMedia 30 FPS)   │
                    └────────────┬────────────┘
                                 │
                      WebSocket (/ws/live)
                                 │
                                 ▼
┌─────────────────────────┐    ┌─────────────────────────────────┐
│ Standalone OpenCV Camera│───►│       SignovaLiveRuntime        │
│ (Local Python Window)   │    │  (Single Shared Unified Engine) │
└─────────────────────────┘    └────────────────┬────────────────┘
                                                │
                                                ▼
                                    MediaPipe Holistic (543 pts)
                                    [Pose (33), Face (468), LH (21), RH (21)]
                                                │
                                                ▼
                                    Feature Normalization
                                    (Canonical Sequence Normalizer)
                                                │
                                                ▼
                                    Activity & Tracking Evaluator
                                    (GOOD / DEGRADED / LOST)
                                    (ACTIVE / LOW_ACTIVITY / UNKNOWN)
                                                │
                                                ▼
                                    Temporal Frame Buffer
                                    (FIFO 64-Frame Window, Stride 16)
                                                │
                                                ▼
                                    Model Input Spec Compatibility
                                                │
                                ┌───────────────┴───────────────┐
                                │                               │
                             BLOCKED                        AUTHORIZED
                                │                               │
                                ▼                               ▼
                          STATE_B Telemetry               Real CTC Model
                          (Diagnostic Mode)                     │
                                │                               ▼
                                │                         CTC Greedy Decoder
                                │                               │
                                │                               ▼
                                │                         Ordered Glosses
                                │                               │
                                │                               ▼
                                │                         Gloss → English
                                │                               │
                                └───────────────┬───────────────┘
                                                ▼
                                    Real-Time Telemetry & UI
```

---

## 3. Core Subsystems

### 3.1 Unified `SignovaLiveRuntime`
Both the WebSocket browser client and the standalone Python OpenCV runner interface through the identical [`SignovaLiveRuntime`](file:///g:/SingLang/SIGNOVA/src/signova/live/runtime.py) class. This guarantees identical preprocessing, feature slicing, temporal windowing, and safety gate enforcement across all client surfaces.

### 3.2 Temporal Frame Buffer & `SIGNAL_RESET`
- **Capacity**: 64 frames (configurable default derived from trained model specs).
- **Stride**: 16 frames.
- **`SIGNAL_RESET`**: An explicit utterance reset triggered by user action, camera toggle, or reset button. Atomically clears the FIFO buffer, resets partial decoding states, and prevents runaway token accumulation across distinct signing sentences.

### 3.3 Activity & Tracking Separation
- **`TrackingStatus`** (`GOOD`, `DEGRADED`, `LOST`): Evaluates anatomical landmark presence rates.
- **`ActivityStatus`** (`ACTIVE`, `LOW_ACTIVITY`, `UNKNOWN`): Evaluates coordinate displacement across wrists and fingers to avoid executing expensive inference when the signer's hands are resting, without making unearned semantic segmentation claims.

### 3.4 Model Registry & Safety Gate
- **`ModelInputSpec`**: Declares expected feature group (`HANDS_POSE`), feature schema version, temporal window (64), temporal stride (16), landmark topology (543), normalization version, and vocabulary version.
- **Phase 19 Gate Check**: Enforces that only checkpoints trained under authorized supervision states can be loaded. Rejects synthetic test fixtures and mock weights.

---

## 4. Execution Commands

### Diagnostic Check
```powershell
python scripts/check_live_runtime.py
```

### Standalone OpenCV Camera Runner
```powershell
python scripts/run_live_camera.py
```

### Live Integration Smoke Test
```powershell
python scripts/test_live_runtime.py
```

### Performance Profiler
```powershell
python scripts/profile_live_runtime.py
```

### Full Interactive Platform Runner
```powershell
python run_signova.py
```

### Backend API & Live WebSocket Gateway
```powershell
python apps/api/main.py
```

### Web Application Frontend
```powershell
cd apps/web
npm run dev
```
