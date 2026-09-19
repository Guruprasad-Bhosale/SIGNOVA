# Phase 24: Live Translation Validation & Registry Ownership

## Overview

Live model integration in SIGNOVA connects trained CTC models with the real-time browser WebSocket and OpenCV camera runtime.

## Single Ownership Architecture

Phase 24 does NOT own or directly mutate `live_model_pointer.json`. Phase 24 generates authorization requests, and the canonical Phase 20 `LiveModelRegistry` remains the exclusive owner of the pointer.

```
Phase 24 Orchestrator
         │
         ▼
LIVE_MODEL_AUTHORIZATION
         │
         ▼
Phase 20 LiveModelRegistry (Single Owner)
         │
         ▼
live_model_pointer.json
```

## Validation Distinction Levels

A successful smoke test does not imply that continuous ISL translation has been scientifically validated. SIGNOVA maintains clear distinction across 3 validation dimensions:

1. **Live Smoke Test / Runtime Validation**:
   - Status: `NOT_PERFORMED` / `LIMITED` / `PERFORMED`
   - Validates: Input tensor shape compatibility, WebSocket latency, frame rate stability (>= 25 FPS), feature normalization compatibility.
2. **ISL Recognition Validation**:
   - Status: `NOT_PERFORMED` / `LIMITED` / `PERFORMED`
   - Validates: Real-time recognition accuracy of continuous ISL signs in front of camera.
3. **Gloss to English Translation Validation**:
   - Status: `NOT_PERFORMED` / `LIMITED` / `PERFORMED`
   - Validates: Downstream syntactic conversion of recognized gloss sequences into fluent English sentences.

## Abstention & Silence Handling

- Low confidence predictions (< 0.70) are abstained rather than emitted as hallucinations.
- Inactive frames (resting hands, absence of signer) produce empty blank tokens and reset temporal buffers cleanly.
