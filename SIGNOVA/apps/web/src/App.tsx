import React, { useEffect, useRef, useState } from 'react';
import './App.css';

const API_BASE = import.meta.env.VITE_SIGNOVA_API_URL || 'http://127.0.0.1:8000';
const WS_BASE = import.meta.env.VITE_SIGNOVA_WS_URL || 'ws://127.0.0.1:8000/ws/live';

interface StageLatencies {
  capture_latency_ms: number;
  frame_decode_latency_ms: number;
  mediapipe_latency_ms: number;
  normalization_latency_ms: number;
  model_inference_latency_ms: number;
  ctc_decoding_latency_ms: number;
  translation_latency_ms: number;
  total_pipeline_latency_ms: number;
}

interface RuntimeSnapshot {
  session_id: string;
  frame_index: number;
  camera_status: 'CONNECTED' | 'DISCONNECTED' | 'ERROR' | 'UNAVAILABLE';
  tracking_status: 'GOOD' | 'DEGRADED' | 'LOST';
  activity_status: 'ACTIVE' | 'LOW_ACTIVITY' | 'UNKNOWN';
  model_status: 'AVAILABLE' | 'UNAVAILABLE' | 'INCOMPATIBLE' | 'BLOCKED';
  translation_status: 'READY' | 'BLOCKED';
  current_gloss: string;
  committed_glosses: string[];
  current_translation: string;
  buffer_frames: number;
  buffer_capacity: number;
  min_frames_for_inference: number;
  input_fps: number;
  processed_fps: number;
  latencies: StageLatencies;
  scientific_state: string;
  supervision_blocker: string;
  status_message: string;
  device: string;
}

export function App() {
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [cameraPermission, setCameraPermission] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const [snapshot, setSnapshot] = useState<RuntimeSnapshot>({
    session_id: '',
    frame_index: 0,
    camera_status: 'DISCONNECTED',
    tracking_status: 'LOST',
    activity_status: 'UNKNOWN',
    model_status: 'UNAVAILABLE',
    translation_status: 'BLOCKED',
    current_gloss: '—',
    committed_glosses: [],
    current_translation: '—',
    buffer_frames: 0,
    buffer_capacity: 64,
    min_frames_for_inference: 64,
    input_fps: 0,
    processed_fps: 0,
    latencies: {
      capture_latency_ms: 0,
      frame_decode_latency_ms: 0,
      mediapipe_latency_ms: 0,
      normalization_latency_ms: 0,
      model_inference_latency_ms: 0,
      ctc_decoding_latency_ms: 0,
      translation_latency_ms: 0,
      total_pipeline_latency_ms: 0,
    },
    scientific_state: 'STATE_B',
    supervision_blocker: 'no_genuine_human_annotations_present',
    status_message:
      'Live camera tracking is active. A genuine sequential ISL recognition model is not currently available.',
    device: 'cpu',
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureIntervalRef = useRef<number | null>(null);

  // Poll REST status on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/live/status`)
      .then((res) => res.json())
      .then((data) => setSnapshot(data))
      .catch(() => {});
  }, []);

  // Connect WebSocket when streaming starts
  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(WS_BASE);
      ws.onopen = () => {
        setWsConnected(true);
      };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'SNAPSHOT' && msg.payload) {
            setSnapshot(msg.payload);
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
        }
      };
      ws.onclose = () => {
        setWsConnected(false);
      };
      ws.onerror = () => {
        setWsConnected(false);
      };
      wsRef.current = ws;
    } catch (err: any) {
      console.error('WebSocket connection error:', err);
    }
  };

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraPermission(true);
      setIsStreaming(true);
      connectWebSocket();

      // Notify backend of session start
      fetch(`${API_BASE}/api/live/session/start`, { method: 'POST' }).catch(() => {});

      // Frame capture loop (~25 FPS = 40ms interval)
      captureIntervalRef.current = window.setInterval(() => {
        captureAndSendFrame();
      }, 40);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setCameraError(err.message || 'Unable to access webcam. Please verify permissions.');
      setCameraPermission(false);
      setIsStreaming(false);
    }
  };

  const stopCamera = () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsStreaming(false);
    setWsConnected(false);

    // Notify backend session stop
    fetch(`${API_BASE}/api/live/session/stop`, { method: 'POST' }).catch(() => {});
  };

  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, 640, 480);
    const base64Data = canvas.toDataURL('image/jpeg', 0.7);

    wsRef.current.send(
      JSON.stringify({
        type: 'FRAME',
        data: base64Data,
        timestamp: performance.now(),
      })
    );
  };

  const handleResetSession = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'SIGNAL_RESET' }));
    } else {
      fetch(`${API_BASE}/api/live/session/reset`, { method: 'POST' })
        .then((res) => res.json())
        .then((data) => {
          if (data.snapshot) setSnapshot(data.snapshot);
        })
        .catch(() => {});
    }
  };

  const handleSpeak = () => {
    if (!snapshot.current_translation || snapshot.current_translation === '—') return;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(snapshot.current_translation);
      utterance.rate = 1.0;
      utterance.lang = 'en-US';
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="signova-app">
      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Header */}
      <header className="header">
        <div className="logo-container">
          <div className="logo-icon">⚡</div>
          <div>
            <h1 className="logo-title">SIGNOVA</h1>
            <p className="logo-subtitle">Continuous Indian Sign Language (ISL) → English Translation System</p>
          </div>
        </div>
        <div className="badge-container">
          <span className="phase-pill">PHASE 20: LIVE STREAMING RUNTIME</span>
          <span className={`status-pill ${wsConnected ? 'online' : 'offline'}`}>
            {wsConnected ? '● WS CONNECTED' : '○ WS DISCONNECTED'}
          </span>
        </div>
      </header>

      {/* Main Grid */}
      <main className="main-content">
        {/* Left: Live Video & Controls */}
        <div className="video-section">
          <div className="card video-card">
            <div className="card-header">
              <h2 className="card-title">Live Camera Stream</h2>
              <div className="status-badges-inline">
                <span className={`badge-pill tracking-${snapshot.tracking_status.toLowerCase()}`}>
                  TRACKING: {snapshot.tracking_status}
                </span>
                <span className={`badge-pill activity-${snapshot.activity_status.toLowerCase()}`}>
                  ACTIVITY: {snapshot.activity_status}
                </span>
              </div>
            </div>

            <div className="video-container">
              <video ref={videoRef} className="live-video" playsInline muted autoPlay />
              {!isStreaming && (
                <div className="video-placeholder">
                  <div className="placeholder-icon">📹</div>
                  <p className="placeholder-title">Live Camera Inactive</p>
                  <p className="placeholder-desc">
                    Click "Start Camera" below to initiate local real-time perception and landmark tracking.
                  </p>
                </div>
              )}
            </div>

            {cameraError && <div className="error-banner">⚠️ {cameraError}</div>}

            {/* Control Bar */}
            <div className="controls-bar">
              {!isStreaming ? (
                <button className="btn btn-primary" onClick={startCamera}>
                  ▶ START CAMERA
                </button>
              ) : (
                <button className="btn btn-danger" onClick={stopCamera}>
                  ⏹ STOP CAMERA
                </button>
              )}
              <button className="btn btn-secondary" onClick={handleResetSession}>
                🔄 RESET SESSION
              </button>
              <button
                className="btn btn-accent"
                onClick={handleSpeak}
                disabled={!snapshot.current_translation || snapshot.current_translation === '—'}
              >
                🔊 SPEAK
              </button>
            </div>
          </div>

          {/* Performance Telemetry */}
          <div className="card telemetry-card">
            <h3 className="card-subtitle">Real-Time Performance Telemetry</h3>
            <div className="metrics-grid">
              <div className="metric-box">
                <span className="metric-label">Input FPS</span>
                <span className="metric-value">{snapshot.input_fps}</span>
              </div>
              <div className="metric-box">
                <span className="metric-label">Processed FPS</span>
                <span className="metric-value">{snapshot.processed_fps}</span>
              </div>
              <div className="metric-box">
                <span className="metric-label">Temporal Buffer</span>
                <span className="metric-value">
                  {snapshot.buffer_frames} / {snapshot.buffer_capacity}
                </span>
              </div>
              <div className="metric-box">
                <span className="metric-label">End-to-End Latency</span>
                <span className="metric-value">
                  {snapshot.latencies.total_pipeline_latency_ms} <span className="unit">ms</span>
                </span>
              </div>
            </div>

            {/* Stage Latency Breakdown */}
            <div className="latency-breakdown">
              <div className="latency-bar-item">
                <span>MediaPipe Holistic:</span>
                <span>{snapshot.latencies.mediapipe_latency_ms} ms</span>
              </div>
              <div className="latency-bar-item">
                <span>Feature Normalization:</span>
                <span>{snapshot.latencies.normalization_latency_ms} ms</span>
              </div>
              <div className="latency-bar-item">
                <span>Model Inference:</span>
                <span>{snapshot.latencies.model_inference_latency_ms} ms</span>
              </div>
              <div className="latency-bar-item">
                <span>Translation Bridge:</span>
                <span>{snapshot.latencies.translation_latency_ms} ms</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Recognition & Scientific State */}
        <div className="translation-section">
          {/* Recognition Card */}
          <div className="card recognition-card">
            <div className="card-header">
              <h2 className="card-title">Continuous ISL Recognition</h2>
              <span className={`badge-pill model-${snapshot.model_status.toLowerCase()}`}>
                MODEL: {snapshot.model_status}
              </span>
            </div>

            <div className="recognition-box">
              <div className="field-group">
                <label className="field-label">Current Partial Gloss</label>
                <div className="gloss-badge current">{snapshot.current_gloss}</div>
              </div>

              <div className="field-group">
                <label className="field-label">Committed Gloss Sequence</label>
                <div className="committed-glosses-list">
                  {snapshot.committed_glosses.length > 0 ? (
                    snapshot.committed_glosses.map((g, idx) => (
                      <span key={idx} className="gloss-pill">
                        {g}
                      </span>
                    ))
                  ) : (
                    <span className="empty-text">No committed glosses yet</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Translation Card */}
          <div className="card translation-card">
            <div className="card-header">
              <h2 className="card-title">Neural English Translation</h2>
              <span className={`badge-pill translation-${snapshot.translation_status.toLowerCase()}`}>
                TRANSLATION: {snapshot.translation_status}
              </span>
            </div>

            <div className="translation-output-box">
              <p className="translation-text">{snapshot.current_translation}</p>
            </div>

            {/* Scientific Gate Honest Notice */}
            <div className="scientific-notice">
              <div className="notice-header">
                <span className="notice-icon">🔬</span>
                <strong>Scientific Gate State: {snapshot.scientific_state}</strong>
              </div>
              <p className="notice-body">{snapshot.status_message}</p>
            </div>
          </div>

          {/* Pipeline Topology Diagnostics */}
          <div className="card pipeline-card">
            <h3 className="card-subtitle">Perception & Feature Pipeline</h3>
            <div className="diagnostics-list">
              <div className="diag-row">
                <span className="diag-key">Landmark Extractor</span>
                <span className="diag-val">MediaPipe Holistic (543 Landmarks)</span>
              </div>
              <div className="diag-row">
                <span className="diag-key">Feature Topology</span>
                <span className="diag-val">HANDS_POSE (Default Sequence Geometry)</span>
              </div>
              <div className="diag-row">
                <span className="diag-key">Target Device</span>
                <span className="diag-val">{snapshot.device.toUpperCase()}</span>
              </div>
              <div className="diag-row">
                <span className="diag-key">Data Provenance</span>
                <span className="diag-val">Strict Real Human Supervision Only</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
