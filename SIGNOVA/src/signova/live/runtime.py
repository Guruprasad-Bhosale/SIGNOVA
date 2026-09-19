"""
Unified SignovaLiveRuntime Engine for Live ISL Perception and Streaming.
"""

from collections import deque
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from signova.features.feature_groups import LandmarkGroup
from signova.features.mediapipe_extractor import MediaPipeHolisticExtractor
from signova.live.buffer import TemporalFrameBuffer
from signova.live.frame import decode_frame, validate_frame
from signova.live.metrics import LiveMetricsTracker
from signova.live.model_registry import LiveModelRegistry, ModelInputSpec
from signova.live.session import LiveSessionManager
from signova.live.status import (
    ActivityStatus,
    CameraStatus,
    ModelStatus,
    RuntimeStateSnapshot,
    StageLatencies,
    TrackingStatus,
    TranslationStatus,
)
from signova.preprocessing.normalization import normalize_landmark_sequence


class SignovaLiveRuntime:
    """
    Single unified live runtime powering both WebSocket browser streaming
    and standalone OpenCV camera pipelines.
    """

    def __init__(
        self,
        buffer_size: int = 64,
        stride: int = 16,
        min_frames_for_inference: int = 64,
        landmark_group: Union[str, LandmarkGroup] = LandmarkGroup.HANDS_POSE,
        device: Optional[str] = None,
        activity_threshold: float = 0.015,
    ):
        if isinstance(landmark_group, str):
            self.landmark_group = LandmarkGroup(landmark_group.lower())
        else:
            self.landmark_group = landmark_group
        self.activity_threshold = activity_threshold

        # Initialize sub-systems
        self.extractor = MediaPipeHolisticExtractor()
        self.buffer = TemporalFrameBuffer(
            buffer_size=buffer_size,
            stride=stride,
            min_frames_for_inference=min_frames_for_inference,
            landmark_group=self.landmark_group,
        )
        self.model_registry = LiveModelRegistry(device=device)
        self.metrics = LiveMetricsTracker()
        self.session_manager = LiveSessionManager()

        # Previous frame landmarks for activity/velocity detection
        self._prev_landmarks: Optional[np.ndarray] = None
        self._frame_counter = 0

        # State tracking
        self._camera_status = CameraStatus.DISCONNECTED
        self._tracking_status = TrackingStatus.LOST
        self._activity_status = ActivityStatus.UNKNOWN
        self._model_status = ModelStatus.UNAVAILABLE
        self._translation_status = TranslationStatus.BLOCKED

        # Decoding & Translation state
        self._current_gloss = "—"
        self._committed_glosses: List[str] = []
        self._current_translation = "—"
        self._recent_token_history: deque[int] = deque(maxlen=8)

        # Initial model check
        self._update_model_availability()

    def _update_model_availability(self):
        """Inspects model availability against Phase 19 gate."""
        status, meta, reason = self.model_registry.inspect_model_availability()
        self._model_status = status
        if status == ModelStatus.AVAILABLE and meta is not None:
            # Sync buffer configuration to model input spec
            self.buffer.configure_from_model_spec(
                temporal_window=meta.input_spec.temporal_window,
                temporal_stride=meta.input_spec.temporal_stride,
                landmark_group=meta.input_spec.feature_group,
            )
            self._translation_status = TranslationStatus.READY
        else:
            self._translation_status = TranslationStatus.BLOCKED

    def set_camera_status(self, status: CameraStatus):
        """Updates camera connection status."""
        self._camera_status = status

    def signal_reset(self) -> RuntimeStateSnapshot:
        """
        Explicit SIGNAL_RESET: Deterministically clears temporal buffer, resets
        utterance decoding, and clears current translation.
        """
        self.buffer.reset()
        self._prev_landmarks = None
        self._current_gloss = "—"
        self._committed_glosses.clear()
        self._current_translation = "—"
        self._recent_token_history.clear()
        self.metrics.reset()
        return self.get_state_snapshot()

    def _compute_activity_level(
        self,
        current_landmarks: np.ndarray,
        detection_mask: np.ndarray,
    ) -> ActivityStatus:
        """
        Determines whether the signer is actively moving their hands/arms (ACTIVE vs LOW_ACTIVITY).
        """
        # If no hands and no pose detected, tracking is lost -> UNKNOWN
        if detection_mask[0] == 0.0 and detection_mask[2] == 0.0 and detection_mask[3] == 0.0:
            return ActivityStatus.UNKNOWN

        if self._prev_landmarks is None:
            self._prev_landmarks = current_landmarks.copy()
            return ActivityStatus.ACTIVE

        # Calculate coordinate velocity across hand & pose landmarks
        # Pose (0..32) + LH (501..521) + RH (522..542)
        indices = list(range(0, 33)) + list(range(501, 543))
        diff = np.abs(current_landmarks[indices, :2] - self._prev_landmarks[indices, :2])
        mean_motion = float(np.mean(diff))
        self._prev_landmarks = current_landmarks.copy()

        if mean_motion >= self.activity_threshold:
            return ActivityStatus.ACTIVE
        return ActivityStatus.LOW_ACTIVITY

    def _evaluate_tracking_status(self, detection_mask: np.ndarray) -> TrackingStatus:
        """
        Evaluates tracking quality from detection masks:
        detection_mask = [pose, face, lh, rh]
        """
        pose_det = detection_mask[0] > 0.5
        lh_det = detection_mask[2] > 0.5
        rh_det = detection_mask[3] > 0.5

        if pose_det and (lh_det or rh_det):
            return TrackingStatus.GOOD
        elif pose_det or lh_det or rh_det:
            return TrackingStatus.DEGRADED
        return TrackingStatus.LOST

    def process_frame(
        self,
        raw_input: Union[np.ndarray, bytes, str],
        capture_timestamp_ms: Optional[float] = None,
    ) -> RuntimeStateSnapshot:
        """
        Processes a single live frame through perception, buffer, model check, and diagnostics.
        """
        t_pipeline_start = time.perf_counter()
        self.metrics.record_input_frame()
        self.session_manager.record_frame_received()
        self._frame_counter += 1

        t_cap_ms = (
            (time.monotonic() * 1000.0 - capture_timestamp_ms)
            if capture_timestamp_ms is not None
            else 0.0
        )

        # 1. Frame Decoding & Validation
        t_dec_0 = time.perf_counter()
        try:
            bgr_frame, _ = decode_frame(raw_input)
            if not validate_frame(bgr_frame):
                self.session_manager.record_frame_dropped()
                return self.get_state_snapshot()
        except Exception:
            self.session_manager.record_frame_dropped()
            return self.get_state_snapshot()
        t_dec_ms = (time.perf_counter() - t_dec_0) * 1000.0

        self._camera_status = CameraStatus.CONNECTED

        # 2. MediaPipe Landmark Extraction
        t_mp_0 = time.perf_counter()
        try:
            raw_landmarks, detection_mask = self.extractor.extract_from_frame(bgr_frame)
            self._tracking_status = self._evaluate_tracking_status(detection_mask)
            self._activity_status = self._compute_activity_level(raw_landmarks, detection_mask)
        except Exception:
            self.session_manager.record_tracking_failure()
            self._tracking_status = TrackingStatus.LOST
            self._activity_status = ActivityStatus.UNKNOWN
            raw_landmarks = np.zeros((543, 3), dtype=np.float32)
            detection_mask = np.zeros(4, dtype=np.float32)
        t_mp_ms = (time.perf_counter() - t_mp_0) * 1000.0

        # 3. Feature Normalization
        t_norm_0 = time.perf_counter()
        # Normalization operates on (T, 543, 3) -> expand dim and normalize
        lm_seq = raw_landmarks[np.newaxis, :, :]  # (1, 543, 3)
        norm_seq = normalize_landmark_sequence(lm_seq)
        norm_landmarks = norm_seq[0]
        t_norm_ms = (time.perf_counter() - t_norm_0) * 1000.0

        # 4. Temporal Buffer Ingestion
        self.buffer.push(
            landmarks=norm_landmarks,
            detection_mask=detection_mask,
            frame_index=self._frame_counter,
            timestamp_ms=time.monotonic() * 1000.0,
        )

        # 5. Model Availability & Inference
        t_inf_0 = time.perf_counter()
        t_inf_ms = 0.0
        t_dec_ctc_ms = 0.0
        t_trans_ms = 0.0

        self._update_model_availability()

        if self._model_status == ModelStatus.AVAILABLE and self.buffer.is_ready_for_inference():
            self.session_manager.record_inference_executed()
            self.buffer.mark_inference_executed()

            # Abstention on Inactive/Low-activity frames
            if self._activity_status in (ActivityStatus.LOW_ACTIVITY, ActivityStatus.UNKNOWN):
                self._current_gloss = "—"
            elif self.model_registry.verify_feature_compatibility(self.landmark_group):
                model, meta = self.model_registry.load_authorized_model()
                if model is not None and meta is not None:
                    feat_tensor, pad_mask, valid_T = self.buffer.get_feature_tensor(
                        device=self.model_registry.device,
                        group_override=meta.input_spec.feature_group,
                    )
                    lengths = torch.tensor([valid_T], dtype=torch.long, device=self.model_registry.device)

                    with torch.no_grad():
                        if hasattr(model, "decode_greedy"):
                            decoded = model.decode_greedy(feat_tensor, lengths=lengths)
                            raw_tokens = decoded[0].get("collapsed_tokens", []) if decoded else []
                            mean_conf = decoded[0].get("mean_confidence", 1.0) if decoded else 0.0
                        else:
                            # Flatten tensor if needed for Phase21BiGRUCTCModel: (1, 64, 50, 3) -> (1, 64, 150)
                            f_in = feat_tensor
                            if len(f_in.shape) == 4:
                                f_in = f_in.reshape(f_in.shape[0], f_in.shape[1], -1)
                            logits = model(f_in, lengths)
                            probs = torch.softmax(logits, dim=-1)
                            preds = logits.argmax(dim=-1)[0, :valid_T].cpu().numpy()
                            prob_np = probs[0, :valid_T].cpu().numpy()
                            collapsed = []
                            confs = []
                            prev = None
                            for t_idx, tok in enumerate(preds):
                                if tok != 0 and tok != prev:
                                    collapsed.append(int(tok))
                                    confs.append(float(prob_np[t_idx, tok]))
                                prev = tok
                            raw_tokens = collapsed
                            mean_conf = float(np.mean(confs)) if confs else 0.0

                    t_inf_ms = (time.perf_counter() - t_inf_0) * 1000.0

                    t_ctc_0 = time.perf_counter()
                    # Confidence Gating: HIGH CONFIDENCE (>=0.50) vs LOW CONFIDENCE (<0.50)
                    if raw_tokens:
                        if mean_conf >= 0.50:
                            from signova.qualification.vocabulary import Phase12GlossVocabulary
                            vocab = Phase12GlossVocabulary.from_json(meta.vocabulary_path)
                            glosses = vocab.decode(raw_tokens, remove_blank=True)
                            if glosses:
                                self._current_gloss = glosses[-1]
                                for g in glosses:
                                    if not self._committed_glosses or self._committed_glosses[-1] != g:
                                        self._committed_glosses.append(g)
                        else:
                            # Low confidence: abstain and wait
                            self._current_gloss = "WAIT / UNCERTAIN"
                    t_dec_ctc_ms = (time.perf_counter() - t_ctc_0) * 1000.0

                    # Translation
                    t_tr_0 = time.perf_counter()
                    if self._committed_glosses:
                        from signova.translation.translator import GlossToEnglishTranslator
                        translator = GlossToEnglishTranslator()
                        self._current_translation = translator.translate_glosses(self._committed_glosses)
                        self.session_manager.record_translation_produced()
                    t_trans_ms = (time.perf_counter() - t_tr_0) * 1000.0
            else:
                self._model_status = ModelStatus.INCOMPATIBLE
        else:
            # Diagnostic Mode under current STATE_B: No fake predictions
            self._current_gloss = "—"
            self._current_translation = "—"

        t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        # Record latencies & metrics
        self.metrics.set_latencies(
            capture_ms=t_cap_ms,
            frame_decode_ms=t_dec_ms,
            mediapipe_ms=t_mp_ms,
            normalization_ms=t_norm_ms,
            model_inference_ms=t_inf_ms,
            ctc_decoding_ms=t_dec_ctc_ms,
            translation_ms=t_trans_ms,
            total_pipeline_ms=t_total_ms,
        )
        self.metrics.record_processed_frame()
        self.session_manager.record_frame_processed()

        return self.get_state_snapshot()

    def get_state_snapshot(self) -> RuntimeStateSnapshot:
        """Emits a complete immutable snapshot of the live runtime state."""
        sess = self.session_manager.get_active_session()
        msg = (
            "Live camera tracking is active. A genuine sequential ISL recognition model "
            "is not currently available."
            if self._model_status != ModelStatus.AVAILABLE
            else "Genuine ISL recognition model active."
        )

        return RuntimeStateSnapshot(
            session_id=sess.session_id,
            frame_index=self._frame_counter,
            camera_status=self._camera_status,
            tracking_status=self._tracking_status,
            activity_status=self._activity_status,
            model_status=self._model_status,
            translation_status=self._translation_status,
            current_gloss=self._current_gloss,
            committed_glosses=list(self._committed_glosses),
            current_translation=self._current_translation,
            buffer_frames=self.buffer.current_frames_count,
            buffer_capacity=self.buffer.buffer_size,
            min_frames_for_inference=self.buffer.min_frames_for_inference,
            input_fps=self.metrics.input_fps,
            processed_fps=self.metrics.processed_fps,
            latencies=self.metrics.stage_latencies,
            scientific_state="STATE_B",
            supervision_blocker="no_genuine_human_annotations_present",
            status_message=msg,
            device=self.model_registry.device,
        )
