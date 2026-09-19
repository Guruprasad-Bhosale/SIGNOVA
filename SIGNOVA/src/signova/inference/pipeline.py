"""
End-to-End Sign Translation Pipeline & Purely Deterministic Recognition-to-Translation Bridge.

Guiding Principles:
1. Purely deterministic Recognition-to-Translation Bridge:
   CTC token IDs -> remove blank -> consecutive repeat collapse -> ID to gloss string -> ordered gloss list -> translator.
2. ZERO semantic filtering, ZERO spelling corrections, ZERO synonym substitutions, ZERO LLM cleanup.
3. Modular stage decoupling with stage-by-stage latency diagnostics.
4. Robust to empty frames, missing landmarks, and invalid inputs.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from signova.data.vocabulary import SignVocabulary
from signova.features.feature_groups import LandmarkGroup, slice_landmark_tensor
from signova.models.ctc_recognizer import CTCContinuousRecognizer
from signova.preprocessing.normalization import normalize_landmark_sequence
from signova.translation.translator import GlossToEnglishTranslator


@dataclass
class PipelineStageDiagnostics:
    """
    Stage-by-stage latency and throughput diagnostics.
    """
    extraction_latency_ms: float = 0.0
    normalization_latency_ms: float = 0.0
    recognition_latency_ms: float = 0.0
    bridge_latency_ms: float = 0.0
    translation_latency_ms: float = 0.0
    total_pipeline_latency_ms: float = 0.0
    num_frames_processed: int = 0
    num_glosses_recognized: int = 0
    throughput_fps: float = 0.0


@dataclass
class TranslationResult:
    """
    Standard result returned by the EndToEndSignTranslationPipeline.
    """
    glosses: List[str] = field(default_factory=list)
    translation: str = ""
    confidence: float = 0.0
    stage_diagnostics: Optional[PipelineStageDiagnostics] = None
    is_synthetic_evaluation: bool = False
    disclaimer: str = "ENGINEERING PIPELINE EXECUTION"


class RecognitionToTranslationBridge:
    """
    Purely deterministic decoding bridge between CTC Sign Recognizer and Neural Translator.
    
    Strictly follows:
    Raw CTC Token IDs -> Blank Removal -> Consecutive Repeat Collapse -> Vocabulary Gloss Lookup -> Ordered Gloss Sequence
    
    Zero semantic tampering, zero spelling alteration, zero LLM hallucination.
    """

    def __init__(
        self,
        sign_vocab: SignVocabulary,
        blank_idx: int = 0,
    ):
        self.sign_vocab = sign_vocab
        self.blank_idx = blank_idx

    def decode_tokens_to_glosses(
        self,
        token_ids: Sequence[int],
    ) -> List[str]:
        """
        Deterministically collapse CTC tokens and map to ordered gloss strings.

        Args:
            token_ids: Sequence of frame-level CTC argmax class predictions.

        Returns:
            Ordered list of lexical sign gloss strings.
        """
        if not token_ids:
            return []

        # 1. CTC consecutive repeat collapse & blank removal
        collapsed_ids: List[int] = []
        prev_id = None

        for tid in token_ids:
            if tid != prev_id:
                if tid != self.blank_idx:
                    collapsed_ids.append(tid)
                prev_id = tid

        # 2. Map integer IDs to vocabulary gloss strings
        gloss_sequence = self.sign_vocab.decode(collapsed_ids, remove_blank=True)
        return gloss_sequence


class EndToEndSignTranslationPipeline:
    """
    Production end-to-end pipeline connecting:
    Input Landmarks -> Normalization -> CTC Continuous Recognizer -> Bridge -> Gloss-to-English Translator -> English Sentence
    """

    def __init__(
        self,
        recognizer: Optional[CTCContinuousRecognizer] = None,
        sign_vocab: Optional[SignVocabulary] = None,
        translator: Optional[GlossToEnglishTranslator] = None,
        normalizer: Optional[Any] = None,
        landmark_group: Union[str, LandmarkGroup] = LandmarkGroup.HANDS_POSE,
        device: str = "cpu",
        is_synthetic_evaluation: bool = True,
    ):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.recognizer = recognizer.to(self.device) if recognizer is not None else None
        if self.recognizer is not None:
            self.recognizer.eval()
        self.sign_vocab = sign_vocab
        self.translator = translator
        self.normalizer = normalizer or normalize_landmark_sequence
        self.landmark_group = (
            LandmarkGroup(landmark_group) if isinstance(landmark_group, str) else landmark_group
        )
        self.is_synthetic_evaluation = is_synthetic_evaluation

        self.bridge = (
            RecognitionToTranslationBridge(self.sign_vocab)
            if self.sign_vocab is not None
            else None
        )

    def process_landmarks(
        self,
        landmarks: Union[np.ndarray, torch.Tensor],
        fps: float = 30.0,
    ) -> TranslationResult:
        """
        Process a continuous landmark tensor through the full pipeline.

        Args:
            landmarks: Array or tensor of shape (T, 543, 3) or (T, num_landmarks, 3).
            fps: Video frames per second.

        Returns:
            TranslationResult with glosses, translation, and stage diagnostics.
        """
        t_start = time.perf_counter()
        diagnostics = PipelineStageDiagnostics()

        # 1. Validation and Tensor conversion
        if isinstance(landmarks, np.ndarray):
            landmarks_tensor = torch.from_numpy(landmarks).float()
        else:
            landmarks_tensor = landmarks.float()

        if landmarks_tensor.dim() == 3:
            # (T, num_landmarks, 3) -> Add batch dim: (1, T, num_landmarks, 3)
            landmarks_tensor = landmarks_tensor.unsqueeze(0)

        num_frames = landmarks_tensor.size(1)
        diagnostics.num_frames_processed = num_frames

        if num_frames == 0:
            return TranslationResult(
                glosses=[],
                translation="",
                confidence=0.0,
                stage_diagnostics=diagnostics,
                is_synthetic_evaluation=self.is_synthetic_evaluation,
            )

        # 2. Stage 2: Feature Group Slicing & Normalization
        t_norm_0 = time.perf_counter()
        sliced_features = slice_landmark_tensor(landmarks_tensor, self.landmark_group)
        sliced_features = sliced_features.to(self.device)
        diagnostics.normalization_latency_ms = (time.perf_counter() - t_norm_0) * 1000.0

        # 3. Stage 3: Continuous Sign Recognition (CTC)
        t_rec_0 = time.perf_counter()
        gloss_sequence: List[str] = []
        confidence: float = 0.0

        if self.recognizer is not None:
            lengths = torch.tensor([num_frames], dtype=torch.long, device=self.device)
            with torch.no_grad():
                decoded = self.recognizer.decode_greedy(sliced_features, lengths=lengths)
                if decoded and len(decoded) > 0:
                    raw_token_ids = decoded[0].get("collapsed_tokens", [])
                    confidence = float(decoded[0].get("confidence", 1.0))
        else:
            raw_token_ids = []

        diagnostics.recognition_latency_ms = (time.perf_counter() - t_rec_0) * 1000.0

        # 4. Stage 4: Recognition-to-Translation Bridge
        t_bridge_0 = time.perf_counter()
        if self.bridge is not None and len(raw_token_ids) > 0:
            # Note: decode_greedy in recognizer already did repeat collapsing on IDs;
            # bridge maps to sign strings cleanly
            gloss_sequence = self.sign_vocab.decode(raw_token_ids, remove_blank=True)
        elif self.sign_vocab is not None and len(raw_token_ids) > 0:
            gloss_sequence = self.sign_vocab.decode(raw_token_ids, remove_blank=True)

        diagnostics.num_glosses_recognized = len(gloss_sequence)
        diagnostics.bridge_latency_ms = (time.perf_counter() - t_bridge_0) * 1000.0

        # 5. Stage 5: Gloss-to-English Neural Translation
        t_trans_0 = time.perf_counter()
        translation_text = ""
        if self.translator is not None and len(gloss_sequence) > 0:
            translation_text = self.translator.translate_glosses(gloss_sequence)
        elif len(gloss_sequence) > 0:
            translation_text = " ".join(g.lower() for g in gloss_sequence)

        diagnostics.translation_latency_ms = (time.perf_counter() - t_trans_0) * 1000.0

        # Total Latency & Throughput
        t_total = time.perf_counter() - t_start
        diagnostics.total_pipeline_latency_ms = t_total * 1000.0
        diagnostics.throughput_fps = (num_frames / t_total) if t_total > 0 else 0.0

        return TranslationResult(
            glosses=gloss_sequence,
            translation=translation_text,
            confidence=confidence,
            stage_diagnostics=diagnostics,
            is_synthetic_evaluation=self.is_synthetic_evaluation,
            disclaimer=(
                "SYNTHETIC FIXTURE VALIDATION"
                if self.is_synthetic_evaluation
                else "REAL FEATURE PIPELINE DIAGNOSTICS"
            ),
        )

    def get_status(self) -> Dict[str, Any]:
        """Return pipeline health and component configuration."""
        return {
            "module": "EndToEndSignTranslationPipeline",
            "device": self.device,
            "has_recognizer": self.recognizer is not None,
            "has_translator": self.translator is not None,
            "has_sign_vocab": self.sign_vocab is not None,
            "landmark_group": self.landmark_group.value,
            "is_synthetic_evaluation": self.is_synthetic_evaluation,
        }
