"""
Unit tests for EndToEndSignTranslationPipeline (Stage diagnostics, full landmark processing, edge cases).
"""

import numpy as np
import torch

from signova.data.vocabulary import SignVocabulary
from signova.features.feature_groups import LandmarkGroup
from signova.inference.pipeline import EndToEndSignTranslationPipeline
from signova.models.ctc_recognizer import CTCContinuousRecognizer
from signova.translation.model import SyntheticGlossToEnglishSeq2Seq
from signova.translation.translator import GlossToEnglishTranslator
from signova.translation.vocabulary import TranslationVocabulary


def test_end_to_end_pipeline_processing():
    # 1. Vocabularies
    sign_classes = ["<BLANK>", "I", "GO", "COLLEGE"]
    sign_vocab = SignVocabulary(tokens=sign_classes)

    src_trans_vocab = TranslationVocabulary().build_from_sequences([["I", "GO", "COLLEGE"]])
    tgt_trans_vocab = TranslationVocabulary().build_from_sequences([["i", "am", "going", "to", "college"]])

    # 2. Recognizer and Translator models
    recognizer = CTCContinuousRecognizer(
        num_landmarks=75,
        num_classes=len(sign_classes),
        backbone="gru",
        projection_dim=32,
        hidden_size=32,
        num_layers=1,
    )
    recognizer.eval()

    trans_model = SyntheticGlossToEnglishSeq2Seq(
        src_vocab_size=len(src_trans_vocab),
        tgt_vocab_size=len(tgt_trans_vocab),
        src_embed_dim=16,
        tgt_embed_dim=16,
        enc_hidden_dim=32,
        dec_hidden_dim=32,
    )
    trans_model.eval()

    translator = GlossToEnglishTranslator(
        model=trans_model,
        source_vocab=src_trans_vocab,
        target_vocab=tgt_trans_vocab,
    )

    # 3. Pipeline
    pipeline = EndToEndSignTranslationPipeline(
        recognizer=recognizer,
        sign_vocab=sign_vocab,
        translator=translator,
        landmark_group=LandmarkGroup.HANDS_POSE,
        is_synthetic_evaluation=True,
    )

    assert pipeline.get_status()["has_recognizer"] is True
    assert pipeline.get_status()["has_translator"] is True

    # 4. Process dummy landmarks (T=20, 543, 3)
    dummy_landmarks = np.random.randn(20, 543, 3).astype(np.float32)
    result = pipeline.process_landmarks(dummy_landmarks, fps=30.0)

    assert isinstance(result.glosses, list)
    assert isinstance(result.translation, str)
    assert result.stage_diagnostics is not None
    assert result.stage_diagnostics.num_frames_processed == 20
    assert result.stage_diagnostics.total_pipeline_latency_ms >= 0.0
    assert result.stage_diagnostics.throughput_fps >= 0.0


def test_end_to_end_pipeline_empty_and_edge_cases():
    pipeline = EndToEndSignTranslationPipeline(is_synthetic_evaluation=True)

    # Empty array
    empty_landmarks = np.empty((0, 543, 3), dtype=np.float32)
    result_empty = pipeline.process_landmarks(empty_landmarks)
    assert result_empty.glosses == []
    assert result_empty.translation == ""
    assert result_empty.stage_diagnostics.num_frames_processed == 0
