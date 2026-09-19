"""
Unit Tests for Phase 4 CTC Architecture, Decoder, and Data Validator.
"""

from pathlib import Path
import pandas as pd
import pytest
import torch

from scripts.check_ctc_data import validate_ctc_data
from signova.models.ctc_recognizer import CTCContinuousRecognizer
from signova.recognition.ctc_decoder import CTCDecoder


def test_ctc_recognizer_forward_and_loss():
    B, T, num_joints = 2, 40, 75
    model = CTCContinuousRecognizer(num_landmarks=num_joints, num_classes=11, backbone="gru")
    model.eval()

    feat = torch.randn((B, T, num_joints, 3), dtype=torch.float32)
    mask = torch.ones((B, T), dtype=torch.bool)
    lens = torch.tensor([T, T], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    tgt_lens = torch.tensor([3, 2], dtype=torch.long)

    out = model(feat, padding_mask=mask, lengths=lens, targets=targets, target_lengths=tgt_lens)
    assert "logits" in out
    assert out["logits"].shape == (B, T, 11)
    assert out["loss"] is not None
    assert out["loss"].item() > 0.0


def test_ctc_recognizer_input_length_validation():
    B, T, num_joints = 1, 2, 75  # Input sequence only 2 frames
    model = CTCContinuousRecognizer(num_landmarks=num_joints, num_classes=11)
    feat = torch.randn((B, T, num_joints, 3), dtype=torch.float32)
    lens = torch.tensor([2], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)  # Target has 4 tokens
    tgt_lens = torch.tensor([4], dtype=torch.long)

    # Must raise ValueError when T_in < T_target
    with pytest.raises(ValueError, match="CTC constraint violation"):
        model(feat, lengths=lens, targets=targets, target_lengths=tgt_lens)


def test_ctc_decoder_greedy_and_token_error_rate():
    decoder = CTCDecoder(blank_idx=0, id_to_label={1: "HELLO", 2: "WORLD", 3: "SIGN"})

    # Simulate log probabilities for a sequence: [0, 1, 1, 0, 2, 2, 0, 3] -> collapses to [1, 2, 3]
    T, C = 8, 4
    logits = torch.zeros((1, T, C))
    indices = [0, 1, 1, 0, 2, 2, 0, 3]
    for t, idx in enumerate(indices):
        logits[0, t, idx] = 10.0

    decoded = decoder.decode_greedy_tensor(logits)
    assert len(decoded) == 1
    assert decoded[0]["tokens"] == [1, 2, 3]
    assert decoded[0]["labels"] == ["HELLO", "WORLD", "SIGN"]

    # Test Token Error Rate
    ref = [1, 2, 3]
    hyp_exact = [1, 2, 3]
    res_exact = decoder.compute_token_error_rate(ref, hyp_exact)
    assert res_exact["exact_match"] is True
    assert res_exact["token_error_rate"] == 0.0

    hyp_sub = [1, 4, 3]
    res_sub = decoder.compute_token_error_rate(ref, hyp_sub)
    assert res_sub["exact_match"] is False
    assert res_sub["substitutions"] == 1
    assert round(res_sub["token_error_rate"], 2) == 0.33


def test_check_ctc_data_validator_detects_missing_supervision(tmp_path: Path):
    manifest_csv = tmp_path / "test_unsupervised_manifest.csv"
    df = pd.DataFrame([
        {"sample_id": "s1", "feature_path": "fake.npz", "split": "train"}
    ])
    df.to_csv(manifest_csv, index=False)

    report = validate_ctc_data(manifest_csv)
    assert report["passed"] is False
    assert any("SUPERVISION MISSING" in v for v in report["violations"])
