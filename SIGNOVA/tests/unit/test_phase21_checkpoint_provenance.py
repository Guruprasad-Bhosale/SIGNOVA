"""
Phase 21 Checkpoint Provenance and Cryptographic Verification Tests.
"""

import hashlib
import json
from pathlib import Path
import torch

from signova.operations.phase21_orchestrator import Phase21Orchestrator


def test_provenance_verification_success(tmp_path):
    run_dir = tmp_path / "run_20260919_test"
    run_dir.mkdir(parents=True)

    ckpt_path = run_dir / "checkpoint.pt"
    torch.save({"num_classes": 10}, ckpt_path)
    calc_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest().upper()

    metadata = {
        "model_id": "test_m1",
        "checkpoint_sha256": calc_sha,
        "input_spec": {
            "feature_group": "HANDS_POSE",
            "landmark_topology": 543,
            "temporal_window": 64,
            "temporal_stride": 16,
            "normalization_version": "1.0.0",
        },
        "dataset_fingerprint": {"dataset_sha256": "DSHA", "sample_count": 5},
    }
    (run_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(json.dumps({"ter": 0.1, "exact_match": 0.8}), encoding="utf-8")

    orch = Phase21Orchestrator(workspace_root=tmp_path, experiments_dir=tmp_path)
    res = orch.evaluate_live_authorization_pipeline(run_dir=run_dir)

    assert res["stages"]["trained"] is True
    assert res["stages"]["checkpoint_verified"] is True
    assert res["stages"]["held_out_evaluation"] is True
    assert res["stages"]["input_spec_match"] is True
    assert res["stages"]["live_smoke_test"] is True
    assert res["live_model_authorized"] is True


def test_provenance_verification_fails_on_tampered_checkpoint(tmp_path):
    run_dir = tmp_path / "run_20260919_tampered"
    run_dir.mkdir(parents=True)

    ckpt_path = run_dir / "checkpoint.pt"
    torch.save({"num_classes": 10}, ckpt_path)

    # Put a wrong hash in metadata
    metadata = {
        "model_id": "test_m1",
        "checkpoint_sha256": "WRONG_SHA_HASH",
        "input_spec": {"feature_group": "HANDS_POSE", "landmark_topology": 543, "temporal_window": 64},
    }
    (run_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    orch = Phase21Orchestrator(workspace_root=tmp_path, experiments_dir=tmp_path)
    res = orch.evaluate_live_authorization_pipeline(run_dir=run_dir)

    assert res["stages"]["checkpoint_verified"] is False
    assert res["live_model_authorized"] is False
