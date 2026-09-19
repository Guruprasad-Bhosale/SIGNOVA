"""
Phase 24 Dynamic Input Spec Compatibility Tests.
"""

from signova.operations.phase24_orchestrator import Phase24Orchestrator


def test_dynamic_compatibility_verification(tmp_path):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")

    # 1. Missing manifest
    res_none = orch.verify_dataset_model_compatibility(None)
    assert res_none["compatible"] is False
    assert "MISSING_DATASET_MANIFEST" in res_none["reasons"]

    # 2. Valid manifest matching canonical spec
    valid_manifest = {
        "dataset_sha256": "SHA_VALID",
        "vocabulary_size": 12,
        "samples": [{"sample_id": "s1"}],
    }
    res_valid = orch.verify_dataset_model_compatibility(valid_manifest)
    assert res_valid["compatible"] is True
    assert res_valid["reasons"] == ["PASSED"]
    assert res_valid["input_spec"]["landmark_topology"] == 543
    assert res_valid["input_spec"]["feature_group"] == "HANDS_POSE"
