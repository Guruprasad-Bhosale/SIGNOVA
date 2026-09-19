"""
Phase 24 Immutable Experiment Dataset Lock & Mutation Detection Tests.
"""

from signova.operations.phase24_orchestrator import Phase24Orchestrator


def test_dataset_experiment_locking_and_mutation_detection(tmp_path):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")

    dummy_manifest = {
        "dataset_version": "phase23_dataset_v001",
        "dataset_sha256": "SHA256_ORIGINAL_MANIFEST",
        "vocabulary_size": 10,
        "vocabulary_sha256": "VOCAB_SHA_01",
        "splits": {
            "train_sample_ids": ["s1", "s2"],
            "val_sample_ids": ["s3"],
            "test_sample_ids": ["s4"],
        },
        "samples": [{"sample_id": "s1"}],
    }

    # Lock dataset for experiment
    lock = orch.lock_experiment_dataset("exp_001", dummy_manifest)
    assert lock.experiment_id == "exp_001"
    assert lock.dataset_sha256 == "SHA256_ORIGINAL_MANIFEST"
    assert lock.split_fingerprint != ""
    assert lock.input_spec_fingerprint != ""

    # Verify matching dataset passes lock check
    check_pass = orch.verify_dataset_lock("exp_001", dummy_manifest)
    assert check_pass["lock_valid"] is True

    # Mutated dataset fails lock check
    mutated_manifest = dict(dummy_manifest)
    mutated_manifest["dataset_sha256"] = "SHA256_MUTATED_MANIFEST"
    check_fail = orch.verify_dataset_lock("exp_001", mutated_manifest)
    assert check_fail["lock_valid"] is False
    assert check_fail["reason"] == "DATASET_MUTATION_DETECTED"
