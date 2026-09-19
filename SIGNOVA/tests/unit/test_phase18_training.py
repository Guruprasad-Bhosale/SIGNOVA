"""
Phase 18 Real CTC Training Gating & Artifact Protection Unit Tests.

Validates:
- Real CTC training is strictly blocked under STATE_B.
- No real model checkpoint or optimizer state is created under STATE_B.
- Dataset fingerprinting and provenance hash computation on qualified data.
"""

from pathlib import Path
from signova.annotation.schema import VideoAnnotation
from signova.operations.checkpoint_provenance import (
    CHECKPOINT_TYPE_REAL,
    CHECKPOINT_TYPE_SYNTHETIC,
    create_real_checkpoint_provenance,
)
from signova.operations.phase18_orchestrator import (
    Phase18Orchestrator,
    evaluate_phase18_readiness,
)


def test_real_training_checkpoint_blocked_under_state_b(tmp_path):
    res = evaluate_phase18_readiness(workspace_root=tmp_path)
    assert res["real_ctc_training_allowed"] is False
    assert res["real_checkpoint_created"] is False


def test_dataset_fingerprint_generation_for_qualified_data():
    ann1 = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="vid_01",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        provenance_id="prov_01",
        glosses=["NAMASTE"],
        metadata={"source_checksum": "CHECKSUM_01"},
    )
    orch = Phase18Orchestrator()
    fp = orch._compute_dataset_fingerprint([ann1])
    assert "dataset_sha256" in fp
    assert len(fp["dataset_sha256"]) == 64
    assert fp["sample_count"] == 1
    assert fp["ordered_annotation_ids"] == ["ann_01"]
