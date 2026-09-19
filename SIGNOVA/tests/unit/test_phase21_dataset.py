"""
Phase 21 Dataset Construction and Fingerprinting Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.operations.phase21_orchestrator import (
    Phase21DatasetSample,
    Phase21Orchestrator,
)
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_dataset_sample_schema():
    sample = Phase21DatasetSample(
        sample_id="s1",
        video_id="v1",
        video_sha256="SHA123",
        annotation_id="a1",
        annotator_id="ann_01",
        reviewer_id="rev_01",
        session_id="sess_01",
        signer_id="signer_01",
        gloss_sequence=["NAMASTE", "HELP"],
        feature_path="path/to/feat.npz",
    )
    d = sample.to_dict()
    assert d["sample_id"] == "s1"
    assert d["gloss_sequence"] == ["NAMASTE", "HELP"]
    assert d["dataset_version"] == "21.0.0"


def test_dataset_fingerprint_deterministic():
    s1 = Phase21DatasetSample(
        sample_id="s1",
        video_id="v1",
        video_sha256="SHA1",
        annotation_id="a1",
        annotator_id="ann1",
        reviewer_id="rev1",
        session_id="sess1",
        signer_id="sig1",
        gloss_sequence=["HELLO"],
        feature_path="f1.npz",
    )
    vocab = Phase12GlossVocabulary()
    splits = {"strategy": "RANDOM"}
    orch = Phase21Orchestrator()
    fp1 = orch._compute_dataset_fingerprint([s1], vocab, splits)
    fp2 = orch._compute_dataset_fingerprint([s1], vocab, splits)

    assert fp1["dataset_sha256"] == fp2["dataset_sha256"]
    assert fp1["sample_count"] == 1
