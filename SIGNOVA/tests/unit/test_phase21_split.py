"""
Phase 21 Strict Split Hierarchy & Random Guard Tests.
"""

import pytest
from signova.operations.phase21_orchestrator import Phase21DatasetSample, Phase21Orchestrator
from signova.pilot.constants import (
    SPLIT_STRATEGY_RANDOM,
    SPLIT_STRATEGY_SESSION_INDEPENDENT,
    SPLIT_STRATEGY_SIGNER_INDEPENDENT,
)


def _make_sample(sid: str, signer: str, sess: str) -> Phase21DatasetSample:
    return Phase21DatasetSample(
        sample_id=sid,
        video_id=f"v_{sid}",
        video_sha256="SHA",
        annotation_id=f"a_{sid}",
        annotator_id="u1",
        reviewer_id="r1",
        session_id=sess,
        signer_id=signer,
        gloss_sequence=["NAMASTE"],
        feature_path="f.npz",
    )


def test_signer_independent_split_selected_when_signers_present():
    samples = [
        _make_sample("s1", "signer_1", "sess_1"),
        _make_sample("s2", "signer_2", "sess_2"),
        _make_sample("s3", "signer_3", "sess_3"),
        _make_sample("s4", "signer_4", "sess_4"),
    ]
    orch = Phase21Orchestrator()
    splits = orch._resolve_splits(samples)
    assert splits["strategy"] == SPLIT_STRATEGY_SIGNER_INDEPENDENT
    assert splits["status"] == "RESEARCH_GRADE"


def test_session_independent_split_selected_when_sessions_present():
    samples = [
        _make_sample("s1", "signer_default", "sess_1"),
        _make_sample("s2", "signer_default", "sess_2"),
        _make_sample("s3", "signer_default", "sess_3"),
        _make_sample("s4", "signer_default", "sess_4"),
    ]
    orch = Phase21Orchestrator()
    splits = orch._resolve_splits(samples)
    assert splits["strategy"] == SPLIT_STRATEGY_SESSION_INDEPENDENT


def test_random_split_requires_explicit_flag_and_warns():
    samples = [
        _make_sample("s1", "signer_default", "sess_default"),
        _make_sample("s2", "signer_default", "sess_default"),
        _make_sample("s3", "signer_default", "sess_default"),
        _make_sample("s4", "signer_default", "sess_default"),
    ]
    orch = Phase21Orchestrator()
    
    # Must raise ValueError when allow_random_split is False
    with pytest.raises(ValueError, match="INDEPENDENT SPLIT UNAVAILABLE"):
        orch._resolve_splits(samples, allow_random_split=False)

    # Succeeds when allow_random_split is True
    splits = orch._resolve_splits(samples, allow_random_split=True)
    assert splits["strategy"] == SPLIT_STRATEGY_RANDOM
    assert "WARNING" in splits["warning"]
    assert splits["status"] == "LIMITED"
