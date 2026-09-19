"""
Phase 18 Split Strategy Selection & Fallback Warning Unit Tests.

Validates:
- Hierarchy: SIGNER_INDEPENDENT -> SESSION_INDEPENDENT -> SOURCE_GROUP_INDEPENDENT -> RANDOM
- Explicit rationale reporting.
- Explicit warning generation and high leakage risk when falling back to RANDOM split.
"""

from signova.annotation.schema import VideoAnnotation
from signova.operations.phase18_orchestrator import Phase18Orchestrator
from signova.pilot.constants import (
    SPLIT_STRATEGY_RANDOM,
    SPLIT_STRATEGY_SESSION_INDEPENDENT,
    SPLIT_STRATEGY_SIGNER_INDEPENDENT,
)


def test_signer_independent_split_selected_when_multiple_signers_present():
    ann1 = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="vid_01",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        glosses=["HELLO"],
        provenance_id="prov_01",
        metadata={"signer_id": "signer_1"},
    )
    ann2 = VideoAnnotation(
        annotation_id="ann_02",
        sample_id="vid_02",
        annotator_id="ann_02",
        is_temporally_aligned=True,
        glosses=["WORLD"],
        provenance_id="prov_02",
        metadata={"signer_id": "signer_2"},
    )
    orch = Phase18Orchestrator()
    strat, reason, warning, id_avail, risk = orch._select_split_strategy_with_warning([ann1, ann2])
    assert strat == SPLIT_STRATEGY_SIGNER_INDEPENDENT
    assert "verified_signer_metadata_available" in reason
    assert warning is None
    assert id_avail is True
    assert risk == "MINIMAL"


def test_session_independent_split_selected_when_signers_unavailable():
    ann1 = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="vid_01",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        glosses=["HELLO"],
        provenance_id="prov_01",
        metadata={"session_id": "session_1"},
    )
    ann2 = VideoAnnotation(
        annotation_id="ann_02",
        sample_id="vid_02",
        annotator_id="ann_02",
        is_temporally_aligned=True,
        glosses=["WORLD"],
        provenance_id="prov_02",
        metadata={"session_id": "session_2"},
    )
    orch = Phase18Orchestrator()
    strat, reason, warning, id_avail, risk = orch._select_split_strategy_with_warning([ann1, ann2])
    assert strat == SPLIT_STRATEGY_SESSION_INDEPENDENT
    assert id_avail is True


def test_random_split_emits_explicit_warning():
    ann1 = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="vid_01",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        glosses=["HELLO"],
        provenance_id="prov_01",
    )
    orch = Phase18Orchestrator()
    strat, reason, warning, id_avail, risk = orch._select_split_strategy_with_warning([ann1])
    assert strat == SPLIT_STRATEGY_RANDOM
    assert warning is not None
    assert "WARNING" in warning
    assert risk == "HIGH"
