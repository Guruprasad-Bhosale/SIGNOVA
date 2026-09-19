"""
Unit tests for Phase 9 Dataset Acquisition Manifest, zero-cost mode blocking, and PAID_NOT_USED flags.
"""

from pathlib import Path
import pandas as pd
import pytest
from signova.data.provenance import ZERO_COST_MODE, ZeroCostPolicyEnforcer


def test_acquisition_manifest_structure():
    manifest_path = Path("data/manifests/phase9_dataset_acquisition.csv")
    assert manifest_path.is_file(), "phase9_dataset_acquisition.csv must exist"

    df = pd.read_csv(manifest_path)
    expected_cols = [
        "dataset_id",
        "source",
        "license",
        "download_method",
        "local_path",
        "checksum",
        "file_count",
        "size",
        "annotation_count",
        "video_count",
        "status",
    ]
    assert list(df.columns) == expected_cols


def test_zero_cost_policy_enforcement():
    enforcer = ZeroCostPolicyEnforcer(mode=ZERO_COST_MODE)
    assert enforcer.mode == "DEFAULT"

    # Free resource
    res_free = enforcer.evaluate_acquisition_request(
        dataset_id="isl-csltr",
        cost_required=False,
        cost_inr=0.0,
    )
    assert res_free["permitted"] is True
    assert res_free["status"] == "APPROVED_FREE"
    assert res_free["authorized_spend_inr"] == 0.0

    # Paid resource must be blocked
    res_paid = enforcer.evaluate_acquisition_request(
        dataset_id="commercial-signs",
        cost_required=True,
        cost_inr=5000.0,
    )
    assert res_paid["permitted"] is False
    assert res_paid["status"] == "PAID_NOT_USED"
    assert res_paid["authorized_spend_inr"] == 0.0
