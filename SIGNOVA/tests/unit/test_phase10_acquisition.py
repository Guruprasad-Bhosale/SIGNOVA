"""
Unit tests for Phase 10 Controlled Dataset Acquisition Engine (100% Mocked/Offline).
"""

from pathlib import Path
import pytest
from signova.data.acquisition import (
    AcquisitionRecord,
    AcquisitionStatus,
    ControlledAcquisitionEngine,
)


def test_controlled_acquisition_zero_cost_rejection(tmp_path):
    engine = ControlledAcquisitionEngine(raw_storage_dir=tmp_path / "raw")

    # Paid candidate must be rejected immediately
    record = engine.evaluate_and_acquire(
        dataset_id="paid_dataset",
        candidate_id="cand_paid",
        source_url="http://mock.local/paid",
        source_name="Paid Vendor",
        license_str="Commercial Proprietary",
        cost_inr=10000.0,
    )
    assert record.acquisition_status == AcquisitionStatus.REJECTED.value
    assert record.rejection_reason == "PAID_NOT_USED"
    assert record.cost_status == "PAID"


def test_controlled_acquisition_license_unclear(tmp_path):
    engine = ControlledAcquisitionEngine(raw_storage_dir=tmp_path / "raw")

    record = engine.evaluate_and_acquire(
        dataset_id="unclear_dataset",
        candidate_id="cand_unclear",
        source_url="http://mock.local/unclear",
        source_name="Unknown Archive",
        license_str="License Unclear / No Terms",
        cost_inr=0.0,
    )
    assert record.acquisition_status == AcquisitionStatus.REJECTED.value
    assert record.rejection_reason == "LICENSE_UNCLEAR"


def test_controlled_acquisition_no_eligible_dataset(tmp_path):
    engine = ControlledAcquisitionEngine(raw_storage_dir=tmp_path / "raw")

    record = engine.evaluate_and_acquire(
        dataset_id="isl_csltr",
        candidate_id="cand_01",
        source_url="http://mock.local/isl_csltr",
        source_name="Mendeley Data",
        license_str="CC BY 4.0",
        cost_inr=0.0,
        has_ordered_glosses=False,
    )
    assert record.acquisition_status == AcquisitionStatus.NO_ELIGIBLE_DATASET.value
    assert record.rejection_reason == "NO_ORDERED_GLOSS"


def test_controlled_acquisition_offline_mock_artifact(tmp_path):
    engine = ControlledAcquisitionEngine(raw_storage_dir=tmp_path / "raw")

    # Create a local mock artifact file
    mock_file = tmp_path / "sample_mock.mp4"
    mock_file.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")

    record = engine.evaluate_and_acquire(
        dataset_id="mock_isl",
        candidate_id="cand_mock",
        source_url="http://mock.local/video",
        source_name="Mock Local",
        license_str="CC BY 4.0",
        cost_inr=0.0,
        local_mock_artifact=mock_file,
        has_ordered_glosses=True,
    )
    assert record.acquisition_status == AcquisitionStatus.ACQUIRED.value
    assert record.verification_status == "SHA256_VERIFIED"
    assert len(record.artifact_sha256) == 64
    assert Path(record.artifact_path).exists()
