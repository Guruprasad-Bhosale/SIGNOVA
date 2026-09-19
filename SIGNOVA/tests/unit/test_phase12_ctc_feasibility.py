"""
Phase 12 CTC Feasibility Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.feasibility import validate_ctc_feasibility


def test_ctc_feasibility_passed():
    annots = [
        VideoAnnotation(
            annotation_id="a1",
            sample_id="s1",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["A", "B", "C"],
            metadata={"frame_count": 60},  # 60 >= 3
        )
    ]
    report = validate_ctc_feasibility(annots)
    assert report.valid_samples == 1
    assert report.rejected_samples == 0
    assert report.min_input_length == 60
    assert report.min_target_length == 3
    assert report.feasibility_status == "PASSED"


def test_ctc_feasibility_length_violation():
    annots = [
        VideoAnnotation(
            annotation_id="a2",
            sample_id="s2",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["A", "B", "C", "D", "E"],
            metadata={"frame_count": 3},  # 3 < 5 -> Violation
        )
    ]
    report = validate_ctc_feasibility(annots)
    assert report.valid_samples == 0
    assert report.rejected_samples == 1
    assert "CTC_LENGTH_VIOLATION" in report.rejected_details[0]["reason"]
    assert report.feasibility_status == "WARNING"
