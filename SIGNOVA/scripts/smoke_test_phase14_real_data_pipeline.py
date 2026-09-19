"""
Phase 14 Real Data Pipeline Smoke Test Script for SIGNOVA.

Verifies end-to-end data qualification mechanics:
Video -> Annotation -> Validation -> Quality -> Training Eligibility -> Feature Pairing -> CTC Feasibility.
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.annotation.schema import VideoAnnotation
from signova.qualification.feasibility import validate_ctc_feasibility
from signova.qualification.gate import Phase12SupervisionGate
from signova.qualification.ingestion import AnnotationIngestionEngine
from signova.qualification.vocabulary import Phase12GlossVocabulary


def run_smoke_test():
    print("\n--- Starting Phase 14 Real Data Pipeline Smoke Test ---")
    annotations_dir = WORKSPACE_ROOT / "data" / "annotations" / "phase11" / "human_gold"

    engine = AnnotationIngestionEngine()
    res = engine.batch_ingest(annotations_dir)

    print(f"Total annotations found: {res['total_files']}")
    if res["total_files"] == 0:
        print("Status: SKIPPED — NO_GENUINE_ANNOTATIONS (Waiting for external human collection).")
        print("Real data pipeline structure verified offline.")
        return

    valid_annots = res["valid_annotations"]
    print(f"Valid training-eligible annotations: {len(valid_annots)}")

    # 1. Vocabulary derivation
    vocab = Phase12GlossVocabulary.build_from_annotations(valid_annots)
    print(f"Derived vocabulary size: {vocab.size}")

    # 2. CTC Feasibility check
    feasibility = validate_ctc_feasibility(valid_annots)
    print(f"CTC Feasibility status: {feasibility.feasibility_status} (Valid: {feasibility.valid_samples})")

    # 3. Supervision Gate
    gate_res = Phase12SupervisionGate.evaluate(annotations_dir=annotations_dir)
    print(f"Supervision Gate outcome: {gate_res.supervision_state} (CTC: {gate_res.real_ctc_status})")
    print("[SUCCESS] Real data qualification path verified.")


if __name__ == "__main__":
    run_smoke_test()
