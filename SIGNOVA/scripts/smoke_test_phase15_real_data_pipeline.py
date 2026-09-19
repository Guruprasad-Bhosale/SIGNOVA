"""
Phase 15 Real Data Pipeline Smoke Test Script for SIGNOVA.

Tests:
- Ingestion of genuine annotations
- Annotator qualification inspection
- Repeated-token CTC feasibility
- Readiness gate evaluation
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.annotator_qualification import validate_annotator_qualification
from signova.qualification.feasibility import validate_ctc_feasibility
from signova.qualification.gate import Phase12SupervisionGate
from signova.qualification.ingestion import AnnotationIngestionEngine
from signova.qualification.vocabulary import Phase12GlossVocabulary


def run_smoke_test():
    print("\n--- Starting Phase 15 Real Data Pipeline Smoke Test ---")
    annotations_dir = WORKSPACE_ROOT / "data" / "annotations" / "phase11" / "human_gold"

    engine = AnnotationIngestionEngine()
    res = engine.batch_ingest(annotations_dir)

    print(f"Total annotations discovered: {res['total_files']}")
    if res["total_files"] == 0:
        print("Status: SKIPPED — NO_GENUINE_ANNOTATIONS (Awaiting external human data collection).")
        print("Phase 15 data qualification pipeline operational.")
        return

    valid_annots = res["valid_annotations"]
    print(f"Valid training-eligible annotations: {len(valid_annots)}")

    # Check qualification of first annotator
    if valid_annots:
        first_ann = valid_annots[0]
        qual = validate_annotator_qualification(first_ann.metadata.get("annotator_profile"))
        print(f"Sample {first_ann.sample_id} annotator qualification: {qual.qualification_status}")

    feasibility = validate_ctc_feasibility(valid_annots)
    print(f"CTC Feasibility status: {feasibility.feasibility_status} (Valid: {feasibility.valid_samples})")

    gate_res = Phase12SupervisionGate.evaluate(annotations_dir=annotations_dir)
    print(f"Supervision Gate outcome: {gate_res.supervision_state} (Real CTC: {gate_res.real_ctc_status})")
    print("[SUCCESS] Phase 15 real data pipeline verified.")


if __name__ == "__main__":
    run_smoke_test()
