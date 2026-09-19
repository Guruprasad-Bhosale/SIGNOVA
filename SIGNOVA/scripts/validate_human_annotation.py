"""
Annotation Quality and Training Eligibility Validation CLI for SIGNOVA Phase 22.
"""

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import (
    HumanAnnotationRecord,
    Phase22Orchestrator,
    evaluate_training_eligibility,
)


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 Annotation Validation")
    parser.add_argument("--id", type=str, required=True, help="Annotation ID to validate")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- ANNOTATION VALIDATION")
    print("===========================================================")

    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    ann_file = orch.annotations_dir / f"{args.id}.json"

    if not ann_file.exists():
        print(f"\n[!] Annotation record '{args.id}' not found in {orch.annotations_dir}")
        print("\n===========================================================\n")
        return

    data = json.loads(ann_file.read_text(encoding="utf-8"))
    ann = HumanAnnotationRecord.from_dict(data)
    prof = orch.load_annotator_profile(ann.annotator_id)
    v_path = WORKSPACE_ROOT / ann.source_uri if ann.source_uri else None

    el_res = evaluate_training_eligibility(ann, prof, v_path)

    print(f"\n[+] Validation Report for [{ann.annotation_id}]:")
    print(f"  Revision:          {ann.revision_id} (Rev #{ann.revision_number})")
    print(f"  Video ID:          {ann.video_id}")
    print(f"  Annotator ID:      {ann.annotator_id}")
    print(f"  Annotation Source: {ann.annotation_source}")
    print(f"  Review State:      {ann.review_state}")
    print(f"  Gloss Tokens:      {' '.join(ann.gloss_tokens)}")
    print(f"  Training Eligible: {'YES' if el_res['eligible'] else 'NO'}")
    if el_res["reasons"]:
        print(f"  Blockers:          {', '.join(el_res['reasons'])}")

    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
