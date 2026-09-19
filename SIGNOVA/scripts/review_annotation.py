"""
Annotation Review Decision CLI for SIGNOVA Phase 22.

Applies VERIFIED, REJECTED, or REVISION_REQUIRED decisions.
"""

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.annotation.constants import REVIEW_STATE_REJECTED, REVIEW_STATE_REVISION_REQUIRED, REVIEW_STATE_VERIFIED
from signova.operations.phase22_orchestrator import Phase22Orchestrator


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 Annotation Review")
    parser.add_argument("--id", type=str, required=True, help="Annotation ID to review")
    parser.add_argument("--decision", type=str, required=True, choices=[REVIEW_STATE_VERIFIED, REVIEW_STATE_REJECTED, REVIEW_STATE_REVISION_REQUIRED], help="Review decision")
    parser.add_argument("--reviewer-id", type=str, required=True, help="Reviewer ID")
    parser.add_argument("--notes", type=str, default="", help="Review notes")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- ANNOTATION REVIEW WORKFLOW")
    print("===========================================================")

    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    try:
        ann = orch.review_annotation(
            annotation_id=args.id,
            decision=args.decision,
            reviewer_id=args.reviewer_id,
            review_notes=args.notes,
        )
        print(f"\n[+] Review Applied Successfully:")
        print(f"  Annotation ID:     {ann.annotation_id}")
        print(f"  New Review State:  {ann.review_state}")
        print(f"  Reviewer ID:       {ann.reviewer_id}")
        print(f"  Reviewed At:       {ann.reviewed_at}")
        print(f"  Notes:             {ann.review_notes or 'None'}")
    except FileNotFoundError as fe:
        print(f"\n[!] Error: {fe}")

    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
