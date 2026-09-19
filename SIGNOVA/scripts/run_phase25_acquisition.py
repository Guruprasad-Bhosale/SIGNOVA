"""
scripts/run_phase25_acquisition.py
Controlled CLI workflow for human sequential ISL annotation intake, submission, and review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    HumanAnnotationRecord,
    Phase25Orchestrator,
    REVIEW_DRAFT,
    REVIEW_SUBMITTED,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 25 Human Annotation Acquisition Runner")
    parser.add_argument("--annotation-id", type=str, help="Unique annotation identifier")
    parser.add_argument("--video-id", type=str, help="Source video identifier")
    parser.add_argument("--video-path", type=str, help="Path to source video file")
    parser.add_argument("--source-sha256", type=str, help="SHA-256 hash of source video file")
    parser.add_argument("--annotator-id", type=str, help="Real annotator identifier")
    parser.add_argument("--glosses", nargs="+", help="Ordered sequence of ISL glosses")
    parser.add_argument("--submit", action="store_true", help="Submit annotation for review")
    parser.add_argument("--review", choices=["VERIFY", "REJECT"], help="Apply reviewer decision")
    parser.add_argument("--reviewer-id", type=str, help="Reviewer identifier")
    parser.add_argument("--reason", type=str, help="Rejection reason if applicable")
    parser.add_argument("--group-id", type=str, help="Double-annotation group ID")
    parser.add_argument("--index", type=int, default=1, help="Independent annotation index (1 or 2)")
    parser.add_argument("--data-root", type=str, default=None, help="Optional data root directory")
    args = parser.parse_args()

    orchestrator = Phase25Orchestrator(data_root=args.data_root)

    if args.review:
        if not args.annotation_id or not args.reviewer_id:
            print("[!] Error: --annotation-id and --reviewer-id are required for review.")
            sys.exit(1)
        res = orchestrator.review_annotation(
            annotation_id=args.annotation_id,
            action=args.review,
            reviewer_id=args.reviewer_id,
            reason=args.reason,
        )
        print(json.dumps(res, indent=2))
        return

    if args.annotation_id and args.video_id and args.annotator_id and args.glosses:
        # If source-sha256 is not provided but video-path is, calculate hash
        src_sha = args.source_sha256 or ""
        if not src_sha and args.video_path:
            chk = orchestrator.verify_source_video_integrity(args.video_path, "")
            src_sha = chk["current_sha256"]

        rec = HumanAnnotationRecord(
            annotation_id=args.annotation_id,
            source_video_id=args.video_id,
            source_sha256=src_sha,
            annotator_id=args.annotator_id,
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=args.glosses,
            review_state=REVIEW_SUBMITTED if args.submit else REVIEW_DRAFT,
            annotation_group_id=args.group_id,
            independent_annotation_index=args.index,
        )

        res = orchestrator.intake_annotation(rec)
        print("=================================================================")
        print(" SIGNOVA Phase 25 -- Human Annotation Intake")
        print("=================================================================")
        print(f"Status:            {res['status']}")
        print(f"Review State:      {res.get('review_state')}")
        print(f"Training Eligible: {res.get('training_eligible')}")
        if "reason" in res:
            print(f"Reason:            {res['reason']}")
        print("=================================================================")
        return

    # If invoked without args, display current acquisition status
    readiness = orchestrator.evaluate_readiness()
    print("=================================================================")
    print(" SIGNOVA Phase 25 -- Annotation Acquisition Status")
    print("=================================================================")
    print(f"Acquisition Status:     {readiness['acquisition_status']}")
    print(f"Total Annotations:      {readiness['human_annotations']['total']}")
    print(f"Submitted:              {readiness['human_annotations']['submitted']}")
    print(f"Verified:               {readiness['human_annotations']['verified']}")
    print(f"Training Eligible:      {readiness['human_annotations']['training_eligible']}")
    print("=================================================================")


if __name__ == "__main__":
    main()
