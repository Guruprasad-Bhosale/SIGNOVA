#!/usr/bin/env python3
"""
SIGNOVA Phase 23 - Batch Human Data Ingestion.

Imports genuine human ISL annotation JSON files/directories through
Phase 22 validation rules without bypassing qualification or revision lineage.
"""

import sys
import json
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase23_orchestrator import Phase23Orchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 23 Batch Human Data Ingestion")
    parser.add_argument("source_path", help="Path to annotation JSON file or directory")
    parser.add_argument("--raw-videos-dir", help="Optional path to directory containing source MP4 video files for SHA-256 verification")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    orch = Phase23Orchestrator(workspace_root=WORKSPACE_ROOT, data_root=args.data_root)

    try:
        res = orch.ingest_batch_human_data(
            source_dir_or_file=args.source_path,
            raw_videos_dir=args.raw_videos_dir,
        )
    except Exception as e:
        print(f"[!] Error during batch ingestion: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print("=" * 65)
        print(" SIGNOVA Phase 23 -- Batch Ingestion Results")
        print("=" * 65)
        print(f"Files Discovered:       {res.files_discovered}")
        print(f"Annotations Parsed:     {res.annotations_parsed}")
        print(f"Annotations Valid:      {res.annotations_valid}")
        print(f"Annotations Invalid:    {res.annotations_invalid}")
        print(f"Source SHA Match:       {res.source_hash_match}")
        print(f"Source SHA Mismatch:    {res.source_hash_mismatch}")
        print(f"Qualified Annotators:   {res.qualified_annotators}")
        print(f"Unqualified Annotators: {res.unqualified_annotators}")
        print(f"Submitted:              {res.submitted}")
        print(f"Verified:               {res.verified}")
        print(f"Rejected:               {res.rejected}")
        print(f"Revision Required:      {res.revision_required}")
        print(f"Training Eligible:      {res.training_eligible}")
        print(f"Training Ineligible:    {res.training_ineligible}")
        print("-" * 65)
        if res.rejection_reasons:
            print("Rejection Breakdown:")
            for reason, count in res.rejection_reasons.items():
                print(f"  * {reason}: {count}")
        print("=" * 65)


if __name__ == "__main__":
    main()
