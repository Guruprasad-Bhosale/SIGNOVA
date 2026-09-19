"""
Human Annotation Import CLI for SIGNOVA Phase 22.

Imports a genuine human annotation JSON record with source SHA-256 and schema validation.
Strictly requires annotation_source == HUMAN_DIRECT.
"""

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import Phase22Orchestrator, SOURCE_HUMAN_DIRECT


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 Annotation Import")
    parser.add_argument("--json-file", type=str, required=True, help="Path to annotation JSON file")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- HUMAN ANNOTATION IMPORT")
    print("===========================================================")

    f_path = Path(args.json_file)
    if not f_path.exists():
        print(f"\n[!] File not found: {f_path}")
        return

    try:
        data = json.loads(f_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"\n[!] Failed to parse JSON: {e}")
        return

    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    try:
        ann = orch.import_annotation(data)
        print(f"\n[+] Annotation Imported Successfully:")
        print(f"  Annotation ID:     {ann.annotation_id}")
        print(f"  Revision ID:       {ann.revision_id}")
        print(f"  Video ID:          {ann.video_id}")
        print(f"  Annotator ID:      {ann.annotator_id}")
        print(f"  Annotation Source: {ann.annotation_source}")
        print(f"  Gloss Tokens:      {' '.join(ann.gloss_tokens)}")
        print(f"  Review State:      {ann.review_state}")
    except ValueError as ve:
        print(f"\n[!] Import Rejected: {ve}")

    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
