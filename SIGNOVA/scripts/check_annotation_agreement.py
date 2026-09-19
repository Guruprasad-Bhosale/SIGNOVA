"""
Double-Annotation Inter-Annotator Agreement CLI for SIGNOVA Phase 22.
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import Phase22Orchestrator


def main():
    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- DOUBLE ANNOTATION AGREEMENT")
    print("===========================================================")

    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    agr = orch.evaluate_double_annotation_agreement()

    print(f"\n[+] Double Annotation Analysis:")
    print(f"  Agreement Status:      {agr['status']}")
    print(f"  Double Annotated Pairs:{agr['double_annotated_pairs']}")
    print(f"  Computable Pairs:      {agr['computable_pairs']}")

    if agr['status'] == 'COMPUTABLE':
        print(f"  Exact Sequence Agr:    {agr['exact_sequence_agreement'] * 100:.2f}%")
        print(f"  Token-Level Agreement: {agr['token_level_agreement'] * 100:.2f}%")
    else:
        print(f"  Message:               {agr.get('message')}")

    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
