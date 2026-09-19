#!/usr/bin/env python3
"""
SIGNOVA Evaluation Entrypoint (Placeholder for Phase 0)
"""

import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    print("=" * 60)
    print("               SIGNOVA EVALUATION PIPELINE                ")
    print("=" * 60)
    print("STATUS: PHASE 0 (FOUNDATION)")
    print("Planned Metrics (Phase 11):")
    print("  - Word Error Rate (WER) & Character Error Rate (CER)")
    print("  - BLEU-1/2/3/4 & ROUGE-L Translation Scores")
    print("  - End-to-End Latency (ms) & Frames Per Second (FPS)")
    print("=" * 60)


if __name__ == "__main__":
    main()
