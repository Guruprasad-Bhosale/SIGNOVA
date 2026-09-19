#!/usr/bin/env python3
"""
SIGNOVA Inference CLI Entrypoint (Placeholder for Phase 0)
"""

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from signova.inference.pipeline import InferencePipelinePlaceholder


def main():
    parser = argparse.ArgumentParser(description="Run SIGNOVA Inference.")
    parser.add_argument("--video", type=str, required=False, help="Path to input video file.")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam stream.")
    args = parser.parse_args()

    print("=" * 60)
    print("                SIGNOVA INFERENCE PIPELINE                ")
    print("=" * 60)
    print("STATUS: PHASE 0 (FOUNDATION)")
    print("Notice: Real-time and offline inference will be fully enabled in Phase 8.")
    pipeline = InferencePipelinePlaceholder()
    print("Pipeline status:", pipeline.get_status())
    print("=" * 60)


if __name__ == "__main__":
    main()
