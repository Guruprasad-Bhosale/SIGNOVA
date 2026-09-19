#!/usr/bin/env python3
"""
SIGNOVA Training Entrypoint (Placeholder for Phase 0)
"""

import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from signova.config.loader import ConfigManager


def main():
    print("=" * 60)
    print("                SIGNOVA TRAINING PIPELINE                 ")
    print("=" * 60)
    print("STATUS: PHASE 0 (FOUNDATION)")
    print("Notice: Model training is intentionally disabled in Phase 0.")
    print("Model architectures will be benchmarked in Phase 3/4 and trained in Phase 5.")
    
    cm = ConfigManager()
    cfg = cm.load_all()
    print(f"\nLoaded Project Config:")
    print(f"  Project Name: {cfg.base.project.name} (v{cfg.base.project.version})")
    print(f"  Target GPU  : NVIDIA RTX 3050 (Batch size: {cfg.training.training.get('batch_size')}, Mixed Precision: {cfg.training.training.get('mixed_precision')})")
    print("=" * 60)


if __name__ == "__main__":
    main()
