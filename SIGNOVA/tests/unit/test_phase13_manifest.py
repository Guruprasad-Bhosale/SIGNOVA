"""
Phase 13 Pilot Manifest Unit Tests.
"""

from pathlib import Path
from signova.pilot.pilot_manager import PilotDatasetManager


def test_generate_pilot_manifest(tmp_path):
    mgr = PilotDatasetManager()
    out_csv = tmp_path / "phase13_pilot.csv"
    mgr.generate_pilot_manifest(output_csv_path=out_csv, num_pilot_samples=5)

    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert "sample_id,video_id,source_checksum" in content
