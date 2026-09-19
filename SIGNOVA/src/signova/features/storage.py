"""
Feature Storage and Serialization Utilities for SIGNOVA.

Provides structured saving/loading of extracted landmark arrays,
binary detection masks, temporal metadata, schema versioning,
and compression benchmarking.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

EXTRACTOR_SCHEMA_VERSION = "0.1.0"


def save_landmark_features(
    output_path: Union[str, Path],
    landmarks: np.ndarray,              # (T, 543, 3)
    detection_masks: np.ndarray,        # (T, 4)
    timestamps_ms: np.ndarray,          # (T,)
    frame_indices: np.ndarray,          # (T,)
    sample_id: str,
    split: str,
    total_source_frames: int,
    use_compression: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save extracted features to disk (.npz or .npy).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "sample_id": sample_id,
        "split": split,
        "total_source_frames": int(total_source_frames),
        "extracted_frames": int(landmarks.shape[0]),
        "extractor_version": EXTRACTOR_SCHEMA_VERSION,
        "landmark_shape": list(landmarks.shape),
        "detection_mask_shape": list(detection_masks.shape),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    if use_compression:
        if not out_path.name.endswith(".npz"):
            out_path = out_path.with_suffix(".npz")
        np.savez_compressed(
            str(out_path),
            landmarks=landmarks.astype(np.float32),
            detection_masks=detection_masks.astype(np.float32),
            timestamps_ms=timestamps_ms.astype(np.float32),
            frame_indices=frame_indices.astype(np.int32),
            metadata_json=np.array(json.dumps(metadata)),
        )
    else:
        if not out_path.name.endswith(".npz"):
            out_path = out_path.with_suffix(".npz")
        np.savez(
            str(out_path),
            landmarks=landmarks.astype(np.float32),
            detection_masks=detection_masks.astype(np.float32),
            timestamps_ms=timestamps_ms.astype(np.float32),
            frame_indices=frame_indices.astype(np.int32),
            metadata_json=np.array(json.dumps(metadata)),
        )

    return out_path


def load_landmark_features(
    file_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load landmark arrays and metadata from an .npz file.

    Returns:
        (landmarks, detection_masks, timestamps_ms, frame_indices, metadata)
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Feature file not found: {p}")

    with np.load(str(p), allow_pickle=True) as data:
        landmarks = data["landmarks"]
        detection_masks = data["detection_masks"]
        timestamps_ms = data["timestamps_ms"]
        frame_indices = data["frame_indices"]
        raw_meta = data.get("metadata_json", None)
        metadata = {}
        if raw_meta is not None:
            metadata = json.loads(str(raw_meta))

    return landmarks, detection_masks, timestamps_ms, frame_indices, metadata


def benchmark_storage_formats(
    sample_sequence: np.ndarray,
    detection_masks: np.ndarray,
    scratch_dir: Path,
) -> Dict[str, Any]:
    """
    Compare .npy uncompressed, .npz uncompressed, and .npz compressed for size and I/O latency.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    T = sample_sequence.shape[0]
    timestamps = np.arange(T, dtype=np.float32) * 33.33
    frame_indices = np.arange(T, dtype=np.int32)

    # 1. Raw .npy (landmarks only)
    npy_path = scratch_dir / "bench_raw.npy"
    t0 = time.perf_counter()
    np.save(str(npy_path), sample_sequence)
    npy_save_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = np.load(str(npy_path))
    npy_load_time = time.perf_counter() - t0
    npy_size = npy_path.stat().st_size

    # 2. NPZ uncompressed
    npz_raw_path = scratch_dir / "bench_raw.npz"
    t0 = time.perf_counter()
    save_landmark_features(
        npz_raw_path,
        sample_sequence,
        detection_masks,
        timestamps,
        frame_indices,
        sample_id="bench",
        split="bench",
        total_source_frames=T,
        use_compression=False,
    )
    npz_raw_save_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = load_landmark_features(npz_raw_path)
    npz_raw_load_time = time.perf_counter() - t0
    npz_raw_size = npz_raw_path.stat().st_size

    # 3. NPZ compressed
    npz_comp_path = scratch_dir / "bench_comp.npz"
    t0 = time.perf_counter()
    save_landmark_features(
        npz_comp_path,
        sample_sequence,
        detection_masks,
        timestamps,
        frame_indices,
        sample_id="bench",
        split="bench",
        total_source_frames=T,
        use_compression=True,
    )
    npz_comp_save_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = load_landmark_features(npz_comp_path)
    npz_comp_load_time = time.perf_counter() - t0
    npz_comp_size = npz_comp_path.stat().st_size

    # Clean up benchmark files
    for p in [npy_path, npz_raw_path, npz_comp_path]:
        if p.exists():
            p.unlink()

    return {
        "sequence_frames": T,
        "npy_size_bytes": npy_size,
        "npy_save_sec": round(npy_save_time, 5),
        "npy_load_sec": round(npy_load_time, 5),
        "npz_uncompressed_size_bytes": npz_raw_size,
        "npz_uncompressed_save_sec": round(npz_raw_save_time, 5),
        "npz_uncompressed_load_sec": round(npz_raw_load_time, 5),
        "npz_compressed_size_bytes": npz_comp_size,
        "npz_compressed_save_sec": round(npz_comp_save_time, 5),
        "npz_compressed_load_sec": round(npz_comp_load_time, 5),
        "compression_ratio": round(npy_size / max(npz_comp_size, 1), 2),
    }
