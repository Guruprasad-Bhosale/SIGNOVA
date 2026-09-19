"""
Unit tests for SIGNOVA Phase 2 Video, Remote Acquisition, Landmark Extraction,
Spatial Normalization, Quality Analytics, and Feature Storage Pipelines.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from signova.data.cache import VideoCacheManager
from signova.data.remote import RemoteVideoResolver
from signova.preprocessing.video import RobustVideoDecoder, VideoMetadata
from signova.preprocessing.normalization import normalize_landmark_sequence, normalize_landmarks_frame
from signova.features.mediapipe_extractor import ExtractionResult, MediaPipeHolisticExtractor
from signova.features.quality import LandmarkQualityEvaluator, QualityMetrics
from signova.features.storage import (
    EXTRACTOR_SCHEMA_VERSION,
    benchmark_storage_formats,
    load_landmark_features,
    save_landmark_features,
)
from signova.features.visualization import create_pilot_visualization_grid, render_landmarks_on_image


def test_video_cache_manager_locking_and_capacity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = VideoCacheManager(cache_dir=Path(tmp_dir), max_cache_gb=0.0001)  # ~100 KB limit
        
        # Create dummy video files
        f1 = Path(tmp_dir) / "sample_1.mp4"
        f2 = Path(tmp_dir) / "sample_2.mp4"
        f1.write_bytes(b"0" * 60000)
        f2.write_bytes(b"0" * 60000)
        
        assert mgr.get_cached_path("sample_1").resolve() == f1.resolve()
        assert mgr.get_cached_path("sample_2").resolve() == f2.resolve()

        
        # Lock sample_1
        mgr.lock_file("sample_1")
        assert mgr.is_locked("sample_1") is True
        assert mgr.is_locked("sample_2") is False
        
        # Enforce capacity: sample_2 should be evicted, sample_1 must be protected
        evicted = mgr.enforce_capacity()
        assert evicted == 1
        assert f1.exists() is True
        assert f2.exists() is False
        
        # Unlock sample_1 and clear
        mgr.unlock_file("sample_1")
        assert mgr.is_locked("sample_1") is False
        mgr.clear()
        assert f1.exists() is False


def test_remote_video_resolver_unmounted():
    resolver = RemoteVideoResolver()
    sample = {"sample_id": "non_existent_sample_9999", "split": "train"}
    path, status = resolver.resolve_video(sample)
    assert path is None
    assert status == "REMOTE_ARCHIVE_UNMOUNTED"


def test_robust_video_decoder_missing_file():
    decoder = RobustVideoDecoder(Path("non_existent_file.mp4"))
    meta = decoder.read_metadata()
    assert meta.is_valid is False
    assert "not found" in meta.error.lower()


def test_mediapipe_extractor_topology():
    extractor = MediaPipeHolisticExtractor()
    assert extractor.EXPECTED_POSE == 33
    assert extractor.EXPECTED_FACE == 468
    assert extractor.EXPECTED_HAND == 21
    assert extractor.EXPECTED_TOTAL == 543
    
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lm, mask = extractor.extract_from_frame(frame)
    assert lm.shape == (543, 3)
    assert mask.shape == (4,)
    extractor.close()


def test_landmark_normalization_fallbacks_and_safety():
    # 1. Normal frame
    lm = np.zeros((543, 3), dtype=np.float32)
    # Left hip (idx 23), Right hip (idx 24)
    lm[23] = [0.4, 0.6, 0.9]
    lm[24] = [0.6, 0.6, 0.9]
    # Left shoulder (idx 11), Right shoulder (idx 12)
    lm[11] = [0.4, 0.3, 0.9]
    lm[12] = [0.6, 0.3, 0.9]
    
    norm = normalize_landmarks_frame(lm, center_anchor="mid_hip", scale_anchor="shoulder_dist")
    assert norm.shape == (543, 3)
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()
    
    # 2. NaN and Inf sanitization
    corrupt_lm = lm.copy()
    corrupt_lm[0] = [np.nan, np.inf, -np.inf]
    cleaned_norm = normalize_landmarks_frame(corrupt_lm)
    assert not np.isnan(cleaned_norm).any()
    assert not np.isinf(cleaned_norm).any()
    
    # 3. Sequence normalization
    seq = np.stack([lm, lm], axis=0)
    norm_seq = normalize_landmark_sequence(seq)
    assert norm_seq.shape == (2, 543, 3)


def test_landmark_quality_evaluator():
    evaluator = LandmarkQualityEvaluator()
    
    # Create sequence with pose and hands present
    T = 20
    lm_seq = np.zeros((T, 543, 3), dtype=np.float32)
    # Pose present
    lm_seq[:, 0:33, :2] = 0.5
    lm_seq[:, 0:33, 2] = 0.95
    # Left hand present
    lm_seq[:, 501:522, :2] = 0.6
    lm_seq[:, 501:522, 2] = 0.90
    
    mask = np.zeros((T, 4), dtype=np.float32)
    mask[:, 0] = 1.0  # pose
    mask[:, 1] = 1.0  # face
    mask[:, 2] = 1.0  # left hand
    mask[:, 3] = 0.0  # right hand
    
    metrics = evaluator.evaluate(lm_seq, mask)
    assert metrics.total_frames == T
    assert metrics.pose_detection_rate == 1.0
    assert metrics.left_hand_detection_rate == 1.0
    assert metrics.right_hand_detection_rate == 0.0
    assert metrics.any_hand_detection_rate == 1.0
    assert metrics.is_usable is True


def test_storage_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / "test_sample.npz"
        T = 15
        landmarks = np.random.uniform(-1, 1, (T, 543, 3)).astype(np.float32)
        masks = np.ones((T, 4), dtype=np.float32)
        timestamps = np.arange(T, dtype=np.float32) * 33.3
        frame_indices = np.arange(T, dtype=np.int32)
        
        saved = save_landmark_features(
            target_path,
            landmarks=landmarks,
            detection_masks=masks,
            timestamps_ms=timestamps,
            frame_indices=frame_indices,
            sample_id="test_01",
            split="train",
            total_source_frames=T,
            use_compression=True,
            extra_metadata={"custom_flag": True},
        )
        assert saved.exists()
        
        lm_loaded, mask_loaded, ts_loaded, fi_loaded, meta = load_landmark_features(saved)
        assert lm_loaded.shape == (T, 543, 3)
        assert mask_loaded.shape == (T, 4)
        assert np.allclose(lm_loaded, landmarks, atol=1e-5)
        assert np.allclose(mask_loaded, masks)
        assert meta["sample_id"] == "test_01"
        assert meta["custom_flag"] is True
        assert meta["extractor_version"] == EXTRACTOR_SCHEMA_VERSION


def test_storage_benchmark():
    with tempfile.TemporaryDirectory() as tmp_dir:
        seq = np.random.uniform(-1, 1, (20, 543, 3)).astype(np.float32)
        masks = np.ones((20, 4), dtype=np.float32)
        res = benchmark_storage_formats(seq, masks, Path(tmp_dir))
        assert "npz_compressed_size_bytes" in res
        assert "compression_ratio" in res
        assert res["sequence_frames"] == 20


def test_visualization_render():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lm = np.zeros((543, 3), dtype=np.float32)
        lm[:, :2] = 0.5
        lm[:, 2] = 0.8
        img = render_landmarks_on_image(None, lm, canvas_size=(200, 200), title="Test")
        assert img.shape == (200, 200, 3)
        
        seq = np.stack([lm] * 6, axis=0)
        out_p = Path(tmp_dir) / "grid.png"
        grid_saved = create_pilot_visualization_grid(seq, "test_vis", out_p, num_key_frames=6)
        assert grid_saved.exists()
