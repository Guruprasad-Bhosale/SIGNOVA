"""
Local Video Cache Management for SIGNOVA with Active Processing Locks.

Manages transient video downloads in `data/cache/videos/` with size limits,
active file locks to prevent in-flight eviction, and clean LRU eviction policies.
"""

from pathlib import Path
import shutil
import threading
from typing import Any, Dict, List, Optional, Set, Tuple


class VideoCacheManager:
    """
    Manages cached video files on local disk with concurrency-safe processing locks.
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_cache_gb: float = 10.0):
        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = (Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache" / "videos").resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = int(max_cache_gb * (1024**3))
        self._locked_files: Set[str] = set()
        self._lock_mutex = threading.Lock()

    def lock_file(self, sample_id: str):
        """Mark a video file as actively processing to protect it from cache eviction."""
        with self._lock_mutex:
            self._locked_files.add(sample_id)

    def unlock_file(self, sample_id: str):
        """Release processing lock on a video file."""
        with self._lock_mutex:
            self._locked_files.discard(sample_id)

    def is_locked(self, sample_id: str) -> bool:
        """Check if a sample is currently locked for processing."""
        with self._lock_mutex:
            return sample_id in self._locked_files

    def get_cached_path(self, sample_id: str) -> Optional[Path]:
        """Check if a video for the given sample_id exists in the cache and is non-empty."""
        for ext in [".mp4", ".avi", ".mkv", ".webm"]:
            candidate = self.cache_dir / f"{sample_id}{ext}"
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None

    def put_file(self, sample_id: str, source_file: Path, extension: str = ".mp4") -> Path:
        """Copy or move a source video into the managed cache."""
        self.enforce_capacity()
        target_path = self.cache_dir / f"{sample_id}{extension}"
        if source_file.resolve() != target_path.resolve():
            shutil.copy2(source_file, target_path)
        return target_path

    def get_cache_size(self) -> Tuple[int, float]:
        """Return (total_files, total_size_gb)."""
        files = [p for p in self.cache_dir.glob("*") if p.is_file() and p.name != ".gitkeep"]
        total_bytes = sum(f.stat().st_size for f in files)
        return len(files), total_bytes / (1024**3)

    def list_cached_samples(self) -> List[Dict[str, Any]]:
        """List all cached video assets and their metadata."""
        cached = []
        for p in self.cache_dir.glob("*"):
            if p.is_file() and p.name != ".gitkeep":
                sample_id = p.stem
                cached.append({
                    "sample_id": sample_id,
                    "filename": p.name,
                    "size_mb": round(p.stat().st_size / (1024**2), 2),
                    "path": str(p),
                    "is_locked": self.is_locked(sample_id),
                })
        return cached

    def enforce_capacity(self) -> int:
        """
        Evict oldest unlocked cached files if cache size exceeds max_cache_bytes.
        Never evicts locked files or files currently in-flight.
        """
        files = [p for p in self.cache_dir.glob("*") if p.is_file() and p.name != ".gitkeep"]
        total_bytes = sum(f.stat().st_size for f in files)

        if total_bytes <= self.max_cache_bytes:
            return 0

        # Sort unlocked files by last modification time (LRU)
        unlocked_files = [f for f in files if not self.is_locked(f.stem)]
        unlocked_files.sort(key=lambda x: x.stat().st_mtime)
        deleted_count = 0

        for f in unlocked_files:
            if total_bytes <= self.max_cache_bytes:
                break
            sz = f.stat().st_size
            try:
                f.unlink()
                total_bytes -= sz
                deleted_count += 1
            except Exception:
                pass

        return deleted_count

    def clear(self, force: bool = False) -> int:
        """Clear all cached video files. If force=False, skips actively locked files."""
        count = 0
        for p in self.cache_dir.glob("*"):
            if p.is_file() and p.name != ".gitkeep":
                if not force and self.is_locked(p.stem):
                    continue
                try:
                    p.unlink()
                    count += 1
                except Exception:
                    pass
        return count
