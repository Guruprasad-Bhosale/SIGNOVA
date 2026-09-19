"""
Remote Video Resolver and Downloader for SIGNOVA.

Resolves sample references to local cached video files. Authentic remote provider
support (Hugging Face / HTTP).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import urllib.request
import os

from signova.data.cache import VideoCacheManager
from signova.data.models import SampleAvailability, SignSample


class RemoteVideoResolver:
    """
    Resolves remote video references, checks cache, and acquires remote video assets.
    """

    def __init__(
        self,
        cache_manager: Optional[VideoCacheManager] = None,
        remote_provider: str = "huggingface",
        dataset_repo: str = "Exploration-Lab/iSign",
        base_url: Optional[str] = None,
        timeout_seconds: int = 30,
    ):
        self.cache_manager = cache_manager or VideoCacheManager()
        self.remote_provider = remote_provider
        self.dataset_repo = dataset_repo
        self.base_url = base_url or "https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main"
        self.timeout_seconds = timeout_seconds

    def is_cached(self, sample_id: str) -> bool:
        """Check if video file exists locally in cache."""
        return self.cache_manager.get_cached_path(sample_id) is not None

    def get_local_path(self, sample_id: str) -> Optional[Path]:
        """Return path to cached local video file if present."""
        return self.cache_manager.get_cached_path(sample_id)

    def download_url(self, url: str, sample_id: str, extension: str = ".mp4") -> Tuple[Optional[Path], str]:
        """
        Download a single video file directly from an HTTP/HTTPS endpoint into the cache.
        """
        target_path = self.cache_manager.cache_dir / f"{sample_id}{extension}"
        if target_path.is_file() and target_path.stat().st_size > 0:
            return target_path, "CACHED"

        temp_path = self.cache_manager.cache_dir / f"{sample_id}_tmp{extension}"
        try:
            self.cache_manager.lock_file(sample_id)
            req = urllib.request.Request(url, headers={"User-Agent": "SIGNOVA/0.2.0"})
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                with open(temp_path, "wb") as out_f:
                    shutil_chunk_size = 1024 * 1024  # 1MB
                    while True:
                        chunk = resp.read(shutil_chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)

            if temp_path.stat().st_size == 0:
                temp_path.unlink(missing_ok=True)
                return None, "DOWNLOAD_EMPTY_PAYLOAD"

            temp_path.rename(target_path)
            return target_path, "DOWNLOADED"
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            return None, f"DOWNLOAD_FAILED: {str(e)}"
        finally:
            self.cache_manager.unlock_file(sample_id)

    def resolve_video(
        self,
        sample: Union[SignSample, Dict[str, Any]],
    ) -> Tuple[Optional[Path], str]:
        """
        Resolve a sample to a concrete local video file on disk.

        Returns:
            (local_path: Optional[Path], status_message: str)
        """
        sample_id = sample.sample_id if isinstance(sample, SignSample) else sample.get("sample_id", "")
        vid_ref = sample.video_reference if isinstance(sample, SignSample) else sample.get("video_reference", sample_id)

        # 1. Check local cache
        cached_path = self.cache_manager.get_cached_path(sample_id)
        if cached_path:
            return cached_path, "CACHED"

        # 2. Check local mount if provided
        if isinstance(sample, SignSample) and sample.local_video_path:
            p = Path(sample.local_video_path)
            if p.is_file() and p.stat().st_size > 0:
                return p, "LOCAL_MOUNT"

        # 3. Check explicit URL in metadata
        metadata = sample.metadata if isinstance(sample, SignSample) else sample.get("metadata", {})
        if isinstance(metadata, dict) and "download_url" in metadata:
            return self.download_url(metadata["download_url"], sample_id)

        # 4. As documented in docs/remote-data-access.md, ISLTranslate videos are packaged
        # in two monolithic multi-part archives on Hugging Face (iSign-videos_v1.1_part_aa/ab).
        # Individual single-video HTTP URLs are not provided by Hugging Face for this dataset.
        return None, "REMOTE_ARCHIVE_UNMOUNTED"
