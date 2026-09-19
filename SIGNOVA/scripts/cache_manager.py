#!/usr/bin/env python3
"""
Video Cache Manager CLI for SIGNOVA.

Inspects cached video files, displays quota utilization, and safely purges
stale unlocked files according to the LRU retention policy.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.data.cache import VideoCacheManager


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Local Video Cache Management CLI.")
    parser.add_argument(
        "--action",
        type=str,
        default="status",
        choices=["status", "list", "enforce-capacity", "clear"],
        help="Action to perform on cache.",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=10.0,
        help="Maximum cache quota in gigabytes (default 10.0 GB).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force clear even locked files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mgr = VideoCacheManager(max_cache_gb=args.max_gb)

    print("=" * 70)
    print("SIGNOVA Video Cache Manager")
    print(f"Cache Location: {mgr.cache_dir}")
    print(f"Quota Limit:    {args.max_gb:.2f} GB")
    print("=" * 70)

    count, size_gb = mgr.get_cache_size()

    if args.action == "status":
        print(f"Total Cached Files: {count}")
        print(f"Current Cache Size: {size_gb:.4f} GB ({size_gb / args.max_gb * 100:.1f}% of quota)")
        print(f"Status:             {'OK' if size_gb <= args.max_gb else 'OVER QUOTA'}")

    elif args.action == "list":
        samples = mgr.list_cached_samples()
        if not samples:
            print("Video cache is currently empty.")
        else:
            print(f"{'Sample ID':<30} {'Filename':<35} {'Size (MB)':<10} {'Locked'}")
            print("-" * 80)
            for s in samples:
                print(f"{s['sample_id']:<30} {s['filename']:<35} {s['size_mb']:<10} {s['is_locked']}")
            print("-" * 80)
            print(f"Total: {len(samples)} file(s), {size_gb:.3f} GB")

    elif args.action == "enforce-capacity":
        evicted = mgr.enforce_capacity()
        print(f"Enforced capacity quota ({args.max_gb} GB). Evicted {evicted} LRU file(s).")
        count_after, size_after = mgr.get_cache_size()
        print(f"New Cache Size: {size_after:.4f} GB ({count_after} files)")

    elif args.action == "clear":
        cleared = mgr.clear(force=args.force)
        print(f"Cleared {cleared} file(s) from cache.")


if __name__ == "__main__":
    main()
