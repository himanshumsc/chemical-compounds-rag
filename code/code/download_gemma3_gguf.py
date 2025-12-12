#!/usr/bin/env python3
"""
Download helper for google/gemma-3-12b-it-qat-q4_0-gguf.
Saves the snapshot under /home/himanshu/dev/models/GEMMA3_QAT_Q4_0_GGUF.

The script mirrors the structure used for other model downloaders so it can be
reviewed before actually running heavy transfers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "google/gemma-3-12b-it-qat-q4_0-gguf"
DEFAULT_TARGET = Path("/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_GGUF")


def format_gb(num_bytes: int) -> str:
    """Return a human readable GB string."""
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def check_disk_space(target_dir: Path, reserve_gb: int) -> None:
    """Print disk stats and warn when remaining space looks unsafe."""
    total, used, free = shutil.disk_usage(target_dir)
    print(f"💾 Disk status for {target_dir}:")
    print(f"   Total: {format_gb(total)}")
    print(f"   Used:  {format_gb(used)}")
    print(f"   Free:  {format_gb(free)}")
    if free < reserve_gb * 1024 ** 3:
        print(
            f"⚠️  Less than {reserve_gb} GB free. "
            "Gemma-3 12B GGUF can exceed 8 GB depending on extra assets."
        )


def download_snapshot(repo_id: str, target_dir: Path, max_workers: int) -> None:
    """Download the Hugging Face snapshot into the requested directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Downloading {repo_id}")
    print(f"📁 Destination: {target_dir}")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("⚠️  No HF_TOKEN/HUGGINGFACE_HUB_TOKEN detected; using default auth.")
    else:
        print("🔐 Using access token from environment (HF_TOKEN/HUGGINGFACE_HUB_TOKEN)")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        max_workers=max_workers,
        resume_download=True,
        token=token,
    )
    print("✅ Snapshot download finished")


def summarize(target_dir: Path) -> None:
    """List the downloaded files so the user can review artifacts."""
    print("\n📋 Downloaded files:")
    for path in sorted(target_dir.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 ** 2)
            print(f" - {path.relative_to(target_dir)} ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download google/gemma-3-12b-it-qat-q4_0-gguf snapshot."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help=f"Hugging Face repo id (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Destination directory (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers passed to huggingface_hub",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip the disk space warning prompt",
    )
    parser.add_argument(
        "--reserve-gb",
        type=int,
        default=10,
        help="Warn if less free space than this threshold (default: 10 GB)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_disk_check:
        check_disk_space(args.target, args.reserve_gb)
    try:
        download_snapshot(args.repo_id, args.target, args.max_workers)
        summarize(args.target)
        print(
            "\nℹ️  Review the listing above. Run the script only after confirming "
            "the directory is correct and you have accepted the Gemma license."
        )
    except Exception as exc:
        print(f"❌ Download failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()


