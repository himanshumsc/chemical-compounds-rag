#!/usr/bin/env python3
"""
Download helper for google/gemma-3-12b-it-qat-q4_0-unquantized.
This pulls the standard (non-GGUF) checkpoint so vLLM can run it with
bitsandbytes quantization.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_TARGET = Path("/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED")


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def check_disk_space(target: Path, reserve_gb: int) -> None:
    target.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(target)
    print(f"💾 Disk status for {target}:")
    print(f"   Total: {format_gb(total)}")
    print(f"   Used:  {format_gb(used)}")
    print(f"   Free:  {format_gb(free)}")
    if free < reserve_gb * 1024 ** 3:
        print(
            f"⚠️  Less than {reserve_gb} GB free. "
            "The unquantized Gemma snapshot exceeds 15 GB."
        )


def clear_target_directory(target_dir: Path) -> None:
    if target_dir.exists():
        print(f"🧹 Clearing existing directory: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def download_snapshot(repo_id: str, target_dir: Path, max_workers: int) -> None:
    clear_target_directory(target_dir)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        print("🔐 Using HF token from environment")
    else:
        print("⚠️  No HF token detected; gated repos will fail without access.")
    print(f"🚀 Downloading {repo_id}")
    print(f"📁 Destination: {target_dir}")
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
    print("\n📋 Downloaded files:")
    for entry in sorted(target_dir.rglob("*")):
        if entry.is_file():
            size_mb = entry.stat().st_size / (1024 ** 2)
            print(f" - {entry.relative_to(target_dir)} ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download google/gemma-3-12b-it-qat-q4_0-unquantized."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--reserve-gb",
        type=int,
        default=40,
        help="Warn when free space falls below this threshold",
    )
    parser.add_argument("--skip-disk-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_disk_check:
        check_disk_space(args.target, args.reserve_gb)
    try:
        download_snapshot(args.repo_id, args.target, args.max_workers)
        summarize(args.target)
    except Exception as exc:
        print(f"❌ Download failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()


