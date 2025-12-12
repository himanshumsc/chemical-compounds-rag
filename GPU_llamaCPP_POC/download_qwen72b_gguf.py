#!/usr/bin/env python3
"""
Download Qwen2.5-72B-Instruct-Q6_K_M.gguf model from Hugging Face.
This is a large model (~50GB), ensure sufficient disk space.

Usage:
    # Using wrapper (recommended - auto-activates correct env):
    ./download_qwen72b.sh
    
    # Or directly (requires huggingface_hub installed):
    python download_qwen72b_gguf.py [--target TARGET_DIR] [--skip-disk-check]
    
Note: This script requires huggingface_hub. Install with:
    pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("❌ huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)

# Use official Qwen repository
DEFAULT_REPO = "Qwen/Qwen2.5-72B-Instruct-GGUF"
DEFAULT_TARGET = Path("/home/himanshu/MSC_FINAL/dev/models/QWEN2_5_72B_GGUF")
# Note: Q6_K_M may not exist as single file, will try Q6_K shards or find best available
MODEL_FILE_NAME = "qwen2.5-72b-instruct-q6_k_m.gguf"


def format_gb(num_bytes: int) -> str:
    """Return a human readable GB string."""
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def check_disk_space(target_dir: Path, reserve_gb: int) -> None:
    """Print disk stats and warn when remaining space looks unsafe."""
    # Check parent directory if target doesn't exist yet
    check_path = target_dir if target_dir.exists() else target_dir.parent
    if not check_path.exists():
        check_path = check_path.parent
    
    total, used, free = shutil.disk_usage(check_path)
    print(f"💾 Disk status for {check_path}:")
    print(f"   Total: {format_gb(total)}")
    print(f"   Used:  {format_gb(used)}")
    print(f"   Free:  {format_gb(free)}")
    if free < reserve_gb * 1024 ** 3:
        print(
            f"⚠️  Less than {reserve_gb} GB free. "
            f"Qwen2.5-72B Q6_K_M requires ~50 GB."
        )


def find_model_repo() -> str | None:
    """Try to find the correct repository with the model file."""
    # Use official Qwen repository
    repos_to_try = [DEFAULT_REPO]
    
    print("🔍 Checking official Qwen repository...")
    for repo_id in repos_to_try:
        try:
            print(f"   Checking {repo_id}...")
            # Try to list files in the repo (this will fail if repo doesn't exist)
            from huggingface_hub import list_repo_files
            files = list(list_repo_files(repo_id))
            
            # Check if our target file exists
            if MODEL_FILE_NAME in files or any(MODEL_FILE_NAME.lower() in f.lower() for f in files):
                print(f"✅ Found model in {repo_id}")
                return repo_id
            else:
                # Check for similar filenames
                matching = [f for f in files if "q6_k_m" in f.lower() and "72b" in f.lower()]
                if matching:
                    print(f"⚠️  Found similar files in {repo_id}: {matching[:3]}")
                    return repo_id
        except Exception as e:
            print(f"   ❌ {repo_id}: {str(e)[:50]}")
            continue
    
    return None


def find_available_quantization(repo_id: str) -> tuple[str, list[str]] | None:
    """Find available quantization files in the repository."""
    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(repo_id))
        gguf_files = [f for f in files if f.endswith('.gguf')]
        
        # Prefer Q6_K_M single file, then Q6_K_M shards, then Q6_K shards
        q6_m_single = [f for f in gguf_files if 'q6_k_m' in f.lower() and 'of-' not in f]
        q6_m_shards = [f for f in gguf_files if 'q6_k_m' in f.lower() and 'of-' in f]
        q6_k_shards = [f for f in gguf_files if 'q6_k' in f.lower() and 'q6_k_m' not in f.lower() and 'of-' in f]
        
        if q6_m_single:
            return ("single", q6_m_single)
        elif q6_m_shards:
            return ("shards", sorted(q6_m_shards))
        elif q6_k_shards:
            return ("shards", sorted(q6_k_shards))
        else:
            # Fallback: any Q6 files
            q6_files = [f for f in gguf_files if 'q6' in f.lower()]
            if q6_files:
                single_q6 = [f for f in q6_files if 'of-' not in f]
                if single_q6:
                    return ("single", single_q6[:1])
                else:
                    return ("shards", sorted(q6_files))
        return None
    except Exception as e:
        print(f"⚠️  Could not list repository files: {e}")
        return None


def download_model_file(repo_id: str, target_dir: Path) -> Path | None:
    """Download the GGUF model file(s) from official Qwen repository."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Searching for model files in {repo_id}...")
    print(f"📁 Destination: {target_dir}")
    
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        print("🔐 Using access token from environment")
    else:
        print("⚠️  No HF_TOKEN detected; using public access")
    
    # Find available quantizations
    quant_info = find_available_quantization(repo_id)
    if not quant_info:
        print("❌ No Q6 quantization files found in repository")
        return None
    
    quant_type, files = quant_info
    print(f"\n📦 Found {len(files)} file(s) for {quant_type} format")
    
    if quant_type == "single":
        # Download single file
        filename = files[0]
        print(f"📥 Downloading: {filename}")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                resume_download=True,
                token=token,
            )
            model_path = Path(local_path)
            if model_path.exists():
                print(f"✅ Model file downloaded: {model_path}")
                return model_path
        except Exception as e:
            print(f"❌ Failed to download: {e}")
            return None
    else:
        # Download all shards
        print(f"📥 Downloading {len(files)} shard files...")
        print("   (This may take a while - ~50GB total)")
        
        downloaded_files = []
        for i, filename in enumerate(files, 1):
            print(f"   [{i}/{len(files)}] Downloading {filename}...")
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(target_dir),
                    resume_download=True,
                    token=token,
                )
                downloaded_files.append(Path(local_path))
            except Exception as e:
                print(f"   ❌ Failed to download {filename}: {e}")
                return None
        
        if downloaded_files:
            print(f"\n✅ All {len(downloaded_files)} shards downloaded")
            print("⚠️  Note: llama.cpp can use sharded GGUF files directly")
            print(f"   First shard: {downloaded_files[0]}")
            return downloaded_files[0]  # Return first shard (llama.cpp handles shards automatically)
    
    return None


def summarize(target_dir: Path, model_path: Path | None) -> None:
    """List the downloaded files so the user can review artifacts."""
    if not model_path or not model_path.exists():
        print("\n❌ Model file not found")
        return
        
    print("\n📋 Downloaded model:")
    size_mb = model_path.stat().st_size / (1024 ** 2)
    size_gb = size_mb / 1024
    print(f" - {model_path.name}")
    print(f"   Size: {size_gb:.2f} GB ({size_mb:.0f} MB)")
    print(f"   Path: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen2.5-72B-Instruct-Q6_K_M.gguf model."
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=f"Hugging Face repo id (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Destination directory (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip the disk space warning prompt",
    )
    parser.add_argument(
        "--reserve-gb",
        type=int,
        default=55,
        help="Warn if less free space than this threshold (default: 55 GB)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.skip_disk_check:
        check_disk_space(args.target, args.reserve_gb)
        print("\n⚠️  This is a large download (~50GB). Continue?")
        response = input("Press Enter to continue or Ctrl+C to cancel: ")
    
    # Find the correct repository
    repo_id = args.repo_id
    if not repo_id:
        repo_id = find_model_repo()
        if not repo_id:
            print(f"\n❌ Could not find model in any repository")
            print(f"   Trying default: {DEFAULT_REPO}")
            repo_id = DEFAULT_REPO
    
    if not repo_id:
        print("❌ No repository specified or found")
        sys.exit(1)
    
    print(f"\n🚀 Using repository: {repo_id}")
    
    try:
        model_path = download_model_file(repo_id, args.target)
        
        if model_path and model_path.exists():
            summarize(args.target, model_path)
            print(
                f"\n✅ Download complete!"
                f"\n🚀 Test it with:"
                f"\n   ./run_gpu_test.sh --model-path {model_path} --n-gpu-layers -1 --benchmark 3"
            )
        else:
            print("\n❌ Download failed or model file not found")
            print(f"   Check repository: {repo_id}")
            print(f"   Expected file: {MODEL_FILE_NAME}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("   You can resume by running the script again (resume_download=True)")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Download failed: {exc}")
        import traceback
        print(f"\nTraceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

