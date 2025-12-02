#!/usr/bin/env python3
"""
Lightweight multimodal validation for Gemma-3 GGUF checkpoints via llama-cpp-python.
The script is designed for review first; run it only after confirming the model
paths and optional mmproj asset exist locally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - llama_cpp might be optional
    print("❌ llama_cpp is required for this test. Install llama-cpp-python first.")
    raise


DEFAULT_MODEL_DIR = Path("/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_GGUF")
DEFAULT_IMAGE = Path("/home/himanshu/dev/input_img/img1.png")


def find_mmproj_path(model_path: Path) -> Optional[Path]:
    """Resolve an mmproj file next to the model if possible."""
    search_root = model_path if model_path.is_dir() else model_path.parent
    candidates = sorted(
        list(search_root.glob("*mmproj*.gguf")) + list(search_root.glob("*vision*.gguf"))
    )
    return candidates[0] if candidates else None


def build_messages(image_path: Path, prompt: str) -> list:
    """Create chat completion messages following llama.cpp multimodal schema."""
    file_uri = image_path.resolve().as_uri()
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a concise chemistry assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": file_uri}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def run_test(
    model_path: Path,
    image_path: Path,
    mmproj_path: Optional[Path],
    chat_format: str,
    n_ctx: int,
    n_gpu_layers: int,
) -> bool:
    """Load the GGUF checkpoint with llama.cpp and run a single multimodal prompt."""
    if not model_path.exists():
        print(f"✗ Model path not found: {model_path}")
        return False
    if not image_path.exists():
        print(f"✗ Image path not found: {image_path}")
        return False
    if mmproj_path:
        print(f"Using mmproj: {mmproj_path}")
    else:
        print("⚠️  No mmproj path provided; vision may fail depending on the build.")

    llm = Llama(
        model_path=str(model_path),
        mmproj_path=str(mmproj_path) if mmproj_path else None,
        chat_format=chat_format,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=False,
        seed=42,
    )

    messages = build_messages(image_path, "Summarize the contents of this image.")
    print("Running multimodal completion...")
    result = llm.create_chat_completion(messages=messages)
    content = result["choices"][0]["message"]["content"]
    print("\n=== Model Response ===")
    print(content)
    print("======================")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gemma-3 GGUF with llama.cpp")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_DIR / "gemma-3-12b-it-qat-q4_0.gguf",
        help="Path to the GGUF file or directory containing it",
    )
    parser.add_argument(
        "--mmproj-path",
        type=Path,
        default=None,
        help="Optional explicit mmproj GGUF path",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Test image to feed into the model",
    )
    parser.add_argument(
        "--chat-format",
        default="gemma-2",
        help="chat_format forwarded to llama.cpp (update if HF release changes)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Context size for llama.cpp",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=35,
        help="Number of layers to offload to GPU (-1 for all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    if model_path.is_dir():
        gguf_files = sorted(model_path.glob("*.gguf"))
        if not gguf_files:
            print(f"✗ No GGUF files found inside {model_path}")
            sys.exit(1)
        model_path = gguf_files[0]
        print(f"Auto-selected GGUF file: {model_path}")

    mmproj = args.mmproj_path or find_mmproj_path(model_path)
    success = run_test(
        model_path=model_path,
        image_path=args.image_path,
        mmproj_path=mmproj,
        chat_format=args.chat_format,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


