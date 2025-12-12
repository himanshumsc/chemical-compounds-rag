#!/usr/bin/env python3
"""
Sanity check for loading Gemma-3 GGUF through vLLM's GGUF backend.
Prepare configs but do not run heavy initialization until the code is reviewed.
"""

from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:
    print("❌ vLLM is required for this test. Install vllm>=0.5.4.")
    raise

# Default to the Hugging Face repo for the unquantized checkpoint
DEFAULT_MODEL = "google/gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_TOKENIZER = "google/gemma-3-12b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gemma-3 GGUF with vLLM")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help="Local path or Hugging Face repo id for the Gemma model",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer identifier passed to vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="max_model_len forwarded to vLLM",
    )
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=0.80,
        help="gpu_memory_utilization parameter",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="bfloat16",
        help="dtype forwarded to vLLM",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM for stability on single GPUs",
    )
    parser.add_argument(
        "--quantization",
        default="bitsandbytes",
        help="Quantization backend to request from vLLM (e.g. bitsandbytes)",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "FLASH_ATTN", "TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION"],
        default="auto",
        help="Override the attention backend if flash-attn is unstable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_input = args.model_path
    model_arg: str
    path_candidate = Path(model_input)
    if path_candidate.exists():
        model_arg = str(path_candidate)
    else:
        model_arg = model_input

    print("=" * 70)
    print("Gemma-3 GGUF vLLM smoke test (dry run until executed manually)")
    print("=" * 70)
    print(f"Model path : {model_arg}")
    print(f"Tokenizer  : {args.tokenizer}")
    print(f"Max len    : {args.max_model_len}")
    print(f"GPU util   : {args.gpu_util}")

    start = time.time()
    if args.attention_backend != "auto":
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    llm = LLM(
        model=model_arg,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=args.enforce_eager,
    )
    init_time = time.time() - start
    print(f"✓ vLLM initialized in {init_time:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )
    prompt = "Outline three safety precautions when handling acetic acid in a lab."
    outputs = llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text.strip()
    print("\n=== Sample Output ===")
    print(text)
    print("=====================")


if __name__ == "__main__":
    main()


