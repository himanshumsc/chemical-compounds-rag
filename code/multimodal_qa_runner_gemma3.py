#!/usr/bin/env python3
"""
Gemma-3 specific wrapper around multimodal_qa_runner_vllm.
Generates concise RAG answers using the unquantized Gemma-3 model via vLLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_qa_runner_vllm import (
    DEFAULT_INPUT_DIR,
    DEFAULT_QA_DIR,
    regenerate_from_existing_answers,
)

GEMMA_MODEL_PATH = Path("/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED")
GEMMA_TOKENIZER = "google/gemma-3-12b-it"
GEMMA_OUTPUT_DIR = Path("/home/himanshu/dev/output/gemma3_rag_concise")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate concise QA answers using Gemma-3 via vLLM"
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Directory containing existing answer files")
    parser.add_argument("--output-dir", type=str, default=str(GEMMA_OUTPUT_DIR),
                        help="Directory to save Gemma-3 regenerated answers")
    parser.add_argument("--qa-dir", type=str, default=str(DEFAULT_QA_DIR),
                        help="Directory containing original QA pair files")
    parser.add_argument("--max-new-tokens", type=int, default=500,
                        help="Maximum tokens for generation")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Limit number of files for testing (None = all)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of files to process per batch")
    parser.add_argument("--chromadb-path", type=str, default=None,
                        help="Path to ChromaDB storage (default env/constant)")
    parser.add_argument("--model-path", type=str, default=str(GEMMA_MODEL_PATH),
                        help="Path to Gemma weights (default: local unquantized snapshot)")
    parser.add_argument("--tokenizer-id", type=str, default=GEMMA_TOKENIZER,
                        help="Tokenizer identifier for vLLM (default: google/gemma-3-12b-it)")
    parser.add_argument("--quantization", type=str, default="bitsandbytes",
                        help="Quantization backend (default: bitsandbytes)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["auto", "float16", "bfloat16"],
                        help="Computation dtype for vLLM (default: bfloat16)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max sequence length for Gemma (default: 8192)")
    parser.add_argument("--gpu-memory-util", type=float, default=0.80,
                        help="GPU memory utilization target (default: 0.80)")
    parser.add_argument("--no-enforce-eager", action="store_true",
                        help="Disable enforce_eager (enabled by default)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    enforce_eager = not args.no_enforce_eager
    summary = regenerate_from_existing_answers(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        qa_dir=Path(args.qa_dir),
        max_new_tokens=args.max_new_tokens,
        test_limit=args.test_limit,
        batch_size=args.batch_size,
        chromadb_path=args.chromadb_path,
        model_path=Path(args.model_path),
        tokenizer_id=args.tokenizer_id,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_util,
        enforce_eager=enforce_eager,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


