#!/bin/bash
# Quick benchmark script - automatically uses llama-env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Qwen2.5-72B Full GPU Benchmark"
echo "=========================================="
echo ""

# Run the benchmark with full GPU offloading
./run_gpu_test.sh \
  --model-path /home/himanshu/MSC_FINAL/dev/models/QWEN2_5_72B_GGUF/qwen2.5-72b-instruct-q6_k-00001-of-00016.gguf \
  --n-gpu-layers -1 \
  --max-tokens 100 \
  --benchmark 5 \
  --export benchmark_results.json

echo ""
echo "✅ Benchmark complete! Results saved to benchmark_results.json"


