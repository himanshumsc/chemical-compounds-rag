#!/bin/bash
# Quick test with small batch size for Gemma3 QA generation

set -euo pipefail

SCRIPT_DIR="/home/himanshu/dev/code"
VENV_PYTHON="/home/himanshu/dev/code/.venv_phi4_req/bin/python3"
OUTPUT_DIR="/home/himanshu/dev/output/gemma3_rag_concise"
LOG_DIR="${OUTPUT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="${LOG_DIR}/small_batch_test_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Gemma3 QA - Small Batch Test (batch-size 6)"
echo "=========================================="
echo "Test log: ${TEST_LOG}"
echo ""

# Run test with 3 files, batch-size 6
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Running test with 3 files, batch-size 6..." | tee "${TEST_LOG}"
HF_TOKEN="${HF_TOKEN}" "${VENV_PYTHON}" "${SCRIPT_DIR}/multimodal_qa_runner_gemma3.py" \
    --test-limit 3 \
    --batch-size 6 \
    --gpu-memory-util 0.80 \
    2>&1 | tee -a "${TEST_LOG}"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ✓ Test passed!" | tee -a "${TEST_LOG}"
    echo "Generated files: $(ls -1 ${OUTPUT_DIR}/*__answers.json 2>/dev/null | wc -l)" | tee -a "${TEST_LOG}"
else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ✗ Test failed with exit code ${EXIT_CODE}" | tee -a "${TEST_LOG}"
fi

