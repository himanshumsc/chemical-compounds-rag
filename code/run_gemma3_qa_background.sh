#!/bin/bash
# Background runner for Gemma3 QA generation with validation
# Runs vLLM test first, then sample, then full set
# Continues running even if SSH disconnects

set -euo pipefail

SCRIPT_DIR="/home/himanshu/dev/code"
VENV_PYTHON="/home/himanshu/dev/code/.venv_phi4_req/bin/python3"
OUTPUT_DIR="/home/himanshu/dev/output/gemma3_rag_concise"
LOG_DIR="${OUTPUT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/background_run_${TIMESTAMP}.log"
TEST_LOG="${LOG_DIR}/vllm_test_${TIMESTAMP}.log"
SAMPLE_LOG="${LOG_DIR}/sample_run_${TIMESTAMP}.log"
FULL_LOG="${LOG_DIR}/full_run_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Gemma3 QA Generation - Background Runner"
echo "=========================================="
echo "Main log: ${MAIN_LOG}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Step 1: Test vLLM initialization
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Step 1: Testing vLLM initialization..." | tee -a "${MAIN_LOG}"
if ! HF_TOKEN="${HF_TOKEN}" "${VENV_PYTHON}" "${SCRIPT_DIR}/test_gemma3_vllm.py" \
    --enforce-eager \
    --gpu-util 0.80 \
    --model-path /home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED \
    > "${TEST_LOG}" 2>&1; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ERROR: vLLM test failed! Check ${TEST_LOG}" | tee -a "${MAIN_LOG}"
    echo "Aborting - vLLM initialization failed" | tee -a "${MAIN_LOG}"
    exit 1
fi
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ✓ vLLM test passed" | tee -a "${MAIN_LOG}"

# Step 2: Run sample (2 files)
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Step 2: Running sample (2 files)..." | tee -a "${MAIN_LOG}"
if ! HF_TOKEN="${HF_TOKEN}" "${VENV_PYTHON}" "${SCRIPT_DIR}/multimodal_qa_runner_gemma3.py" \
    --test-limit 2 \
    --batch-size 1 \
    > "${SAMPLE_LOG}" 2>&1; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ERROR: Sample run failed! Check ${SAMPLE_LOG}" | tee -a "${MAIN_LOG}"
    echo "Aborting - sample run failed" | tee -a "${MAIN_LOG}"
    exit 1
fi
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ✓ Sample run passed" | tee -a "${MAIN_LOG}"

# Step 3: Full generation
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Step 3: Starting full generation (batch-size 6)..." | tee -a "${MAIN_LOG}"
HF_TOKEN="${HF_TOKEN}" "${VENV_PYTHON}" "${SCRIPT_DIR}/multimodal_qa_runner_gemma3.py" \
    --batch-size 6 \
    --gpu-memory-util 0.80 \
    > "${FULL_LOG}" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ✓ Full generation completed successfully" | tee -a "${MAIN_LOG}"
else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ERROR: Full generation failed with exit code ${EXIT_CODE}" | tee -a "${MAIN_LOG}"
    echo "Check ${FULL_LOG} for details" | tee -a "${MAIN_LOG}"
fi

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] All done. Summary:" | tee -a "${MAIN_LOG}"
echo "  - Output files: $(ls -1 ${OUTPUT_DIR}/*__answers.json 2>/dev/null | wc -l)" | tee -a "${MAIN_LOG}"
echo "  - Summary: ${OUTPUT_DIR}/rag_regeneration_summary.json" | tee -a "${MAIN_LOG}"

