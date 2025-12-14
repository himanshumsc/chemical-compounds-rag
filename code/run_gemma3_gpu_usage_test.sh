#!/usr/bin/env bash
# Helper script to pair the Gemma-3 multimodal test with GPU monitoring.
# The monitor runs in the background and is stopped after the test completes.

set -euo pipefail

MONITOR_SCRIPT="/home/himanshu/dev/code/monitor_gpu.sh"
MULTIMODAL_TEST="/home/himanshu/dev/code/test_gemma3_multimodal.py"
MODEL_DEFAULT="/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_GGUF/gemma-3-12b-it-qat-q4_0.gguf"
LOG_DIR="/home/himanshu/dev/output/gemma3/logs"

MODEL_PATH="${1:-$MODEL_DEFAULT}"
IMAGE_PATH="${2:-/home/himanshu/dev/input_img/img1.png}"
CHAT_FORMAT="${3:-gemma-2}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "Model path : ${MODEL_PATH}"
echo "Image path : ${IMAGE_PATH}"
echo "Chat format: ${CHAT_FORMAT}"
echo "GPU log    : ${LOG_FILE}"
echo ""
echo "Starting GPU monitor..."
"${MONITOR_SCRIPT}" "${LOG_FILE}" &
MONITOR_PID=$!

cleanup() {
    if ps -p "${MONITOR_PID}" >/dev/null 2>&1; then
        echo "Stopping GPU monitor (PID ${MONITOR_PID})"
        kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "Running Gemma-3 multimodal test..."
python "${MULTIMODAL_TEST}" \
    --model-path "${MODEL_PATH}" \
    --image-path "${IMAGE_PATH}" \
    --chat-format "${CHAT_FORMAT}"

echo "Test finished. GPU metrics stored in ${LOG_FILE}"


