#!/bin/bash
# Run full RAG regeneration with Q1 max_tokens=300, Q2-Q4 max_tokens=500
# Runs in background with logging and GPU monitoring

SCRIPT_DIR="/home/himanshu/dev/code"
OUTPUT_DIR="/home/himanshu/dev/output/qwen_rag"
LOGS_DIR="${OUTPUT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOGS_DIR}/rag_regeneration_${TIMESTAMP}.log"
GPU_LOG="${LOGS_DIR}/gpu_monitor_${TIMESTAMP}.log"

# Create output and logs directories
mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}"

echo "Starting full RAG regeneration (Q1: 300 tokens, Q2-Q4: 500 tokens)..."
echo "Log file: ${LOG_FILE}"
echo "GPU log: ${GPU_LOG}"
echo ""

# Start GPU monitoring in background
(
    echo "GPU Monitoring started at $(date)" > "${GPU_LOG}"
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$(nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | tr '\n' ' ')" >> "${GPU_LOG}"
        sleep 5
    done
) &
GPU_MONITOR_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "Stopping GPU monitor (PID: ${GPU_MONITOR_PID})..."
    kill ${GPU_MONITOR_PID} 2>/dev/null
    echo "GPU monitoring stopped. Final log: ${GPU_LOG}"
}

trap cleanup EXIT

# Run the regeneration script
cd "${SCRIPT_DIR}"
nohup python3 multimodal_qa_runner_vllm.py \
    --input-dir /home/himanshu/dev/output/qwen_regenerated \
    --output-dir "${OUTPUT_DIR}" \
    --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components \
    --chromadb-path /home/himanshu/dev/data/chromadb \
    --batch-size 10 \
    --max-new-tokens 500 \
    > "${LOG_FILE}" 2>&1 &

MAIN_PID=$!

echo "Main process PID: ${MAIN_PID}"
echo "GPU monitor PID: ${GPU_MONITOR_PID}"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To monitor GPU:"
echo "  tail -f ${GPU_LOG}"
echo ""
echo "To check if still running:"
echo "  ps aux | grep ${MAIN_PID}"
echo ""
echo "To stop:"
echo "  kill ${MAIN_PID}"
echo ""

# Wait a moment and check if process started
sleep 2
if ps -p ${MAIN_PID} > /dev/null; then
    echo "✓ Process started successfully"
    echo "Logs are being written to: ${LOG_FILE}"
else
    echo "✗ Process failed to start. Check log: ${LOG_FILE}"
    exit 1
fi

