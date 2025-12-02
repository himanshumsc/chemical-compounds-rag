#!/bin/bash
# Full concise RAG regeneration with character limits - background execution

cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate

OUTPUT_DIR="/home/himanshu/dev/output/qwen_rag_concise"
LOGS_DIR="${OUTPUT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOGS_DIR}/concise_rag_regeneration_${TIMESTAMP}.log"
GPU_LOG="${LOGS_DIR}/gpu_monitor_${TIMESTAMP}.log"

# Create directories
mkdir -p "${LOGS_DIR}"

echo "=========================================="
echo "Starting Full Concise RAG Regeneration"
echo "=========================================="
echo "Input: /home/himanshu/dev/output/qwen_regenerated"
echo "Output: ${OUTPUT_DIR}"
echo "Character Limits: Q1=600, Q2=1000, Q3=1800, Q4=2000"
echo "Log file: ${LOG_FILE}"
echo "GPU log: ${GPU_LOG}"
echo "=========================================="

# Start GPU monitoring in background
(
    while true; do
        echo "=== $(date) ===" >> "${GPU_LOG}"
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv >> "${GPU_LOG}"
        echo "" >> "${GPU_LOG}"
        sleep 30
    done
) &
GPU_MONITOR_PID=$!

# Start main process in background
nohup python3 multimodal_qa_runner_vllm.py \
    --input-dir /home/himanshu/dev/output/qwen_regenerated \
    --output-dir "${OUTPUT_DIR}" \
    --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components \
    --chromadb-path /home/himanshu/dev/data/chromadb \
    --batch-size 10 \
    > "${LOG_FILE}" 2>&1 &
MAIN_PID=$!

# Kill GPU monitor when main process ends (in background)
(
    wait $MAIN_PID
    kill $GPU_MONITOR_PID 2>/dev/null
) &

echo ""
echo "Process started!"
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
echo "To check completed files:"
echo "  ls -1 ${OUTPUT_DIR}/*__answers.json 2>/dev/null | wc -l"
echo ""
echo "To stop:"
echo "  kill ${MAIN_PID}"
echo ""
echo "=========================================="

