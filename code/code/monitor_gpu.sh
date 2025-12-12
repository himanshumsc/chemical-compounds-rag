#!/bin/bash
# Standalone GPU monitoring script
# Usage: ./monitor_gpu.sh [output_file]

OUTPUT_FILE="${1:-/home/himanshu/dev/output/qwen_rag/logs/gpu_monitor_$(date +%Y%m%d_%H%M%S).log}"

echo "GPU Monitoring started at $(date)"
echo "Output: ${OUTPUT_FILE}"
echo "Press Ctrl+C to stop"
echo ""

# Write header
echo "timestamp,gpu_util%,mem_util%,mem_used_mb,mem_total_mb,temp_c" > "${OUTPUT_FILE}"

# Monitor loop
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | tr ',' ' ')
    echo "${TIMESTAMP},${GPU_STATS}" >> "${OUTPUT_FILE}"
    # Also print to console
    echo "[${TIMESTAMP}] GPU: ${GPU_STATS}"
    sleep 5
done

