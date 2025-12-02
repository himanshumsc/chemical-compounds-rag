#!/bin/bash
# Quick script to check RAG test status and logs

echo "=== RAG Test Status ==="
echo ""

# Check if process is running
if pgrep -f "multimodal_qa_runner_vllm.py" > /dev/null; then
    echo "✓ Test is RUNNING"
    echo "PID: $(pgrep -f 'multimodal_qa_runner_vllm.py')"
else
    echo "✗ Test is NOT running (may have completed or failed)"
fi

echo ""
echo "=== Latest Log Output (last 20 lines) ==="
if [ -f /home/himanshu/dev/output/qwen_rag/test_run.log ]; then
    tail -n 20 /home/himanshu/dev/output/qwen_rag/test_run.log
else
    echo "Log file not found yet"
fi

echo ""
echo "=== Generated Files ==="
if [ -d /home/himanshu/dev/output/qwen_rag ]; then
    ls -lh /home/himanshu/dev/output/qwen_rag/*.json 2>/dev/null | wc -l | xargs echo "JSON files found:"
    ls -lh /home/himanshu/dev/output/qwen_rag/*.json 2>/dev/null | head -5
else
    echo "Output directory not created yet"
fi

echo ""
echo "=== Monitor Commands ==="
echo "Real-time logs: tail -f /home/himanshu/dev/output/qwen_rag/test_run.log"
echo "Detailed logs: tail -f /home/himanshu/dev/output/qwen_rag/logs/rag_regeneration_*.log"


