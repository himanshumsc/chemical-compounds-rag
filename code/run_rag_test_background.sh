#!/bin/bash
# Run RAG test in background

cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate

# Create output directory if it doesn't exist
mkdir -p /home/himanshu/dev/output/qwen_rag/logs

# Run in background with nohup
nohup python3 multimodal_qa_runner_vllm.py \
  --test-limit 3 \
  --input-dir /home/himanshu/dev/output/qwen_regenerated \
  --output-dir /home/himanshu/dev/output/qwen_rag \
  --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components \
  > /home/himanshu/dev/output/qwen_rag/test_run.log 2>&1 &

# Get the PID
PID=$!
echo "Test started in background with PID: $PID"
echo "Monitor logs with: tail -f /home/himanshu/dev/output/qwen_rag/test_run.log"
echo "Or check detailed logs: tail -f /home/himanshu/dev/output/qwen_rag/logs/rag_regeneration_*.log"
echo ""
echo "To check if still running: ps -p $PID"
echo "To stop: kill $PID"


