#!/bin/bash
# Run full RAG regeneration for all 178 files in the background
# This script will run the regeneration process with vLLM + RAG for all files

cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate

# Create output directory if it doesn't exist
mkdir -p /home/himanshu/dev/output/qwen_rag

# Run in background with nohup
# Remove --test-limit to process all files
nohup python3 multimodal_qa_runner_vllm.py --input-dir /home/himanshu/dev/output/qwen_regenerated --output-dir /home/himanshu/dev/output/qwen_rag --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components --chromadb-path /home/himanshu/dev/data/chromadb --batch-size 10 > /home/himanshu/dev/output/qwen_rag/full_regeneration.log 2>&1 &

echo "Full regeneration started in background"
echo "Process ID: $!"
echo "Log file: /home/himanshu/dev/output/qwen_rag/full_regeneration.log"
echo ""
echo "To monitor progress:"
echo "  tail -f /home/himanshu/dev/output/qwen_rag/full_regeneration.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep multimodal_qa_runner_vllm.py | grep -v grep"
echo ""
echo "To stop the process:"
echo "  pkill -f multimodal_qa_runner_vllm.py"

