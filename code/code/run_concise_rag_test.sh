#!/bin/bash
# Test script for concise RAG regeneration with character limits

cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate

echo "=========================================="
echo "Testing Concise RAG Regeneration"
echo "=========================================="
echo "Input: /home/himanshu/dev/output/qwen_regenerated"
echo "Output: /home/himanshu/dev/output/qwen_rag_concise"
echo "Character Limits: Q1=600, Q2=1000, Q3=1800, Q4=2000"
echo "Test: 3 sample compounds"
echo "=========================================="

python3 multimodal_qa_runner_vllm.py \
    --test-limit 3 \
    --input-dir /home/himanshu/dev/output/qwen_regenerated \
    --output-dir /home/himanshu/dev/output/qwen_rag_concise \
    --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components \
    --chromadb-path /home/himanshu/dev/data/chromadb \
    --batch-size 3

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo "Check results in: /home/himanshu/dev/output/qwen_rag_concise"
echo "Check logs in: /home/himanshu/dev/output/qwen_rag_concise/logs/"
echo "=========================================="

