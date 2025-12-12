#!/bin/bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate
python3 multimodal_qa_runner_vllm.py \
  --test-limit 3 \
  --input-dir /home/himanshu/dev/output/qwen_regenerated \
  --output-dir /home/himanshu/dev/output/qwen_rag \
  --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components

