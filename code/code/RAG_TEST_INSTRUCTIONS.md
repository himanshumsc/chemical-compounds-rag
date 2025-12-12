# RAG Integration Test Instructions

## Overview
The `multimodal_qa_runner_vllm.py` script has been updated to integrate ChromaDB RAG for all 4 questions:
- **Q1**: Image-only similarity search (finds similar molecular structures)
- **Q2-Q4**: Text search (finds relevant compound information)

## Token Limit
- **Generation limit**: 500 tokens (`max_new_tokens=500`)
- **Model context**: 8192 tokens (`max_model_len=8192`)

## Test with 3 Sample Files

## Test with 3 Sample Files

```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate
python3 multimodal_qa_runner_vllm.py \
  --test-limit 3 \
  --input-dir /home/himanshu/dev/output/qwen_regenerated \
  --output-dir /home/himanshu/dev/output/qwen_rag \
  --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components
```

## Expected Output Changes

After running with RAG, answer files will include:

1. **File-level metadata**:
   ```json
   {
     "rag_enabled": true,
     "rag_regenerated_at": "2025-11-23 HH:MM:SS",
     "max_tokens": 500,
     ...
   }
   ```

2. **Answer-level metadata** (for each answer):
   ```json
   {
     "question": "...",
     "answer": "...",
     "latency_s": 10.15,
     "rag_used": true,
     "search_type": "image",  // or "text" for Q2-Q4
     "n_chunks_used": 5
   }
   ```

## Full Batch Run (178 files)

```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate
nohup python3 multimodal_qa_runner_vllm.py \
  --input-dir /home/himanshu/dev/output/qwen_regenerated \
  --output-dir /home/himanshu/dev/output/qwen_rag \
  --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components \
  > /home/himanshu/dev/output/qwen_rag/logs/rag_regeneration.log 2>&1 &
```

## Monitor Progress

```bash
# Watch logs in real-time
tail -f /home/himanshu/dev/output/qwen_rag/logs/rag_regeneration_*.log

# Check latest log file
ls -lt /home/himanshu/dev/output/qwen_rag/logs/ | head -5
```

## Verify RAG is Working

After running, check an answer file:
```bash
cat /home/himanshu/dev/output/qwen_rag/37_Carbon_Dioxide__answers.json | jq '.rag_enabled, .answers[0].rag_used, .answers[0].n_chunks_used'
```

Expected output:
- `rag_enabled`: `true`
- `rag_used`: `true` (for each answer)
- `n_chunks_used`: `5` (or number of chunks retrieved)

## Troubleshooting

1. **ChromaDB not found**: Ensure `/home/himanshu/dev/data/chromadb` exists
2. **Import errors**: Check that `chromadb_search.py` is in the same directory
3. **GPU memory**: RAG uses CPU for ChromaDB, shouldn't affect GPU memory
4. **Disable RAG**: Use `--no-rag` flag to test without RAG

## Configuration

- **ChromaDB path**: Default `/home/himanshu/dev/data/chromadb` (can override with `--chromadb-path`)
- **Number of chunks**: Fixed at 5 chunks per question
- **Search types**:
  - Q1: `image_similarity_search` (image-only)
  - Q2-Q4: `text_search` (text-based)

