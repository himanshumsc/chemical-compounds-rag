# Run RAG Test Now

## Quick Test Command

Execute this command to test with 3 sample files:

```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate
python3 multimodal_qa_runner_vllm.py \
  --test-limit 3 \
  --input-dir /home/himanshu/dev/output/qwen_regenerated \
  --output-dir /home/himanshu/dev/output/qwen_rag \
  --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components
```

## Or Use the Shell Script

```bash
cd /home/himanshu/dev/code
./run_rag_test.sh
```

## What to Expect

1. **Initialization**: 
   - vLLM loading
   - ChromaDB search system initialization
   - Model loading

2. **Processing 3 files**:
   - `1_13-Butadiene__answers.json`
   - `2_2-4-Isobutylphenylpropionic_Acid_Ibuprofen__answers.json`
   - `3_22'-Dichlorodiethyl_Sulfide_Mustard_Gas__answers.json`

3. **For each file**:
   - Q1: Image similarity search → Generate with RAG context
   - Q2-Q4: Text search → Generate with RAG context

4. **Output**: Updated JSON files with RAG metadata

## Verify Results

After running, check one of the files:

```bash
cat /home/himanshu/dev/output/qwen_rag/1_13-Butadiene__answers.json | jq '.rag_enabled, .answers[0].rag_used, .answers[0].n_chunks_used'
```

Expected:
- `rag_enabled`: `true`
- `rag_used`: `true`
- `n_chunks_used`: `5` (or number retrieved)

## Check Logs

Logs will be saved to:
```
/home/himanshu/dev/output/qwen_rag/logs/rag_regeneration_YYYYMMDD_HHMMSS.log
```

## Output Directory Structure

```
dev/output/qwen_rag/
├── 1_13-Butadiene__answers.json
├── 2_2-4-Isobutylphenylpropionic_Acid_Ibuprofen__answers.json
├── ...
├── logs/
│   └── rag_regeneration_YYYYMMDD_HHMMSS.log
└── rag_regeneration_summary.json
```

