# Running Full RAG Regeneration (Q1: 300 tokens, Q2-Q4: 500 tokens)

## Quick Start

### Option 1: Use the automated script (recommended)
```bash
cd /home/himanshu/dev/code
./run_full_rag_regeneration_q1_300.sh
```

This will:
- Start the regeneration process in background
- Start GPU monitoring automatically
- Show you the log file paths and PIDs

### Option 2: Manual execution

1. **Start GPU monitoring** (in a separate terminal):
```bash
cd /home/himanshu/dev/code
./monitor_gpu.sh /home/himanshu/dev/output/qwen_rag/logs/gpu_monitor_$(date +%Y%m%d_%H%M%S).log
```

2. **Start the regeneration** (in another terminal):
```bash
cd /home/himanshu/dev/code
nohup python3 multimodal_qa_runner_vllm.py \
    --input-dir /home/himanshu/dev/output/qwen_regenerated \
    --output-dir /home/himanshu/dev/output/qwen_rag \
    --qa-dir /home/himanshu/dev/test/data/processed/qa_pairs_individual_components \
    --chromadb-path /home/himanshu/dev/data/chromadb \
    --batch-size 10 \
    --max-new-tokens 500 \
    > /home/himanshu/dev/output/qwen_rag/logs/rag_regeneration_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Monitoring

### Check main log:
```bash
tail -f /home/himanshu/dev/output/qwen_rag/logs/rag_regeneration_*.log
```

### Check GPU usage:
```bash
tail -f /home/himanshu/dev/output/qwen_rag/logs/gpu_monitor_*.log
```

### Check if process is running:
```bash
ps aux | grep multimodal_qa_runner_vllm.py
```

### Check GPU status (real-time):
```bash
watch -n 1 nvidia-smi
```

## Token Settings

- **Q1 (image-based)**: `max_tokens=300` (matches OpenAI Q1 settings)
- **Q2-Q4 (text-based)**: `max_tokens=500` (matches OpenAI comprehensive settings)

## Expected Output

- **Output directory**: `/home/himanshu/dev/output/qwen_rag/`
- **Logs directory**: `/home/himanshu/dev/output/qwen_rag/logs/`
- **Summary file**: `/home/himanshu/dev/output/qwen_rag/rag_regeneration_summary.json`

## Stopping the Process

1. Find the process ID:
```bash
ps aux | grep multimodal_qa_runner_vllm.py
```

2. Kill the process:
```bash
kill <PID>
```

3. If using the automated script, it will automatically stop GPU monitoring when the main process exits.

## Expected Duration

- Processing 178 files with batch size 10
- Estimated: ~2-3 hours (depending on GPU performance)
- Each file: ~4 questions (1 vision + 3 text)
- Total: ~712 questions

