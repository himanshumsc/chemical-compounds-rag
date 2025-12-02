# Gemma3 QA Generation - Monitoring Guide

## Starting the Background Process

Run with `nohup` to continue even if SSH disconnects:

```bash
cd /home/himanshu/dev
nohup bash code/run_gemma3_qa_background.sh > /tmp/gemma3_qa_nohup.log 2>&1 &
echo $! > /tmp/gemma3_qa.pid  # Save PID for later reference
```

Or use `screen` (recommended for interactive monitoring):

```bash
screen -S gemma3_qa
cd /home/himanshu/dev
bash code/run_gemma3_qa_background.sh
# Press Ctrl+A then D to detach
# Reattach later with: screen -r gemma3_qa
```

## Checking Progress

### 1. Check if Process is Running

```bash
# Check by PID (if saved)
ps -p $(cat /tmp/gemma3_qa.pid 2>/dev/null) || echo "Process not found"

# Check by process name
ps aux | grep run_gemma3_qa_background.sh | grep -v grep

# Check Python process
ps aux | grep multimodal_qa_runner_gemma3.py | grep -v grep
```

### 2. Monitor Log Files in Real-Time

```bash
# Main background runner log
tail -f /home/himanshu/dev/output/gemma3_rag_concise/logs/background_run_*.log

# Latest full generation log
tail -f /home/himanshu/dev/output/gemma3_rag_concise/logs/full_run_*.log

# Latest regeneration log (from multimodal_qa_runner)
tail -f /home/himanshu/dev/output/gemma3_rag_concise/logs/rag_regeneration_*.log | tail -1
```

### 3. Count Generated Output Files

```bash
# Total files generated
ls -1 /home/himanshu/dev/output/gemma3_rag_concise/*__answers.json 2>/dev/null | wc -l

# List recently generated files
ls -lt /home/himanshu/dev/output/gemma3_rag_concise/*__answers.json | head -10

# Expected total (should match input directory)
ls -1 /home/himanshu/dev/output/qwen_regenerated/*__answers.json | wc -l
```

### 4. Check Summary File

```bash
# View summary (updated after each batch)
cat /home/himanshu/dev/output/gemma3_rag_concise/rag_regeneration_summary.json | jq .

# Or without jq
cat /home/himanshu/dev/output/gemma3_rag_concise/rag_regeneration_summary.json
```

### 5. Monitor GPU Usage

```bash
# Current GPU status
nvidia-smi

# Watch GPU in real-time
watch -n 2 nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 6. Check Latest Log Entries

```bash
# Last 50 lines of latest log
tail -50 $(ls -t /home/himanshu/dev/output/gemma3_rag_concise/logs/rag_regeneration_*.log | head -1)

# Search for errors
grep -i error $(ls -t /home/himanshu/dev/output/gemma3_rag_concise/logs/*.log | head -1) | tail -20

# Search for successful saves
grep "✓ Saved" $(ls -t /home/himanshu/dev/output/gemma3_rag_concise/logs/rag_regeneration_*.log | head -1) | tail -10
```

### 7. Quick Status Check Script

```bash
#!/bin/bash
# Quick status check
OUTPUT_DIR="/home/himanshu/dev/output/gemma3_rag_concise"
echo "=== Gemma3 QA Generation Status ==="
echo "Generated files: $(ls -1 ${OUTPUT_DIR}/*__answers.json 2>/dev/null | wc -l)"
echo "Process running: $(ps aux | grep -c '[m]ultimodal_qa_runner_gemma3.py' || echo 'No')"
echo "Latest log: $(ls -t ${OUTPUT_DIR}/logs/rag_regeneration_*.log 2>/dev/null | head -1)"
echo "Last update: $(stat -c %y $(ls -t ${OUTPUT_DIR}/*__answers.json 2>/dev/null | head -1) 2>/dev/null || echo 'N/A')"
```

## Stopping the Process

```bash
# Find and kill the process
pkill -f multimodal_qa_runner_gemma3.py

# Or kill by PID
kill $(cat /tmp/gemma3_qa.pid 2>/dev/null)

# Kill all related processes
pkill -f run_gemma3_qa_background.sh
pkill -f multimodal_qa_runner_gemma3.py
```

## Expected Output Structure

```
/home/himanshu/dev/output/gemma3_rag_concise/
├── *_answers.json          # Generated answer files (178 total expected)
├── rag_regeneration_summary.json  # Final summary
└── logs/
    ├── background_run_*.log        # Main runner log
    ├── vllm_test_*.log            # vLLM initialization test
    ├── sample_run_*.log           # Sample run log
    ├── full_run_*.log              # Full generation log
    └── rag_regeneration_*.log       # Detailed regeneration log
```

