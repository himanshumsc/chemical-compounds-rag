# Monitor RAG Test Progress

## Test Running in Background

The test is running with 3 sample files in the background. Even if your SSH connection disconnects, it will continue running.

## Monitor Logs

### Real-time log monitoring:
```bash
tail -f /home/himanshu/dev/output/qwen_rag/test_run.log
```

### Check latest log file in logs directory:
```bash
tail -f /home/himanshu/dev/output/qwen_rag/logs/rag_regeneration_*.log
```

### View last 50 lines:
```bash
tail -n 50 /home/himanshu/dev/output/qwen_rag/test_run.log
```

## Check Process Status

```bash
# Check if process is still running
ps aux | grep "multimodal_qa_runner_vllm.py" | grep -v grep

# Check process details
pgrep -af "multimodal_qa_runner_vllm.py"
```

## Check Output Files

```bash
# List generated files
ls -lh /home/himanshu/dev/output/qwen_rag/*.json 2>/dev/null

# Check a specific file
cat /home/himanshu/dev/output/qwen_rag/1_13-Butadiene__answers.json | jq '.rag_enabled, .answers[0].rag_used, .answers[0].n_chunks_used'
```

## Expected Output

After completion, you should see:
- 3 JSON files in `/home/himanshu/dev/output/qwen_rag/`
- Log file: `/home/himanshu/dev/output/qwen_rag/test_run.log`
- Detailed logs: `/home/himanshu/dev/output/qwen_rag/logs/rag_regeneration_*.log`
- Summary: `/home/himanshu/dev/output/qwen_rag/rag_regeneration_summary.json`

## Stop the Process (if needed)

```bash
# Find and kill the process
pkill -f "multimodal_qa_runner_vllm.py"

# Or find PID first
ps aux | grep "multimodal_qa_runner_vllm.py" | grep -v grep
kill <PID>
```


