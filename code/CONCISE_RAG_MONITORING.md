# Concise RAG Regeneration - Monitoring Guide

**Process Started:** November 23, 2025 11:23:31  
**Main Process PID:** 41626  
**Output Directory:** `/home/himanshu/dev/output/qwen_rag_concise`

## Configuration

- **Character Limits:** Q1=600, Q2=1000, Q3=1800, Q4=2000
- **Max Tokens:** Q1=200, Q2=333, Q3=600, Q4=666 (division by 3.0)
- **Input:** `/home/himanshu/dev/output/qwen_regenerated`
- **Total Files:** 178 compounds

## Monitoring Commands

### Check Main Progress
```bash
tail -f /home/himanshu/dev/output/qwen_rag_concise/logs/concise_rag_regeneration_20251123_112331.log
```

### Check GPU Usage
```bash
tail -f /home/himanshu/dev/output/qwen_rag_concise/logs/gpu_monitor_20251123_112331.log
```

### Check if Process is Running
```bash
ps aux | grep 41626
```

### Count Completed Files
```bash
ls -1 /home/himanshu/dev/output/qwen_rag_concise/*__answers.json 2>/dev/null | wc -l
```

### Check Recent Output Files
```bash
ls -lt /home/himanshu/dev/output/qwen_rag_concise/*__answers.json | head -10
```

### Real-time GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Check Answer Lengths (Sample)
```bash
cd /home/himanshu/dev/output/qwen_rag_concise
for f in *__answers.json | head -5; do
  echo "=== $f ==="
  jq -r '.answers[] | "Q\(.question_index + 1): \(.answer | length) chars (limit: \(.char_limit))"' "$f"
done
```

## Expected Duration

- **Estimated Time:** 2-3 hours for 178 files
- **Average per file:** ~9-10 seconds
- **Batch size:** 10 files per batch

## Stop Process

If needed to stop:
```bash
kill 41626
```

## Verification

After completion, verify:
1. All 178 files generated
2. Answers are within character limits
3. No truncation warnings in logs
4. Summary file created: `rag_regeneration_summary.json`

