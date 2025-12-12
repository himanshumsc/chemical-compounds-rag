# Instructions: Running QWEN Answer Regeneration

## Quick Start

### Test with Small Sample (5 files)

```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate  # or appropriate venv
python multimodal_qa_runner_vllm.py \
    --input-dir ../output/qwen_regenerated \
    --qa-dir ../test/data/processed/qa_pairs_individual_components \
    --max-new-tokens 500 \
    --test-limit 5 \
    --batch-size 5
```

### Run Full Batch (All 178 files) in Background

```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate
nohup python multimodal_qa_runner_vllm.py \
    --input-dir ../output/qwen_regenerated \
    --qa-dir ../test/data/processed/qa_pairs_individual_components \
    --max-new-tokens 500 \
    --batch-size 10 \
    > ../output/qwen_regenerated/logs/regeneration_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > ../output/qwen_regenerated/regeneration.pid
echo "Process started. PID: $(cat ../output/qwen_regenerated/regeneration.pid)"
```

## Monitoring Progress

### Real-time Log Monitoring
```bash
cd /home/himanshu/dev/output/qwen_regenerated
tail -f logs/regeneration_*.log
```

### Check Current Progress
```bash
cd /home/himanshu/dev/output/qwen_regenerated
# Count processed files
grep "Processing:" logs/regeneration_*.log | wc -l

# Check last processed file
grep "Processing:" logs/regeneration_*.log | tail -1

# Check for errors
grep -i "error\|failed" logs/regeneration_*.log
```

### Check Process Status
```bash
# Check if process is running
ps aux | grep multimodal_qa_runner_vllm | grep -v grep

# Check PID from file
cat output/qwen_regenerated/regeneration.pid
ps -p $(cat output/qwen_regenerated/regeneration.pid)
```

### Check Summary
```bash
cd /home/himanshu/dev/output/qwen_regenerated
cat regeneration_summary.json | python3 -m json.tool
```

## Script Features

### vLLM Optimization
- **Q2-Q4 (text-only)**: Uses vLLM for faster inference (if available)
- **Q1 (with image)**: Uses Transformers (vLLM doesn't support vision)
- **Fallback**: Automatically falls back to Transformers if vLLM unavailable

### Token Limit
- **Max tokens**: 500 (matching OpenAI calls)
- **Previous**: 128 tokens
- **Improvement**: 2-3x longer answers

### Input/Output
- **Input**: Reads questions from existing answer files
- **Output**: Updates same files with new answers
- **Preserves**: Metadata, file structure

### Batch Processing
- Processes files in batches (default: 10 files per batch)
- Groups Q2-Q4 for efficient vLLM batching
- Q1 processed individually (requires images)

## Expected Performance

### With vLLM
- **Q1 (vision)**: ~3-4 seconds (Transformers)
- **Q2-Q4 (text)**: ~1.5-2 seconds each (vLLM)
- **Per file**: ~10-12 seconds
- **Total (178 files)**: ~30-35 minutes

### Without vLLM (Transformers only)
- **All questions**: ~3-4 seconds each
- **Per file**: ~14-16 seconds
- **Total (178 files)**: ~42-48 minutes

## Troubleshooting

### vLLM Not Available
- Script automatically falls back to Transformers
- Check log for: "WARNING: vLLM not available"
- All questions will use Transformers

### Out of Memory
- Reduce batch_size (e.g., --batch-size 5)
- vLLM may use more memory than Transformers

### Process Stopped
- Check logs for errors
- Restart with same command
- Already processed files won't be reprocessed (but script doesn't skip them yet)

## Files Modified

- **Input**: `output/qwen_regenerated/*__answers.json`
- **Logs**: `output/qwen_regenerated/logs/regeneration_*.log`
- **Summary**: `output/qwen_regenerated/regeneration_summary.json`
- **PID**: `output/qwen_regenerated/regeneration.pid`

## Next Steps After Test

1. Review test results (5 files)
2. Check answer quality and length
3. Verify token usage (~500 max)
4. If satisfied, run full batch
5. Monitor progress using commands above

