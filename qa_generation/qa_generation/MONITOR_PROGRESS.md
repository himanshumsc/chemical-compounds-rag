# Monitor Comprehensive QA Update Progress

## Process Status
- **Script**: `generate_qa_pairs_comprehensive_update.py`
- **Log File**: `comprehensive_qa_update_20251122_113209.log`
- **Process ID**: Check with `ps aux | grep generate_qa_pairs_comprehensive_update`

## Monitor Progress

### Real-time monitoring (follows log as it updates):
```bash
cd /home/himanshu/dev/qa_generation
tail -f comprehensive_qa_update_20251122_113209.log
```

### Check last 50 lines:
```bash
cd /home/himanshu/dev/qa_generation
tail -50 comprehensive_qa_update_20251122_113209.log
```

### Check progress count:
```bash
cd /home/himanshu/dev/qa_generation
grep "Processing.*/" comprehensive_qa_update_20251122_113209.log | tail -5
```

### Check for errors:
```bash
cd /home/himanshu/dev/qa_generation
grep -i "error\|failed\|exception" comprehensive_qa_update_20251122_113209.log
```

### Check completion status:
```bash
cd /home/himanshu/dev/qa_generation
grep "COMPREHENSIVE QA UPDATE RESULTS" comprehensive_qa_update_20251122_113209.log
```

## Process Management

### Check if process is still running:
```bash
ps aux | grep generate_qa_pairs_comprehensive_update | grep -v grep
```

### Kill process if needed (use PID from ps command):
```bash
kill <PID>
```

## Settings
- **Max Tokens**: 500 (updated from 2000)
- **Total Files**: 178
- **Estimated Time**: ~25 minutes
- **Runs in background**: Yes (survives SSH disconnect)

## Output Location
- **Updated Files**: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive/`

