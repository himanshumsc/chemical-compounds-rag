#!/bin/bash
# Monitor Qwen-Gemma comparison test progress

LOG_FILE="/tmp/qwen_gemma_full_test.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "=== Qwen-Gemma Comparison Test Monitor ==="
echo ""

# Check if process is running
if pgrep -f "test_qwen_with_gemma_context" > /dev/null; then
    echo "✓ Test is RUNNING"
else
    echo "✗ Test is NOT running (may have completed)"
fi

echo ""

# Get latest progress
echo "Latest progress:"
tail -3 "$LOG_FILE" | grep -E "\[.*/.*\]" | tail -1

echo ""

# Count completed batches
BATCHES=$(grep -c "Processing batch" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Batches processed: $BATCHES"

echo ""

# Show last few lines
echo "Last 5 log lines:"
tail -5 "$LOG_FILE"

echo ""
echo "To watch live progress: tail -f $LOG_FILE"

