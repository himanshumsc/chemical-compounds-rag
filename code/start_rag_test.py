#!/usr/bin/env python3
"""Start RAG test in background"""
import subprocess
import os
import sys
from pathlib import Path

# Change to code directory
os.chdir('/home/himanshu/dev/code')

# Create output directory
output_dir = Path('/home/himanshu/dev/output/qwen_rag')
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / 'logs').mkdir(parents=True, exist_ok=True)

# Command to run
cmd = [
    'python3',
    'multimodal_qa_runner_vllm.py',
    '--test-limit', '3',
    '--input-dir', '/home/himanshu/dev/output/qwen_regenerated',
    '--output-dir', '/home/himanshu/dev/output/qwen_rag',
    '--qa-dir', '/home/himanshu/dev/test/data/processed/qa_pairs_individual_components'
]

log_file = output_dir / 'test_run.log'

print("="*70)
print("Starting RAG Test in Background")
print("="*70)
print(f"Command: {' '.join(cmd)}")
print(f"Log file: {log_file}")
print("="*70)

# Start process in background
with open(log_file, 'w') as f:
    process = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd='/home/himanshu/dev/code',
        env={**os.environ, 'PATH': '/home/himanshu/dev/code/.venv_phi4_req/bin:' + os.environ.get('PATH', '')}
    )

print(f"\n✓ Process started with PID: {process.pid}")
print(f"\nMonitor logs with:")
print(f"  tail -f {log_file}")
print(f"\nOr check status with:")
print(f"  ps -p {process.pid}")
print("\nProcess will continue running even if SSH disconnects!")


