import subprocess
import os
import sys
import time

# Define paths
VENV_PYTHON = "/home/himanshu/dev/code/.venv_phi4_req/bin/python3"
SCRIPT_PATH = "/home/himanshu/dev/code/multimodal_qa_runner_vllm.py"
INPUT_DIR = "/home/himanshu/dev/output/qwen_regenerated"
OUTPUT_DIR = "/home/himanshu/dev/output/qwen_rag"
QA_DIR = "/home/himanshu/dev/test/data/processed/qa_pairs_individual_components"
LOG_FILE = "/home/himanshu/dev/output/qwen_rag/test_run.log"

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)

print(f"Starting RAG test in background...")
print(f"Log file: {LOG_FILE}")

# Open log file
with open(LOG_FILE, "w") as log_f:
    # Launch process
    # start_new_session=True creates a new process group, similar to nohup behavior
    process = subprocess.Popen(
        [
            VENV_PYTHON, 
            SCRIPT_PATH,
            "--test-limit", "3",
            "--input-dir", INPUT_DIR,
            "--output-dir", OUTPUT_DIR,
            "--qa-dir", QA_DIR
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd="/home/himanshu/dev/code"
    )

print(f"Process started with PID: {process.pid}")
print(f"You can monitor the logs using: tail -f {LOG_FILE}")


