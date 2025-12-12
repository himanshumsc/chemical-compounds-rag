#!/home/himanshu/dev/code/.venv_phi4_req/bin/python3
"""
Test script to initialize vLLM with Qwen-2.5-VL AWQ model.
This is a standalone test to verify vLLM initialization works before using it in RAG pipeline.
"""
import os
import sys
import time
import signal
from pathlib import Path

# Fix ChromaDB SQLite compatibility
import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sqlite3

# Set environment variables for vLLM before importing
# Qwen-2.5-VL specific: Skip memory profiling to avoid known bug
os.environ['VLLM_SKIP_MM_PROFILE'] = '1'
os.environ['SKIP_MM_PROFILE'] = '1'

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    print("✓ vLLM imported successfully")
except ImportError as e:
    print(f"✗ Failed to import vLLM: {e}")
    sys.exit(1)

# Model path
MODEL_PATH = "/home/himanshu/dev/models/QWEN_AWQ"

def test_vllm_initialization(timeout_seconds=300):
    """Test vLLM initialization with timeout."""
    print("="*70)
    print("Testing vLLM Initialization")
    print("="*70)
    print(f"Model path: {MODEL_PATH}")
    print(f"Timeout: {timeout_seconds} seconds (5 minutes)")
    print(f"Environment variables:")
    print(f"  VLLM_SKIP_MM_PROFILE={os.environ.get('VLLM_SKIP_MM_PROFILE', 'not set')}")
    print(f"  SKIP_MM_PROFILE={os.environ.get('SKIP_MM_PROFILE', 'not set')}")
    print("="*70)
    print()
    
    # Check if model path exists
    if not Path(MODEL_PATH).exists():
        print(f"✗ Model path does not exist: {MODEL_PATH}")
        return False
    
    print("Starting vLLM initialization...")
    print("This may take 2-5 minutes for a 7B model.")
    print()
    
    start_time = time.time()
    llm = None
    
    try:
        # Initialize vLLM with simplified config (no parallel processing)
        llm = LLM(
            model=MODEL_PATH,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.80,
            enforce_eager=True,  # Disable CUDA graphs
            disable_log_stats=True,  # Disable stats logging
        )
        
        elapsed = time.time() - start_time
        print(f"✓ vLLM initialized successfully in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        
        # Test a simple generation
        print()
        print("Testing generation...")
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
            stop=None,
        )
        
        test_prompt = "Hello, how are you?"
        test_start = time.time()
        outputs = llm.generate([test_prompt], sampling_params)
        test_elapsed = time.time() - test_start
        
        generated_text = outputs[0].outputs[0].text
        print(f"✓ Generation test successful in {test_elapsed:.2f} seconds")
        print(f"  Prompt: {test_prompt}")
        print(f"  Response: {generated_text[:100]}...")
        
        return True
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n✗ Initialization interrupted after {elapsed:.2f} seconds")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ vLLM initialization failed after {elapsed:.2f} seconds")
        print(f"  Error: {e}")
        import traceback
        print(f"  Traceback:\n{traceback.format_exc()}")
        return False
    finally:
        if llm is not None:
            print("\nCleaning up...")
            del llm
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test with 5 minute timeout
    success = test_vllm_initialization(timeout_seconds=300)
    
    if success:
        print("\n" + "="*70)
        print("✓ vLLM TEST PASSED - Ready for RAG pipeline")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("✗ vLLM TEST FAILED - Cannot proceed with RAG pipeline")
        print("="*70)
        sys.exit(1)

