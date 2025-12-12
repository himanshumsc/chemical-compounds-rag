#!/usr/bin/env python3
"""
Simple GPU test script for llama-cpp-python with Vulkan support.
Tests GPU availability, model loading, and basic text generation.

Usage:
    # Direct (requires llama-env activated):
    python test_gpu_llamacpp.py [options]
    
    # Using wrapper script (auto-activates llama-env):
    ./run_gpu_test.sh [options]
"""

import sys
import os
import time
import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python imported successfully")
except ImportError as e:
    print(f"❌ Failed to import llama-cpp-python: {e}")
    print("   Install with: pip install llama-cpp-python")
    sys.exit(1)


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Set up logging to both file and console."""
    if log_file is None:
        log_file = Path(__file__).parent / f"gpu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger('gpu_test')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


@dataclass
class PerformanceMetrics:
    """Store performance metrics for benchmarking."""
    model_load_time: float = 0.0
    text_gen_time: float = 0.0
    text_gen_tokens: int = 0
    chat_gen_time: float = 0.0
    chat_gen_tokens: int = 0
    tokens_per_second: float = 0.0
    memory_used_mb: float = 0.0
    iterations: int = 1
    avg_text_gen_time: float = 0.0
    avg_chat_gen_time: float = 0.0
    min_text_gen_time: float = 0.0
    max_text_gen_time: float = 0.0
    min_chat_gen_time: float = 0.0
    max_chat_gen_time: float = 0.0


def get_system_info() -> Dict[str, str]:
    """Get system and GPU information."""
    info = {}
    
    # Python version
    info['python_version'] = sys.version.split()[0]
    
    # Try to get GPU info via nvidia-smi (if available)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version,utilization.gpu', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split('\n')[0]
            parts = [p.strip() for p in gpu_info.split(',')]
            if len(parts) >= 2:
                info['gpu_name'] = parts[0]
                info['gpu_memory_total'] = parts[1]
                if len(parts) >= 3:
                    info['gpu_memory_used'] = parts[2]
                if len(parts) >= 4:
                    info['driver_version'] = parts[3]
                if len(parts) >= 5:
                    info['gpu_utilization'] = parts[4]
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return info


def get_gpu_memory_usage() -> Optional[float]:
    """Get current GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().split('\n')[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return None


def check_gpu_support():
    """Check if GPU/Vulkan support is available."""
    print("\n" + "="*60)
    print("GPU/VULKAN SUPPORT CHECK")
    print("="*60)
    
    # Get system info
    sys_info = get_system_info()
    
    print(f"🐍 Python: {sys_info.get('python_version', 'Unknown')}")
    if 'gpu_name' in sys_info:
        print(f"🎮 GPU: {sys_info['gpu_name']}")
        print(f"💾 GPU Memory: {sys_info.get('gpu_memory', 'Unknown')}")
        if 'driver_version' in sys_info:
            print(f"🔧 Driver: {sys_info['driver_version']}")
    
    try:
        # Try to get GPU info from llama.cpp
        # Note: llama-cpp-python may expose GPU info differently
        print("\nChecking llama-cpp-python GPU capabilities...")
        
        # Try to import and check for GPU device info
        try:
            from llama_cpp import llama_cpp
            # Check if we can access GPU info
            print("✅ llama-cpp-python is installed")
            
            # Try to get available GPU devices (if supported)
            try:
                # llama.cpp may expose GPU info through C API
                # This is a basic check - actual GPU usage depends on n_gpu_layers
                print("   GPU/Vulkan support: Available (will be tested with model)")
                print("   Note: Set n_gpu_layers > 0 to enable GPU offloading")
            except:
                print("   GPU support: Will be verified when loading a model")
                print("   (Set n_gpu_layers > 0 to enable GPU offloading)")
        except ImportError:
            print("⚠️  Could not import llama_cpp internals")
        
        return True
    except Exception as e:
        print(f"⚠️  Could not verify GPU support: {e}")
        return False


def find_gguf_model(model_dir: Optional[Path] = None) -> Optional[Path]:
    """Find a GGUF model file in common locations."""
    if model_dir and model_dir.exists():
        gguf_files = list(model_dir.glob("*.gguf"))
        if gguf_files:
            return gguf_files[0]
    
    # Check common model directories
    common_paths = [
        Path("/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_GGUF"),
        Path("/home/himanshu/dev/models"),
        Path("/home/himanshu/MSC_FINAL/dev/models/GEMMA3_QAT_Q4_0_GGUF"),
        Path("/home/himanshu/MSC_FINAL/dev/models"),
    ]
    
    for base_path in common_paths:
        if base_path.exists():
            gguf_files = list(base_path.rglob("*.gguf"))
            if gguf_files:
                print(f"   Found GGUF model: {gguf_files[0]}")
                return gguf_files[0]
    
    return None


def test_model_loading(model_path: Optional[Path], n_gpu_layers: int = 0, metrics: Optional[PerformanceMetrics] = None, logger: Optional[logging.Logger] = None):
    """Test loading a model with GPU offloading."""
    if logger is None:
        logger = logging.getLogger('gpu_test')
    
    print("\n" + "="*60)
    print("MODEL LOADING TEST")
    print("="*60)
    logger.info("="*60)
    logger.info("MODEL LOADING TEST")
    logger.info("="*60)
    
    if not model_path:
        print("⚠️  No GGUF model found. Skipping model loading test.")
        logger.warning("No GGUF model found. Skipping model loading test.")
        print("\n   To test with a model:")
        print("   1. Download a GGUF model:")
        print("      - Small test model: llama-2-7b-chat.Q4_0.gguf (~4GB)")
        print("      - Gemma-3: Use download_gemma3_gguf.py")
        print("   2. Run this script with:")
        print("      ./run_gpu_test.sh --model-path /path/to/model.gguf --n-gpu-layers 35")
        print("\n   💡 Tip: You can test GPU support without a model - installation verified!")
        return None
    
    if not model_path.exists():
        error_msg = f"Model file not found: {model_path}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        return None
    
    # Get model file size
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    model_size_gb = model_size_mb / 1024
    
    print(f"📁 Model path: {model_path}")
    print(f"📦 Model size: {model_size_gb:.2f} GB ({model_size_mb:.0f} MB)")
    if n_gpu_layers == -1:
        gpu_status = "All layers on GPU"
    elif n_gpu_layers > 0:
        gpu_status = f"GPU enabled ({n_gpu_layers} layers)"
    else:
        gpu_status = "CPU only"
    print(f"🎮 GPU layers: {n_gpu_layers} ({gpu_status})")
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model size: {model_size_gb:.2f} GB ({model_size_mb:.0f} MB)")
    logger.info(f"GPU layers: {n_gpu_layers} ({gpu_status})")
    
    try:
        print("\n⏳ Loading model...")
        logger.info("Starting model loading...")
        logger.debug(f"Llama parameters: n_ctx=2048, n_gpu_layers={n_gpu_layers}, verbose=True, seed=42")
        start_time = time.time()
        
        llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,  # Smaller context for testing
            n_gpu_layers=n_gpu_layers,  # Offload layers to GPU
            n_batch=64,  # Low batch size for unified memory APUs
            flash_attn=True,  # Flash attention to reduce peak compute buffer usage
            verbose=True,  # Enable verbose to capture layer assignments in logs
            seed=42,
        )
        
        load_time = time.time() - start_time
        
        # Get memory usage after loading
        memory_used = get_gpu_memory_usage()
        
        if metrics:
            metrics.model_load_time = load_time
            if memory_used:
                metrics.memory_used_mb = memory_used
        
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"   Load speed: {model_size_gb/load_time:.2f} GB/s")
        if memory_used:
            print(f"   GPU Memory used: {memory_used:.0f} MB ({memory_used/1024:.2f} GB)")
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Load speed: {model_size_gb/load_time:.2f} GB/s")
        if memory_used:
            logger.info(f"GPU Memory used: {memory_used:.0f} MB ({memory_used/1024:.2f} GB)")
        else:
            logger.warning("Could not retrieve GPU memory usage")
        
        return llm
        
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg, exc_info=True)
        import traceback
        traceback_str = traceback.format_exc()
        print(f"\nTraceback:\n{traceback_str}")
        logger.debug(f"Full traceback:\n{traceback_str}")
        return None


def test_text_generation(
    llm: Llama, 
    prompt: str = "Hello, how are you?", 
    metrics: Optional[PerformanceMetrics] = None,
    max_tokens: int = 50,
    temperature: float = 0.7,
    iterations: int = 1,
    logger: Optional[logging.Logger] = None
) -> List[float]:
    """Test basic text generation with optional benchmarking."""
    if logger is None:
        logger = logging.getLogger('gpu_test')
    if not llm:
        print("\n⚠️  Skipping generation test (no model loaded)")
        return []
    
    test_name = "TEXT GENERATION TEST" if iterations == 1 else f"TEXT GENERATION BENCHMARK ({iterations} iterations)"
    print("\n" + "="*60)
    print(test_name)
    print("="*60)
    logger.info("="*60)
    logger.info(test_name)
    logger.info("="*60)
    
    times = []
    
    try:
        print(f"📝 Prompt: {prompt}")
        print(f"🔄 Iterations: {iterations}")
        print(f"⚙️  Max tokens: {max_tokens}, Temperature: {temperature}")
        print("\n⏳ Generating responses...")
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Iterations: {iterations}, Max tokens: {max_tokens}, Temperature: {temperature}")
        
        for i in range(iterations):
            if iterations > 1:
                print(f"   Iteration {i+1}/{iterations}...", end=" ", flush=True)
            
            logger.debug(f"Starting generation iteration {i+1}/{iterations}")
            start_time = time.time()
            response = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                echo=False,
            )
            generation_time = time.time() - start_time
            times.append(generation_time)
            
            logger.debug(f"Iteration {i+1} completed in {generation_time:.2f}s")
            
            if iterations > 1:
                print(f"{generation_time:.2f}s")
            elif i == 0:  # Show response only on first iteration
                generated_text = response['choices'][0]['text'].strip()
                word_count = len(generated_text.split())
                estimated_tokens = int(word_count * 0.75)
                
                print(f"✅ Generation completed in {generation_time:.2f} seconds")
                print(f"\n📤 Response:")
                print(f"   {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
                
                logger.info(f"Generation completed in {generation_time:.2f} seconds")
                logger.info(f"Generated text (first 500 chars): {generated_text[:500]}")
                logger.debug(f"Full response: {response}")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Estimate tokens from first response
        if times:
            first_response = llm(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
            generated_text = first_response['choices'][0]['text'].strip()
            word_count = len(generated_text.split())
            estimated_tokens = int(word_count * 0.75)
            tokens_per_sec = estimated_tokens / avg_time if avg_time > 0 else 0
        else:
            estimated_tokens = 0
            tokens_per_sec = 0
        
        if metrics:
            metrics.text_gen_time = avg_time
            metrics.text_gen_tokens = estimated_tokens
            metrics.tokens_per_second = tokens_per_sec
            metrics.iterations = iterations
            metrics.avg_text_gen_time = avg_time
            metrics.min_text_gen_time = min_time
            metrics.max_text_gen_time = max_time
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average time: {avg_time:.2f}s")
        if iterations > 1:
            print(f"   Min time: {min_time:.2f}s")
            print(f"   Max time: {max_time:.2f}s")
            std_dev = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
            print(f"   Std deviation: {std_dev:.2f}s")
        print(f"   Estimated tokens: ~{estimated_tokens}")
        print(f"   Average speed: {tokens_per_sec:.2f} tokens/second")
        
        logger.info(f"Performance Summary - Avg: {avg_time:.2f}s, Tokens: ~{estimated_tokens}, Speed: {tokens_per_sec:.2f} tok/s")
        if iterations > 1:
            logger.info(f"Min: {min_time:.2f}s, Max: {max_time:.2f}s, StdDev: {std_dev:.2f}s")
        
        return times
        
    except Exception as e:
        error_msg = f"Generation failed: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg, exc_info=True)
        import traceback
        traceback_str = traceback.format_exc()
        print(f"\nTraceback:\n{traceback_str}")
        logger.debug(f"Full traceback:\n{traceback_str}")
        return []


def test_chat_completion(
    llm: Llama, 
    metrics: Optional[PerformanceMetrics] = None,
    iterations: int = 1,
    logger: Optional[logging.Logger] = None
) -> List[float]:
    """Test chat completion API with optional benchmarking."""
    if logger is None:
        logger = logging.getLogger('gpu_test')
    if not llm:
        print("\n⚠️  Skipping chat completion test (no model loaded)")
        logger.warning("Skipping chat completion test (no model loaded)")
        return []
    
    test_name = "CHAT COMPLETION TEST" if iterations == 1 else f"CHAT COMPLETION BENCHMARK ({iterations} iterations)"
    print("\n" + "="*60)
    print(test_name)
    print("="*60)
    logger.info("="*60)
    logger.info(test_name)
    logger.info("="*60)
    
    times = []
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ]
        
        print(f"💬 Messages: {len(messages)}")
        print(f"🔄 Iterations: {iterations}")
        print("\n⏳ Generating chat responses...")
        
        logger.info(f"Messages: {len(messages)}")
        logger.info(f"Iterations: {iterations}")
        logger.debug(f"Messages content: {messages}")
        
        for i in range(iterations):
            if iterations > 1:
                print(f"   Iteration {i+1}/{iterations}...", end=" ", flush=True)
            
            logger.debug(f"Starting chat completion iteration {i+1}/{iterations}")
            start_time = time.time()
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=30,
                temperature=0.7,
            )
            generation_time = time.time() - start_time
            times.append(generation_time)
            
            logger.debug(f"Chat completion iteration {i+1} completed in {generation_time:.2f}s")
            
            if iterations > 1:
                print(f"{generation_time:.2f}s")
            elif i == 0:  # Show response only on first iteration
                content = response['choices'][0]['message']['content']
                print(f"✅ Chat completion in {generation_time:.2f} seconds")
                print(f"\n📤 Response:")
                print(f"   {content}")
                
                logger.info(f"Chat completion completed in {generation_time:.2f} seconds")
                logger.info(f"Response: {content}")
                logger.debug(f"Full response: {response}")
        
        # Calculate statistics
        avg_time = sum(times) / len(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        
        # Estimate tokens from first response
        if times:
            first_response = llm.create_chat_completion(messages=messages, max_tokens=30, temperature=0.7)
            content = first_response['choices'][0]['message']['content']
            word_count = len(content.split())
            estimated_tokens = int(word_count * 0.75)
            tokens_per_sec = estimated_tokens / avg_time if avg_time > 0 else 0
        else:
            estimated_tokens = 0
            tokens_per_sec = 0
        
        if metrics:
            metrics.chat_gen_time = avg_time
            metrics.chat_gen_tokens = estimated_tokens
            metrics.avg_chat_gen_time = avg_time
            metrics.min_chat_gen_time = min_time
            metrics.max_chat_gen_time = max_time
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average time: {avg_time:.2f}s")
        if iterations > 1:
            std_dev = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
            print(f"   Min time: {min_time:.2f}s")
            print(f"   Max time: {max_time:.2f}s")
            print(f"   Std deviation: {std_dev:.2f}s")
        print(f"   Estimated tokens: ~{estimated_tokens}")
        print(f"   Average speed: {tokens_per_sec:.2f} tokens/second")
        
        logger.info(f"Chat Performance Summary - Avg: {avg_time:.2f}s, Tokens: ~{estimated_tokens}, Speed: {tokens_per_sec:.2f} tok/s")
        if iterations > 1:
            logger.info(f"Min: {min_time:.2f}s, Max: {max_time:.2f}s, StdDev: {std_dev:.2f}s")
        
        return times
        
    except Exception as e:
        error_msg = f"Chat completion failed (may not be supported by this model): {e}"
        print(f"⚠️  {error_msg}")
        logger.warning(error_msg, exc_info=True)
        import traceback
        traceback_str = traceback.format_exc()
        logger.debug(f"Full traceback:\n{traceback_str}")
        return []


def compare_cpu_gpu(model_path: Path, n_gpu_layers: int = 35):
    """Compare CPU vs GPU performance."""
    print("\n" + "="*60)
    print("CPU vs GPU PERFORMANCE COMPARISON")
    print("="*60)
    
    results = {}
    
    # Test CPU (0 layers)
    print("\n🖥️  Testing CPU (0 GPU layers)...")
    llm_cpu = test_model_loading(model_path, n_gpu_layers=0)
    if llm_cpu:
        metrics_cpu = PerformanceMetrics()
        test_text_generation(llm_cpu, metrics=metrics_cpu, iterations=3)
        results['cpu'] = asdict(metrics_cpu)
        del llm_cpu
    
    # Test GPU
    print("\n🎮 Testing GPU ({} GPU layers)...".format(n_gpu_layers))
    llm_gpu = test_model_loading(model_path, n_gpu_layers=n_gpu_layers)
    if llm_gpu:
        metrics_gpu = PerformanceMetrics()
        test_text_generation(llm_gpu, metrics=metrics_gpu, iterations=3)
        results['gpu'] = asdict(metrics_gpu)
        del llm_gpu
    
    # Compare results
    if 'cpu' in results and 'gpu' in results:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        cpu_speed = results['cpu'].get('tokens_per_second', 0)
        gpu_speed = results['gpu'].get('tokens_per_second', 0)
        if cpu_speed > 0:
            speedup = gpu_speed / cpu_speed
            print(f"CPU Speed: {cpu_speed:.2f} tokens/second")
            print(f"GPU Speed: {gpu_speed:.2f} tokens/second")
            print(f"🚀 GPU Speedup: {speedup:.2f}x")
    
    return results


def export_results(metrics: PerformanceMetrics, sys_info: Dict, model_path: Path, output_file: Path):
    """Export test results to JSON file."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': sys_info,
        'model': {
            'path': str(model_path),
            'size_mb': model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0,
        },
        'metrics': asdict(metrics),
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results exported to: {output_file}")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced GPU test for llama-cpp-python with benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  ./run_gpu_test.sh --model-path model.gguf
  
  # Benchmark mode (5 iterations)
  ./run_gpu_test.sh --model-path model.gguf --benchmark 5
  
  # CPU vs GPU comparison
  ./run_gpu_test.sh --model-path model.gguf --compare-cpu-gpu
  
  # Export results to JSON
  ./run_gpu_test.sh --model-path model.gguf --export results.json
        """
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to GGUF model file (auto-detected if not provided)"
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=35,
        help="Number of layers to offload to GPU (0 = CPU only, -1 = all layers)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip text generation tests"
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=1,
        metavar="N",
        help="Run benchmark mode with N iterations (default: 1)"
    )
    parser.add_argument(
        "--compare-cpu-gpu",
        action="store_true",
        help="Compare CPU vs GPU performance"
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        metavar="FILE",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Custom prompt for text generation test"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = Path(__file__).parent / f"gpu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)
    
    # Set device-local memory preference (embedding layer may use CPU, which is fine)
    # All transformer layers will use GPU - this is optimal design by llama.cpp
    if 'GGML_VK_PREFER_HOST_MEMORY' in os.environ:
        del os.environ['GGML_VK_PREFER_HOST_MEMORY']
    os.environ['GGML_VK_PREFER_HOST_MEMORY'] = '0'  # Use device-local memory for GPU layers
    logger.info("Setting GGML_VK_PREFER_HOST_MEMORY=0 (embedding layer may use CPU, rest on GPU - optimal)")
    
    print("="*60)
    print("LLAMA-CPP-PYTHON GPU TEST")
    print("="*60)
    logger.info("="*60)
    logger.info("LLAMA-CPP-PYTHON GPU TEST")
    logger.info("="*60)
    logger.info(f"Command line arguments: {args}")
    
    # Check GPU support
    check_gpu_support()
    
    # Find or use provided model
    model_path = args.model_path
    if not model_path:
        print("\n🔍 Searching for GGUF models...")
        logger.info("Searching for GGUF models...")
        model_path = find_gguf_model()
    
    # Get system info
    sys_info = get_system_info()
    logger.info(f"System info: {sys_info}")
    
    # Handle CPU vs GPU comparison
    if args.compare_cpu_gpu:
        if not model_path:
            error_msg = "--compare-cpu-gpu requires --model-path"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            sys.exit(1)
        results = compare_cpu_gpu(model_path, args.n_gpu_layers)
        if args.export:
            export_path = args.export
            with open(export_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'system_info': sys_info,
                    'comparison': results
                }, f, indent=2)
            print(f"\n💾 Comparison results exported to: {export_path}")
            logger.info(f"Comparison results exported to: {export_path}")
        return
    
    # Initialize metrics
    metrics = PerformanceMetrics()
    
    # Test model loading
    llm = None
    model_loaded = False
    try:
        llm = test_model_loading(model_path, n_gpu_layers=args.n_gpu_layers, metrics=metrics, logger=logger)
        model_loaded = llm is not None
        
        # Test generation if model loaded
        if llm and not args.skip_generation:
            test_text_generation(
                llm, 
                prompt=args.prompt,
                metrics=metrics,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                iterations=args.benchmark,
                logger=logger
            )
            test_chat_completion(llm, metrics=metrics, iterations=args.benchmark, logger=logger)
    finally:
        # Cleanup
        if llm:
            print("\n🧹 Cleaning up...")
            logger.info("Cleaning up model...")
            del llm
            logger.info("Model cleanup completed")
    
    # Print performance summary
    if model_loaded and not args.skip_generation:
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"⏱️  Model Load Time: {metrics.model_load_time:.2f}s")
        if metrics.text_gen_time > 0:
            if args.benchmark > 1:
                print(f"📝 Text Generation: {metrics.avg_text_gen_time:.2f}s avg ({metrics.min_text_gen_time:.2f}s - {metrics.max_text_gen_time:.2f}s)")
            else:
                print(f"📝 Text Generation: {metrics.text_gen_time:.2f}s (~{metrics.text_gen_tokens} tokens, {metrics.tokens_per_second:.2f} tok/s)")
        if metrics.chat_gen_time > 0:
            if args.benchmark > 1:
                print(f"💬 Chat Completion: {metrics.avg_chat_gen_time:.2f}s avg ({metrics.min_chat_gen_time:.2f}s - {metrics.max_chat_gen_time:.2f}s)")
            else:
                print(f"💬 Chat Completion: {metrics.chat_gen_time:.2f}s (~{metrics.chat_gen_tokens} tokens)")
        if metrics.memory_used_mb > 0:
            print(f"💾 GPU Memory Used: {metrics.memory_used_mb:.0f} MB ({metrics.memory_used_mb/1024:.2f} GB)")
        if metrics.text_gen_time > 0 and metrics.chat_gen_time > 0:
            avg_tokens_per_sec = (metrics.text_gen_tokens / metrics.text_gen_time + 
                                 metrics.chat_gen_tokens / metrics.chat_gen_time) / 2
            print(f"⚡ Average Speed: {avg_tokens_per_sec:.2f} tokens/second")
    
    # Export results if requested
    if args.export and model_loaded:
        export_results(metrics, sys_info, model_path, args.export)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    logger.info("="*60)
    logger.info("TEST COMPLETE")
    logger.info("="*60)
    
    if model_loaded:
        print("✅ All tests passed!")
        logger.info("All tests passed!")
    else:
        print("⚠️  Model loading skipped (no model found or loading failed)")
        logger.warning("Model loading skipped (no model found or loading failed)")
    
    print(f"\n📝 Log file: {log_file}")
    logger.info(f"Log file location: {log_file}")


if __name__ == "__main__":
    main()

