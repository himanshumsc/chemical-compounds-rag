# GPU llama-cpp-python Test

Simple test script to verify GPU support and functionality with llama-cpp-python.

## Prerequisites

- `llama-env` virtual environment at `/home/himanshu/llama-env`
- `llama-cpp-python` installed in llama-env (with Vulkan support)
- Optional: GGUF model file for full testing

## Usage

### Option 1: Using the wrapper script (Recommended)

The wrapper script automatically activates `llama-env`:

```bash
cd /home/himanshu/MSC_FINAL/dev/GPU_llamaCPP_POC
./run_gpu_test.sh [options]
```

### Option 2: Manual activation

```bash
# Activate llama-env
source /home/himanshu/llama-env/bin/activate

# Run the test
python test_gpu_llamacpp.py [options]
```

## Command Line Options

```bash
./run_gpu_test.sh --help

Main Options:
  --model-path PATH     Path to GGUF model file (auto-detected if not provided)
  --n-gpu-layers N      Number of layers to offload to GPU (default: 35, 0 = CPU only, -1 = all)
  --skip-generation     Skip text generation tests

Benchmarking:
  --benchmark N         Run benchmark mode with N iterations (default: 1)
  --compare-cpu-gpu     Compare CPU vs GPU performance side-by-side

Generation Parameters:
  --prompt PROMPT       Custom prompt for text generation test
  --max-tokens N        Maximum tokens to generate (default: 50)
  --temperature F       Temperature for generation (default: 0.7)

Export:
  --export FILE         Export results to JSON file
```

## Examples

### Basic test (checks installation, no model required)
```bash
./run_gpu_test.sh
```

### Test with specific model
```bash
./run_gpu_test.sh --model-path /path/to/model.gguf
```

### Test with all layers on GPU
```bash
./run_gpu_test.sh --model-path /path/to/model.gguf --n-gpu-layers -1
```

### CPU-only test
```bash
./run_gpu_test.sh --model-path /path/to/model.gguf --n-gpu-layers 0
```

### Benchmark mode (multiple iterations)
```bash
./run_gpu_test.sh --model-path /path/to/model.gguf --benchmark 5
```
This runs each test 5 times and shows average, min, max, and standard deviation.

### CPU vs GPU comparison
```bash
./run_gpu_test.sh --model-path /path/to/model.gguf --compare-cpu-gpu
```
Compares performance between CPU (0 layers) and GPU (specified layers) with speedup calculation.

### Custom generation parameters
```bash
./run_gpu_test.sh --model-path /path/to/model.gguf \
  --prompt "Explain quantum computing" \
  --max-tokens 100 \
  --temperature 0.8
```

### Export results to JSON
```bash
./run_gpu_test.sh --model-path /path/to/model.gguf --export results.json
```
Exports all metrics, system info, and model details to a JSON file for analysis.

## What the Test Does

1. **System & GPU Info**: Detects Python version, GPU name, memory, and driver version
2. **GPU Support Check**: Verifies llama-cpp-python installation and GPU capabilities
3. **Model Loading**: Loads GGUF model with performance metrics (load time, speed, memory usage)
4. **Text Generation**: Tests text completion with customizable prompts and parameters
5. **Chat Completion**: Tests chat API (if supported by model)
6. **Benchmarking**: Optional multi-iteration testing with statistical analysis
7. **CPU vs GPU Comparison**: Side-by-side performance comparison
8. **Result Export**: JSON export of all metrics for further analysis

## Enhanced Features

### Performance Metrics
- Model load time and load speed (GB/s)
- Token generation speed (tokens/second)
- GPU memory usage tracking
- Statistical analysis (min, max, avg, std dev) in benchmark mode

### Benchmark Mode
When using `--benchmark N`, the test runs each generation N times and provides:
- Average, minimum, and maximum generation times
- Standard deviation
- More reliable performance measurements

### CPU vs GPU Comparison
The `--compare-cpu-gpu` option:
- Tests the same model on CPU (0 layers) and GPU
- Calculates GPU speedup factor
- Shows side-by-side performance metrics
- Exports comparison data to JSON if `--export` is used

## Model Auto-Detection

The script automatically searches for GGUF models in:
- `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_GGUF`
- `/home/himanshu/dev/models`
- `/home/himanshu/MSC_FINAL/dev/models/GEMMA3_QAT_Q4_0_GGUF`
- `/home/himanshu/MSC_FINAL/dev/models`

## Downloading a Test Model

If you don't have a GGUF model yet, you can download a small test model:

```bash
./download_test_model.sh
```

This downloads Llama-2-7B-Chat Q4_0 (~4GB) to `/home/himanshu/MSC_FINAL/dev/models/LLAMA2_7B_CHAT_GGUF/`

Then test it:
```bash
./run_gpu_test.sh --model-path /home/himanshu/MSC_FINAL/dev/models/LLAMA2_7B_CHAT_GGUF/llama-2-7b-chat.Q4_0.gguf --n-gpu-layers 35
```

## Test Results

✅ **Installation Verified**: The test confirmed:
- llama-env is properly configured
- llama-cpp-python is installed with Vulkan support
- Python 3.12.3 is active

⚠️ **No Model Found**: To test GPU functionality with actual inference, download a GGUF model using the helper script above.

## Notes

- If no model is found, the script will still run and verify llama-cpp-python installation
- GPU offloading requires `n_gpu_layers > 0`
- Vulkan support is detected automatically by llama-cpp-python
- The script gracefully handles missing models and provides helpful error messages
- Test completed successfully even without a model - installation is verified!

