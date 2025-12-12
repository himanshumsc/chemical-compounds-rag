#!/bin/bash
# Wrapper script to run GPU test with llama-env activated

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_ENV="/home/himanshu/llama-env"
TEST_SCRIPT="${SCRIPT_DIR}/test_gpu_llamacpp.py"

echo "=========================================="
echo "LLAMA-CPP-PYTHON GPU TEST"
echo "=========================================="
echo "Using llama-env: ${LLAMA_ENV}"
echo "Test script: ${TEST_SCRIPT}"
echo ""

# Check if llama-env exists
if [ ! -d "${LLAMA_ENV}" ]; then
    echo "❌ Error: llama-env not found at ${LLAMA_ENV}"
    exit 1
fi

# Check if activate script exists
if [ ! -f "${LLAMA_ENV}/bin/activate" ]; then
    echo "❌ Error: activate script not found in llama-env"
    exit 1
fi

# Activate llama-env and run the test
echo "🔧 Activating llama-env..."
source "${LLAMA_ENV}/bin/activate"

echo "✅ llama-env activated"
echo "🐍 Python: $(which python)"
echo "🐍 Python version: $(python --version)"
echo ""

# Check if llama-cpp-python is installed
echo "📦 Checking llama-cpp-python installation..."
if python -c "from llama_cpp import Llama; print('✅ llama-cpp-python is installed')" 2>/dev/null; then
    echo ""
else
    echo "❌ llama-cpp-python not found in llama-env"
    echo "   Install with: pip install llama-cpp-python"
    exit 1
fi

echo ""
echo "🚀 Running GPU test..."
echo ""

# Run the test script with all passed arguments
python "${TEST_SCRIPT}" "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully"
else
    echo "❌ Test failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE

