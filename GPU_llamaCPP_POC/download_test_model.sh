#!/bin/bash
# Helper script to download a small test model for GPU testing
# Downloads llama-2-7b-chat Q4_0 quantized model (~4GB)

set -e

MODEL_DIR="/home/himanshu/MSC_FINAL/dev/models"
TEST_MODEL_DIR="${MODEL_DIR}/LLAMA2_7B_CHAT_GGUF"
MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"
MODEL_FILE="${TEST_MODEL_DIR}/llama-2-7b-chat.Q4_0.gguf"

echo "=========================================="
echo "Download Test Model for GPU Testing"
echo "=========================================="
echo "Model: Llama-2-7B-Chat Q4_0 (~4GB)"
echo "Destination: ${TEST_MODEL_DIR}"
echo ""

# Check if model already exists
if [ -f "${MODEL_FILE}" ]; then
    echo "✅ Model already exists: ${MODEL_FILE}"
    echo "   Size: $(du -h "${MODEL_FILE}" | cut -f1)"
    echo ""
    echo "You can test it with:"
    echo "  ./run_gpu_test.sh --model-path ${MODEL_FILE} --n-gpu-layers 35"
    exit 0
fi

# Create directory
mkdir -p "${TEST_MODEL_DIR}"

# Check disk space (need at least 5GB free)
AVAILABLE=$(df "${MODEL_DIR}" | tail -1 | awk '{print $4}')
if [ "${AVAILABLE}" -lt 5242880 ]; then  # 5GB in KB
    echo "⚠️  Warning: Less than 5GB free space available"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📥 Downloading model..."
echo "   This may take a while (~4GB download)..."
echo ""

# Download using wget or curl
if command -v wget &> /dev/null; then
    wget -O "${MODEL_FILE}" "${MODEL_URL}" --progress=bar:force
elif command -v curl &> /dev/null; then
    curl -L -o "${MODEL_FILE}" "${MODEL_URL}" --progress-bar
else
    echo "❌ Error: Neither wget nor curl found. Please install one."
    exit 1
fi

if [ -f "${MODEL_FILE}" ]; then
    echo ""
    echo "✅ Model downloaded successfully!"
    echo "   Location: ${MODEL_FILE}"
    echo "   Size: $(du -h "${MODEL_FILE}" | cut -f1)"
    echo ""
    echo "🚀 Test it with:"
    echo "  ./run_gpu_test.sh --model-path ${MODEL_FILE} --n-gpu-layers 35"
else
    echo "❌ Download failed!"
    exit 1
fi

