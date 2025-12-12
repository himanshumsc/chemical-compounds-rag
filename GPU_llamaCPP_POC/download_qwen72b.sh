#!/bin/bash
# Wrapper script to download Qwen2.5-72B-Instruct-Q6_K_M.gguf
# Tries to find the correct Python environment with huggingface_hub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_qwen72b_gguf.py"

echo "=========================================="
echo "Download Qwen2.5-72B-Instruct-Q6_K_M.gguf"
echo "=========================================="
echo ""

# Try different Python environments (prioritize .venv_phi4_req as it likely has huggingface_hub)
PYTHON_ENVS=(
    
    "/home/himanshu/llama-env/bin/python"
    "python3"
    "python"
)

PYTHON_CMD=""
ENV_NAME=""

for py_cmd in "${PYTHON_ENVS[@]}"; do
    if [ -f "$py_cmd" ] || command -v "$py_cmd" &> /dev/null; then
        echo "🔍 Checking $py_cmd..."
        if "$py_cmd" -c "import huggingface_hub" 2>/dev/null; then
            PYTHON_CMD="$py_cmd"
            if [[ "$py_cmd" == *"llama-env"* ]]; then
                ENV_NAME="llama-env"
           
            else
                ENV_NAME="system"
            fi
            echo "✅ Found huggingface_hub in $ENV_NAME"
            break
        else
            echo "   ❌ huggingface_hub not available"
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "❌ Error: huggingface_hub not found in any environment"
    echo ""
    echo "Please install it in one of these environments:"
    echo "  1. llama-env:"
    echo "     source /home/himanshu/llama-env/bin/activate"
    echo "     pip install huggingface_hub"
    echo ""
    echo "  2. .venv_phi4_req:"
    echo "     source /home/himanshu/dev/code/.venv_phi4_req/bin/activate"
    echo "     pip install huggingface_hub"
    echo ""
    echo "  3. System Python:"
    echo "     pip3 install huggingface_hub"
    exit 1
fi

echo ""
echo "🚀 Using Python: $PYTHON_CMD ($ENV_NAME)"
echo ""

# Activate the environment if it's a venv
if [[ "$PYTHON_CMD" == *"llama-env"* ]]; then
    source /home/himanshu/llama-env/bin/activate
elif [[ "$PYTHON_CMD" == *".venv_phi4_req"* ]]; then
    source /home/himanshu/dev/code/.venv_phi4_req/bin/activate
fi

# Run the download script
"$PYTHON_CMD" "${DOWNLOAD_SCRIPT}" "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Download completed successfully"
else
    echo ""
    echo "❌ Download failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE

