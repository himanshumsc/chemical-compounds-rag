#!/bin/bash
# Run EDA scripts with llama-env activated

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_ENV="/home/himanshu/llama-env"

echo "=========================================="
echo "EDA Scripts with llama-env"
echo "=========================================="
echo ""

# Check if llama-env exists
if [ ! -d "${LLAMA_ENV}" ]; then
    echo "❌ Error: llama-env not found at ${LLAMA_ENV}"
    echo ""
    echo "Please create it or use a different environment:"
    echo "  python3 -m venv ${LLAMA_ENV}"
    exit 1
fi

# Activate llama-env
echo "🔧 Activating llama-env..."
source "${LLAMA_ENV}/bin/activate"

echo "✅ llama-env activated"
echo "🐍 Python: $(which python)"
echo "🐍 Python version: $(python --version)"
echo ""

# Check and install dependencies if needed
echo "📦 Checking dependencies..."
MISSING_DEPS=()

python -c "import matplotlib" 2>/dev/null || MISSING_DEPS+=("matplotlib")
python -c "import seaborn" 2>/dev/null || MISSING_DEPS+=("seaborn")
python -c "import sklearn" 2>/dev/null || MISSING_DEPS+=("scikit-learn")
python -c "import pandas" 2>/dev/null || MISSING_DEPS+=("pandas")
python -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "⚠️  Missing dependencies: ${MISSING_DEPS[*]}"
    echo ""
    read -p "Install missing dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Installing dependencies..."
        pip install "${MISSING_DEPS[@]}"
        echo "✅ Dependencies installed"
    else
        echo "⚠️  Continuing without full dependencies..."
    fi
else
    echo "✅ All dependencies available"
fi

echo ""
echo "🚀 Running EDA scripts..."
echo ""

# Run the requested script or all scripts
if [ $# -eq 0 ]; then
    # Run all scripts
    python "${SCRIPT_DIR}/scripts/run_all_eda.py"
else
    # Run specific script
    SCRIPT_PATH="${SCRIPT_DIR}/scripts/$1"
    if [ -f "${SCRIPT_PATH}" ]; then
        python "${SCRIPT_PATH}"
    else
        echo "❌ Script not found: ${SCRIPT_PATH}"
        exit 1
    fi
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ EDA completed successfully"
else
    echo "❌ EDA failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE

