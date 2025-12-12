#!/bin/bash
# Rebuild llama-cpp-python with Vulkan GPU support

set -e

echo "=========================================="
echo "Rebuild llama-cpp-python with GPU Support"
echo "=========================================="
echo ""

# Check for glslc (required for Vulkan)
if ! which glslc > /dev/null 2>&1; then
    echo "❌ glslc (GLSL compiler) not found!"
    echo "   This is required for Vulkan GPU support"
    echo ""
    echo "📦 Installing glslc (from universe repository)..."
    sudo apt-get update
    sudo apt-get install -y glslc
    
    if ! which glslc > /dev/null 2>&1; then
        echo "❌ Failed to install glslc. Please install manually:"
        echo "   sudo apt-get update"
        echo "   sudo apt-get install -y glslc"
        exit 1
    fi
    echo "✅ glslc installed: $(which glslc)"
    echo ""
fi

echo "⚠️  This will rebuild llama-cpp-python with Vulkan support"
echo "   This may take 10-30 minutes"
echo ""

# Activate llama-env
source /home/himanshu/llama-env/bin/activate
echo "✅ Using environment: $(which python)"
echo ""

if pip show llama-cpp-python > /dev/null 2>&1; then
    echo "Current installation:"
    pip show llama-cpp-python | grep -E "Name|Version"
    echo ""
fi

read -p "Continue with rebuild? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 1
fi

echo ""
echo "📦 Uninstalling current llama-cpp-python..."
pip uninstall -y llama-cpp-python 2>&1 | grep -v "WARNING: Skipping" || true

echo ""
echo "🧹 Clearing pip cache and wheels..."
pip cache purge 2>&1 | tail -1
rm -rf ~/.cache/pip/wheels/*llama* /tmp/pip-*wheel* 2>/dev/null || true
echo "✅ Cache cleared"

echo ""
echo "🔨 Building llama-cpp-python with Vulkan GPU support..."
echo "   Using flag: GGML_VULKAN=on"
echo "   Forcing source build (no pre-built wheels)"
echo "   This will take 10-30 minutes..."
echo ""

# Rebuild with Vulkan support - force source build
FORCE_CMAKE=1 CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir

echo ""
echo "✅ Build complete!"
echo ""
echo "🔍 Verifying GPU support..."
python3 << 'EOF'
import llama_cpp
has_gpu = hasattr(llama_cpp.llama_cpp, 'llama_gpu_device_count')
if has_gpu:
    print("✅ GPU support is now available!")
    try:
        count = llama_cpp.llama_cpp.llama_gpu_device_count()
        print(f"   GPU devices: {count}")
    except:
        print("   Could not get device count, but GPU support is present")
else:
    print("❌ GPU support still not available")
    print("   Build may have failed or Vulkan not found")
EOF

echo ""
echo "🚀 You can now test with GPU:"
echo "   ./run_gpu_test.sh --model-path <model.gguf> --n-gpu-layers -1"

