#!/usr/bin/env python3
"""
Download Qwen2.5-VL-7B-Instruct-AWQ quantized model from Hugging Face
Saves to dev/models/QWEN_AWQ folder
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import argparse

def download_qwen2_5_vl_awq_model():
    """Download Qwen2.5-VL-7B-Instruct-AWQ quantized model"""
    
    # Model configuration - AWQ quantized version
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    local_dir = Path("/home/himanshu/dev/models/QWEN_AWQ")
    
    print(f"🚀 Starting download of {model_name}")
    print(f"📁 Saving to: {local_dir}")
    print("⚡ This is the AWQ quantized version (4-bit precision)")
    
    # Create directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the model
        print("📥 Downloading quantized model files...")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True,  # Resume if interrupted
            max_workers=4,  # Parallel downloads
        )
        
        print("✅ Quantized model downloaded successfully!")
        print(f"📂 Model saved to: {local_dir}")
        
        # Check disk usage
        import shutil
        total_size = shutil.disk_usage(local_dir.parent).used
        print(f"💾 Total disk usage in models folder: {total_size / (1024**3):.2f} GB")
        
        # List downloaded files
        print("\n📋 Downloaded files:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(str(local_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024**2)  # MB
                print(f"{subindent}{file} ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading quantized model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Qwen2.5-VL-7B-Instruct-AWQ quantized model")
    parser.add_argument("--check-disk", action="store_true", 
                       help="Check available disk space before download")
    
    args = parser.parse_args()
    
    if args.check_disk:
        # Check available disk space
        import shutil
        models_dir = Path("/home/himanshu/dev/models")
        total, used, free = shutil.disk_usage(models_dir)
        
        print(f"💾 Disk space check:")
        print(f"   Total: {total / (1024**3):.2f} GB")
        print(f"   Used:  {used / (1024**3):.2f} GB") 
        print(f"   Free:  {free / (1024**3):.2f} GB")
        
        # AWQ quantized model is typically ~2.6GB (much smaller than full precision)
        if free < 5 * (1024**3):  # Less than 5GB free
            print("⚠️  Warning: Low disk space. AWQ model requires ~2.6GB")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("❌ Download cancelled")
                return
    
    success = download_qwen2_5_vl_awq_model()
    
    if success:
        print("\n🎉 AWQ quantized model download completed successfully!")
        print("💡 Benefits of AWQ quantization:")
        print("   - ~6x smaller model size (2.6GB vs 15.6GB)")
        print("   - Faster inference")
        print("   - Lower memory requirements")
        print("   - Minimal accuracy loss")
        print("\n🚀 You can now use this model with:")
        print("   - transformers library")
        print("   - Your multimodal RAG system")
        print("   - Any compatible inference framework")
    else:
        print("\n❌ Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
