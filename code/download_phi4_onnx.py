#!/usr/bin/env python3
"""
Download Microsoft Phi-4 Multimodal Instruct ONNX Model
Quantized version for research comparison with Qwen AWQ
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_phi4_onnx():
    """Download Phi-4 Multimodal Instruct ONNX model"""
    
    # Model configuration
    model_id = "microsoft/Phi-4-multimodal-instruct-onnx"
    local_dir = Path("/home/himanshu/dev/models/PHI4_ONNX")
    
    logger.info(f"Downloading Phi-4 ONNX model: {model_id}")
    logger.info(f"Local directory: {local_dir}")
    
    try:
        # Create directory if it doesn't exist
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the model
        logger.info("Starting download...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True,  # Resume if interrupted
            token=None  # Use public access
        )
        
        logger.info("✅ Phi-4 ONNX model downloaded successfully!")
        
        # List downloaded files
        logger.info("Downloaded files:")
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_path.relative_to(local_dir)} ({size_mb:.1f} MB)")
        
        # Check total size
        total_size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
        total_size_gb = total_size / (1024 * 1024 * 1024)
        logger.info(f"Total model size: {total_size_gb:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download Phi-4 ONNX model: {e}")
        return False

def main():
    """Main function"""
    logger.info("=== Phi-4 ONNX Model Downloader ===")
    
    # Check if we're in the right environment
    if not os.path.exists("/home/himanshu/dev/code/.venv_phi4_req"):
        logger.warning("Virtual environment not found. Make sure to activate .venv_phi4_req")
    
    # Download the model
    success = download_phi4_onnx()
    
    if success:
        logger.info("🎉 Download completed successfully!")
        logger.info("You can now use the Phi-4 ONNX model for research comparison")
    else:
        logger.error("💥 Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
