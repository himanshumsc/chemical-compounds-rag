#!/usr/bin/env python3
"""
Simple test to isolate the parallel execution issue
"""

import logging
import sys
import time
from pathlib import Path

# Add the code directory to path
sys.path.append(str(Path(__file__).parent))

from model_manager import ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test just model loading without generation"""
    logger.info("=== Testing Model Loading Only ===")
    
    # Initialize model manager
    phi4_path = Path("/home/himanshu/dev/models/PHI4_ONNX")
    qwen_path = Path("/home/himanshu/dev/models/QWEN_AWQ")
    
    model_manager = ModelManager(phi4_path, qwen_path, device="auto")
    
    # Load models
    logger.info("Loading models...")
    start_time = time.time()
    status = model_manager.load_all_models()
    load_time = time.time() - start_time
    
    logger.info(f"Model loading status: {status}")
    logger.info(f"Model loading time: {load_time:.2f}s")
    
    # Check individual model status
    logger.info(f"Phi-4 loaded: {model_manager.is_model_loaded('phi4')}")
    logger.info(f"Qwen loaded: {model_manager.is_model_loaded('qwen')}")
    
    return status['both_loaded']

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        logger.info("✅ Model loading test completed successfully!")
    else:
        logger.error("❌ Model loading test failed!")
