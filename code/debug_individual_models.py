#!/usr/bin/env python3
"""
Debug script to test individual model generation outside parallel context
"""

import logging
import sys
from pathlib import Path

# Add the code directory to path
sys.path.append(str(Path(__file__).parent))

from model_manager import ModelManager
from parallel_generator import ParallelGenerator

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_individual_models():
    """Test each model individually"""
    logger.info("=== Testing Individual Model Generation ===")
    
    # Initialize model manager
    phi4_path = Path("/home/himanshu/dev/models/PHI4_ONNX")
    qwen_path = Path("/home/himanshu/dev/models/QWEN_AWQ")
    
    logger.info(f"Phi-4 path: {phi4_path}")
    logger.info(f"Qwen path: {qwen_path}")
    
    model_manager = ModelManager(phi4_path, qwen_path, device="auto")
    
    # Load models
    logger.info("Loading models...")
    status = model_manager.load_all_models()
    logger.info(f"Model loading status: {status}")
    
    if not status['both_loaded']:
        logger.error("Failed to load models")
        return False
    
    # Initialize parallel generator
    parallel_generator = ParallelGenerator(model_manager)
    
    # Test query with multimodal input
    test_prompt = "Analyze this image."
    test_images = [str(Path("/home/himanshu/dev/input_img/img1.png"))]
    
    logger.info(f"Testing with prompt: '{test_prompt}'")
    
    # Test Phi-4 individually
    logger.info("=== Testing Phi-4 ONNX ===")
    try:
        phi4_result = parallel_generator.generate_phi4_response(test_prompt, test_images)
        logger.info(f"Phi-4 result: {phi4_result}")
        
        if phi4_result['success']:
            logger.info(f"✅ Phi-4 SUCCESS: {phi4_result['response'][:100]}...")
        else:
            logger.error(f"❌ Phi-4 FAILED: {phi4_result['error']}")
    except Exception as e:
        logger.error(f"❌ Phi-4 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Qwen individually
    logger.info("=== Testing Qwen AWQ ===")
    try:
        qwen_result = parallel_generator.generate_qwen_response(test_prompt, test_images)
        logger.info(f"Qwen result: {qwen_result}")
        
        if qwen_result['success']:
            logger.info(f"✅ Qwen SUCCESS: {qwen_result['response'][:100]}...")
        else:
            logger.error(f"❌ Qwen FAILED: {qwen_result['error']}")
    except Exception as e:
        logger.error(f"❌ Qwen EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with second image (Phi-4 only for speed)
    logger.info("=== Testing Phi-4 with Second Image ===")
    test_images_2 = [str(Path("/home/himanshu/dev/input_img/img2.png"))]
    
    try:
        phi4_result_2 = parallel_generator.generate_phi4_response(test_prompt, test_images_2)
        logger.info(f"Phi-4 result for img2: {phi4_result_2}")
        
        if phi4_result_2['success']:
            logger.info(f"✅ Phi-4 SUCCESS for img2: {phi4_result_2['response'][:100]}...")
        else:
            logger.error(f"❌ Phi-4 FAILED for img2: {phi4_result_2['error']}")
    except Exception as e:
        logger.error(f"❌ Phi-4 EXCEPTION for img2: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    parallel_generator.shutdown()
    
    return True

if __name__ == "__main__":
    test_individual_models()
