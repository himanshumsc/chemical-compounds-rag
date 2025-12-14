#!/usr/bin/env python3
"""
Simple model loading test to diagnose issues
"""

import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test basic model loading without generation"""
    logger.info("=== Testing Model Loading ===")
    
    # Test Phi-4
    try:
        logger.info("Testing Phi-4 model loading...")
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
        
        phi4_path = "/home/himanshu/dev/models/PHI4_ONNX"
        
        # Load config
        cfg = AutoConfig.from_pretrained(phi4_path, trust_remote_code=True)
        cfg._attn_implementation = "sdpa"
        logger.info("Phi-4 config loaded")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            phi4_path,
            config=cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        logger.info("Phi-4 model loaded successfully")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            phi4_path,
            trust_remote_code=True,
            use_fast=False
        )
        logger.info("Phi-4 processor loaded successfully")
        
        phi4_success = True
        
    except Exception as e:
        logger.error(f"Phi-4 loading failed: {e}")
        phi4_success = False
    
    # Test Qwen
    try:
        logger.info("Testing Qwen model loading...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        qwen_path = "/home/himanshu/dev/models/QWEN_AWQ"
        
        # Load model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Qwen model loaded successfully")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            qwen_path,
            trust_remote_code=True
        )
        logger.info("Qwen processor loaded successfully")
        
        qwen_success = True
        
    except Exception as e:
        logger.error(f"Qwen loading failed: {e}")
        qwen_success = False
    
    # Summary
    logger.info("=== Loading Test Summary ===")
    logger.info(f"Phi-4 Model: {'PASS' if phi4_success else 'FAIL'}")
    logger.info(f"Qwen Model: {'PASS' if qwen_success else 'FAIL'}")
    
    return phi4_success, qwen_success

if __name__ == "__main__":
    test_model_loading()



