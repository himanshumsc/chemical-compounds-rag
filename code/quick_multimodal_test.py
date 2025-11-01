#!/usr/bin/env python3
"""
Quick multimodal test for individual models
"""

import torch
from PIL import Image
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phi4_multimodal():
    """Test Phi-4 model with multimodal input"""
    logger.info("=== Testing Phi-4 Multimodal ===")
    
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
        
        phi4_path = "/home/himanshu/dev/models/PHI4_ONNX"
        
        # Load config and override attention
        cfg = AutoConfig.from_pretrained(phi4_path, trust_remote_code=True)
        cfg._attn_implementation = "sdpa"
        
        # For Phi-4 ONNX, we need to use the custom model class
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            phi4_path,
            config=cfg,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(phi4_path, trust_remote_code=True)
        
        logger.info("✅ Phi-4 model loaded successfully")
        
        # Load test image
        image_path = "/home/himanshu/dev/input_img/img1.png"
        image = Image.open(image_path)
        logger.info(f"✅ Image loaded: {image.size}")
        
        # Prepare multimodal input
        prompt = "Please analyze this chemical compound image and explain what you see."
        
        # Process inputs
        inputs = processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        logger.info("✅ Inputs processed")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Phi-4 Response: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Phi-4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen_multimodal():
    """Test Qwen model with multimodal input"""
    logger.info("=== Testing Qwen Multimodal ===")
    
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        qwen_path = "/home/himanshu/dev/models/QWEN_AWQ"
        
        # Load model - Qwen2.5-VL is a vision-language model
        model = AutoModelForVision2Seq.from_pretrained(
            qwen_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
        
        logger.info("✅ Qwen model loaded successfully")
        
        # Load test image
        image_path = "/home/himanshu/dev/input_img/img1.png"
        image = Image.open(image_path)
        logger.info(f"✅ Image loaded: {image.size}")
        
        # Prepare multimodal input
        prompt = "Please analyze this chemical compound image and explain what you see."
        
        # Process inputs
        inputs = processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        logger.info("✅ Inputs processed")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Qwen Response: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Qwen FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting quick multimodal tests...")
    
    # Test Phi-4
    phi4_success = test_phi4_multimodal()
    
    # Test Qwen
    qwen_success = test_qwen_multimodal()
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"Phi-4: {'✅ SUCCESS' if phi4_success else '❌ FAILED'}")
    logger.info(f"Qwen: {'✅ SUCCESS' if qwen_success else '❌ FAILED'}")
