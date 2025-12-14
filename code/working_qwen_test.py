#!/usr/bin/env python3
"""
Working multimodal test using correct Qwen approach
"""

import torch
from PIL import Image
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qwen_multimodal():
    """Test Qwen model with multimodal input using correct approach"""
    logger.info("=== Testing Qwen Multimodal ===")
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        qwen_path = "/home/himanshu/dev/models/QWEN_AWQ"
        
        # Load model - Use the correct class
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        
        # Prepare messages in Qwen's expected format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please analyze this chemical compound image and explain what you see."}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.info(f"✅ Chat template applied: {text[:100]}...")
        
        # Process inputs
        inputs = processor(text=[text], images=[image], return_tensors="pt")
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

def test_qwen_with_second_image():
    """Test Qwen with the second image"""
    logger.info("=== Testing Qwen with Second Image ===")
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        qwen_path = "/home/himanshu/dev/models/QWEN_AWQ"
        
        # Load model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
        
        logger.info("✅ Qwen model loaded successfully")
        
        # Load second test image
        image_path = "/home/himanshu/dev/input_img/img2.png"
        image = Image.open(image_path)
        logger.info(f"✅ Second image loaded: {image.size}")
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please analyze this chemical compound image and explain what you see."}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(text=[text], images=[image], return_tensors="pt")
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
        logger.info(f"✅ Qwen Response for img2: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Qwen FAILED for img2: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting working multimodal tests...")
    
    # Test Qwen with first image
    qwen_success = test_qwen_multimodal()
    
    # Test Qwen with second image
    qwen_success_2 = test_qwen_with_second_image()
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"Qwen img1: {'✅ SUCCESS' if qwen_success else '❌ FAILED'}")
    logger.info(f"Qwen img2: {'✅ SUCCESS' if qwen_success_2 else '❌ FAILED'}")

