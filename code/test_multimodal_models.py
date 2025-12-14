#!/usr/bin/env python3
"""
Simple test script to verify both Phi-4 and Qwen models work with multimodal input
"""

import torch
from PIL import Image
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phi4_model():
    """Test Phi-4 model with multimodal input"""
    logger.info("=== Testing Phi-4 Model ===")
    
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        phi4_path = Path("/home/himanshu/dev/models/PHI4_ONNX")
        if not phi4_path.exists():
            logger.error(f"Phi-4 model not found at {phi4_path}")
            return False
        
        # Load config and override attention
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(str(phi4_path), trust_remote_code=True)
        cfg._attn_implementation = "sdpa"
        if hasattr(cfg, 'attn_implementation'):
            cfg.attn_implementation = "sdpa"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(phi4_path),
            config=cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            str(phi4_path),
            trust_remote_code=True,
            use_fast=False
        )
        
        logger.info("Phi-4 model loaded successfully")
        
        # Test with image
        image_path = "/home/himanshu/dev/input_img/img1.png"
        if Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
            prompt = "Please analyze this chemical compound image and explain what you see"
            
            # Prepare inputs
            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            )
            
            # Move to device
            device_inputs = {}
            for k, v in inputs.items():
                if v is not None:
                    device_inputs[k] = v.to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **device_inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = processor.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            logger.info(f"Phi-4 Response: {response}")
            return True
        else:
            logger.error(f"Test image not found at {image_path}")
            return False
            
    except Exception as e:
        logger.error(f"Phi-4 test failed: {e}")
        return False

def test_qwen_model():
    """Test Qwen model with multimodal input"""
    logger.info("=== Testing Qwen Model ===")
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        qwen_path = Path("/home/himanshu/dev/models/QWEN_AWQ")
        if not qwen_path.exists():
            logger.error(f"Qwen model not found at {qwen_path}")
            return False
        
        # Load model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(qwen_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            str(qwen_path),
            trust_remote_code=True
        )
        
        logger.info("Qwen model loaded successfully")
        
        # Test with image
        image_path = "/home/himanshu/dev/input_img/img2.png"
        if Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
            prompt = "Please analyze this chemical compound image and explain what you see"
            
            # Prepare messages for Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device
            device_inputs = {}
            for k, v in inputs.items():
                if v is not None:
                    device_inputs[k] = v.to(model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **device_inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Qwen Response: {response}")
            return True
        else:
            logger.error(f"Test image not found at {image_path}")
            return False
            
    except Exception as e:
        logger.error(f"Qwen test failed: {e}")
        return False

def main():
    """Run tests for both models"""
    logger.info("Starting multimodal model tests...")
    
    # Test Phi-4
    phi4_success = test_phi4_model()
    
    # Test Qwen
    qwen_success = test_qwen_model()
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"Phi-4 Model: {'PASS' if phi4_success else 'FAIL'}")
    logger.info(f"Qwen Model: {'PASS' if qwen_success else 'FAIL'}")
    
    if phi4_success and qwen_success:
        logger.info("All tests passed! Both models work with multimodal input.")
    else:
        logger.warning("Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()



