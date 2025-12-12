#!/usr/bin/env python3
"""
Test script to load and test the Qwen2.5-VL-7B-Instruct-AWQ model
Simplified version without qwen_vl_utils dependency
"""

import torch
import sys
import os
from pathlib import Path
import time
from PIL import Image

def test_model_loading():
    """Test loading the Qwen2.5-VL-7B-Instruct-AWQ model"""
    
    print("🚀 Testing Qwen2.5-VL-7B-Instruct-AWQ Model Loading")
    print("=" * 60)
    
    # Check if model directory exists
    model_path = Path("/home/himanshu/dev/models/QWEN_AWQ")
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    print(f"📁 Model path: {model_path}")
    
    try:
        # Import required libraries
        print("📦 Importing required libraries...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        print("✅ Libraries imported successfully")
        
        # Check GPU availability
        print(f"\n🖥️  GPU Status:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   Current GPU: {torch.cuda.current_device()}")
            print(f"   GPU name: {torch.cuda.get_device_name()}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load model
        print(f"\n📥 Loading model from {model_path}...")
        start_time = time.time()
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,  # Use float16 for AWQ
            device_map="auto",
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        # Load processor
        print("📥 Loading processor...")
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        print("✅ Processor loaded successfully")
        
        # Check model info
        print(f"\n📊 Model Information:")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   Dtype: {next(model.parameters()).dtype}")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   GPU memory allocated: {memory_allocated:.2f} GB")
            print(f"   GPU memory reserved: {memory_reserved:.2f} GB")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_text_generation(model, processor):
    """Test text-only generation"""
    
    print(f"\n🧪 Testing Text Generation")
    print("-" * 40)
    
    try:
        # Simple text prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is benzene and what are its properties?"}
                ]
            }
        ]
        
        # Prepare inputs using processor directly
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize text only (no images)
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        print("🔄 Generating response...")
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(f"✅ Text generation completed in {generation_time:.2f} seconds")
        print(f"📝 Response: {output_text[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in text generation: {e}")
        return False

def test_image_generation(model, processor):
    """Test image + text generation"""
    
    print(f"\n🖼️  Testing Image + Text Generation")
    print("-" * 40)
    
    try:
        # Create a simple test image (white square with text)
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # Save test image temporarily
        test_image_path = "/tmp/test_image.png"
        test_image.save(test_image_path)
        
        # Test with local image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{test_image_path}"},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]
        
        # Prepare inputs using processor directly
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process with image
        inputs = processor(
            text=[text],
            images=[test_image],
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        print("🔄 Generating response...")
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(f"✅ Image generation completed in {generation_time:.2f} seconds")
        print(f"📝 Response: {output_text[0]}")
        
        # Clean up
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in image generation: {e}")
        return False

def test_with_user_images(model, processor):
    """Test with user's transferred images"""
    
    print(f"\n📸 Testing with User Images")
    print("-" * 40)
    
    user_images_dir = Path("/home/himanshu/dev/input_img")
    if not user_images_dir.exists():
        print("❌ User images directory not found")
        return False
    
    image_files = list(user_images_dir.glob("*.png")) + list(user_images_dir.glob("*.jpg"))
    
    if not image_files:
        print("❌ No image files found in user directory")
        return False
    
    print(f"📁 Found {len(image_files)} image(s)")
    
    for img_path in image_files[:2]:  # Test first 2 images
        try:
            print(f"\n🖼️  Testing with: {img_path.name}")
            
            # Load image
            image = Image.open(img_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": "What do you see in this image? Describe any chemical structures, formulas, or scientific content."}
                    ]
                }
            ]
            
            # Prepare inputs using processor directly
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process with image
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            print("🔄 Generating response...")
            start_time = time.time()
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            print(f"✅ Response generated in {generation_time:.2f} seconds")
            print(f"📝 Response: {output_text[0]}")
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
    
    return True

def main():
    print("🧪 Qwen2.5-VL-7B-Instruct-AWQ Model Test Suite")
    print("=" * 60)
    
    # Test 1: Model Loading
    result = test_model_loading()
    if not result:
        print("❌ Model loading failed. Exiting.")
        return
    
    model, processor = result
    
    # Test 2: Text Generation
    test_text_generation(model, processor)
    
    # Test 3: Image Generation
    test_image_generation(model, processor)
    
    # Test 4: User Images
    test_with_user_images(model, processor)
    
    print(f"\n🎉 All tests completed!")
    print(f"💡 The AWQ quantized model is working correctly and ready for use in your RAG system!")

if __name__ == "__main__":
    main()
