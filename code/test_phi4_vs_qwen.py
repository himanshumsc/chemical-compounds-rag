#!/usr/bin/env python3
"""
Comparative test script for Phi-4 and Qwen2.5-VL-7B-Instruct-AWQ models
Loads both models and compares their responses side by side
"""

import torch
import sys
import os
from pathlib import Path
import time
from PIL import Image
import gc

def load_phi4_model():
    """Load Phi-4 multimodal model"""
    
    print("🚀 Loading Phi-4 Multimodal Model")
    print("=" * 50)
    
    phi4_path = Path("/home/himanshu/dev/models/PHI4")
    if not phi4_path.exists():
        print(f"❌ Phi-4 model directory not found: {phi4_path}")
        return None, None
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        
        print("📥 Loading Phi-4 model...")
        start_time = time.time()
        
        # Load Phi-4 with specific configuration (override flash attention)
        from transformers import AutoConfig
        
        # Load config and override attention implementation
        config = AutoConfig.from_pretrained(str(phi4_path), trust_remote_code=True)
        config._attn_implementation = "sdpa"  # Override flash attention
        
        model = AutoModelForCausalLM.from_pretrained(
            str(phi4_path),
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"  # Use scaled dot-product attention
        )
        
        processor = AutoProcessor.from_pretrained(str(phi4_path), trust_remote_code=True)
        
        load_time = time.time() - start_time
        print(f"✅ Phi-4 loaded in {load_time:.2f} seconds")
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 Phi-4 GPU memory: {memory_allocated:.2f} GB")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading Phi-4: {e}")
        return None, None

def load_qwen_model():
    """Load Qwen2.5-VL-AWQ model"""
    
    print("\n🚀 Loading Qwen2.5-VL-7B-Instruct-AWQ Model")
    print("=" * 50)
    
    qwen_path = Path("/home/himanshu/dev/models/QWEN_AWQ")
    if not qwen_path.exists():
        print(f"❌ Qwen model directory not found: {qwen_path}")
        return None, None
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print("📥 Loading Qwen AWQ model...")
        start_time = time.time()
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(qwen_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(str(qwen_path), trust_remote_code=True)
        
        load_time = time.time() - start_time
        print(f"✅ Qwen AWQ loaded in {load_time:.2f} seconds")
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 Qwen GPU memory: {memory_allocated:.2f} GB")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading Qwen: {e}")
        return None, None

def test_text_query_phi4(model, processor, query):
    """Test text query with Phi-4"""
    
    try:
        print(f"\n🤖 Phi-4 Response:")
        print("-" * 30)
        
        # Prepare prompt for Phi-4
        prompt = f"<|image|>\n<|user|>\n{query}\n<|assistant|>\n"
        
        # Process text only (no image for text queries)
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"⏱️  Time: {generation_time:.2f}s")
        print(f"📝 Response: {response}")
        
        return response, generation_time
        
    except Exception as e:
        print(f"❌ Phi-4 error: {e}")
        return None, 0

def test_text_query_qwen(model, processor, query):
    """Test text query with Qwen"""
    
    try:
        print(f"\n🤖 Qwen Response:")
        print("-" * 30)
        
        # Prepare messages for Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ]
            }
        ]
        
        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"⏱️  Time: {generation_time:.2f}s")
        print(f"📝 Response: {response}")
        
        return response, generation_time
        
    except Exception as e:
        print(f"❌ Qwen error: {e}")
        return None, 0

def test_image_query_phi4(model, processor, image_path, query):
    """Test image query with Phi-4"""
    
    try:
        print(f"\n🤖 Phi-4 Response:")
        print("-" * 30)
        
        # Load image
        image = Image.open(image_path)
        
        # Prepare prompt for Phi-4
        prompt = f"<|image|>\n<|user|>\n{query}\n<|assistant|>\n"
        
        # Process with image
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"⏱️  Time: {generation_time:.2f}s")
        print(f"📝 Response: {response}")
        
        return response, generation_time
        
    except Exception as e:
        print(f"❌ Phi-4 image error: {e}")
        return None, 0

def test_image_query_qwen(model, processor, image_path, query):
    """Test image query with Qwen"""
    
    try:
        print(f"\n🤖 Qwen Response:")
        print("-" * 30)
        
        # Load image
        image = Image.open(image_path)
        
        # Prepare messages for Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": query}
                ]
            }
        ]
        
        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"⏱️  Time: {generation_time:.2f}s")
        print(f"📝 Response: {response}")
        
        return response, generation_time
        
    except Exception as e:
        print(f"❌ Qwen image error: {e}")
        return None, 0

def run_text_tests(phi4_model, phi4_processor, qwen_model, qwen_processor):
    """Run text-only comparison tests"""
    
    print("\n" + "="*80)
    print("📝 TEXT-ONLY COMPARISON TESTS")
    print("="*80)
    
    text_queries = [
        "What is benzene and what are its chemical properties?",
        "Explain the process of distillation in chemistry.",
        "What are the safety considerations when working with organic solvents?",
        "Describe the molecular structure of methane and its uses."
    ]
    
    results = []
    
    for i, query in enumerate(text_queries, 1):
        print(f"\n🔬 Test {i}: {query}")
        print("="*60)
        
        # Test Phi-4
        phi4_response, phi4_time = test_text_query_phi4(phi4_model, phi4_processor, query)
        
        # Test Qwen
        qwen_response, qwen_time = test_text_query_qwen(qwen_model, qwen_processor, query)
        
        # Store results
        results.append({
            'query': query,
            'phi4_response': phi4_response,
            'phi4_time': phi4_time,
            'qwen_response': qwen_response,
            'qwen_time': qwen_time
        })
        
        # Clear GPU cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def run_image_tests(phi4_model, phi4_processor, qwen_model, qwen_processor):
    """Run image comparison tests"""
    
    print("\n" + "="*80)
    print("🖼️  IMAGE COMPARISON TESTS")
    print("="*80)
    
    user_images_dir = Path("/home/himanshu/dev/input_img")
    if not user_images_dir.exists():
        print("❌ User images directory not found")
        return []
    
    image_files = list(user_images_dir.glob("*.png")) + list(user_images_dir.glob("*.jpg"))
    
    if not image_files:
        print("❌ No image files found")
        return []
    
    image_query = "What do you see in this image? Describe any chemical structures, formulas, or scientific content."
    
    results = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n🔬 Image Test {i}: {img_path.name}")
        print("="*60)
        
        # Test Phi-4
        phi4_response, phi4_time = test_image_query_phi4(phi4_model, phi4_processor, img_path, image_query)
        
        # Test Qwen
        qwen_response, qwen_time = test_image_query_qwen(qwen_model, qwen_processor, img_path, image_query)
        
        # Store results
        results.append({
            'image': img_path.name,
            'query': image_query,
            'phi4_response': phi4_response,
            'phi4_time': phi4_time,
            'qwen_response': qwen_response,
            'qwen_time': qwen_time
        })
        
        # Clear GPU cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def print_summary(text_results, image_results):
    """Print comparison summary"""
    
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    # Text results summary
    if text_results:
        print("\n📝 TEXT TESTS SUMMARY:")
        print("-" * 40)
        
        phi4_total_time = sum(r['phi4_time'] for r in text_results)
        qwen_total_time = sum(r['qwen_time'] for r in text_results)
        
        print(f"Phi-4 Total Time: {phi4_total_time:.2f}s")
        print(f"Qwen Total Time:  {qwen_total_time:.2f}s")
        print(f"Speed Advantage: {'Phi-4' if phi4_total_time < qwen_total_time else 'Qwen'} ({abs(phi4_total_time - qwen_total_time):.2f}s difference)")
    
    # Image results summary
    if image_results:
        print("\n🖼️  IMAGE TESTS SUMMARY:")
        print("-" * 40)
        
        phi4_total_time = sum(r['phi4_time'] for r in image_results)
        qwen_total_time = sum(r['qwen_time'] for r in image_results)
        
        print(f"Phi-4 Total Time: {phi4_total_time:.2f}s")
        print(f"Qwen Total Time:  {qwen_total_time:.2f}s")
        print(f"Speed Advantage: {'Phi-4' if phi4_total_time < qwen_total_time else 'Qwen'} ({abs(phi4_total_time - qwen_total_time):.2f}s difference)")
    
    # Overall GPU memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"\n💾 Current GPU Memory Usage: {memory_allocated:.2f} GB")

def main():
    print("🧪 Phi-4 vs Qwen2.5-VL-AWQ Comparative Test Suite")
    print("=" * 80)
    
    # Check GPU status
    print(f"\n🖥️  GPU Status:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load both models
    phi4_model, phi4_processor = load_phi4_model()
    if phi4_model is None:
        print("❌ Failed to load Phi-4. Exiting.")
        return
    
    qwen_model, qwen_processor = load_qwen_model()
    if qwen_model is None:
        print("❌ Failed to load Qwen. Exiting.")
        return
    
    print(f"\n✅ Both models loaded successfully!")
    
    # Run text comparison tests
    text_results = run_text_tests(phi4_model, phi4_processor, qwen_model, qwen_processor)
    
    # Run image comparison tests
    image_results = run_image_tests(phi4_model, phi4_processor, qwen_model, qwen_processor)
    
    # Print summary
    print_summary(text_results, image_results)
    
    print(f"\n🎉 Comparative testing completed!")
    print(f"💡 Both models are ready for integration into your RAG system!")

if __name__ == "__main__":
    main()
