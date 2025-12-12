#!/usr/bin/env python3
"""
ParallelGenerator: Manages parallel inference across both Phi-4 and Qwen models
Part of the modular multimodal RAG system
"""

import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import onnxruntime_genai as og
import signal
import multiprocessing as mp
from multiprocessing import Process, Queue

logger = logging.getLogger(__name__)

def phi4_worker(prompt: str, images: List[str], model_path: str, result_queue: Queue):
    """Worker function for Phi-4 generation in separate process"""
    try:
        import onnxruntime_genai as og
        from pathlib import Path
        
        logger.info(f"🚀 Phi-4 worker starting for prompt: '{prompt[:50]}...'")
        
        # Create config and model
        config = og.Config(model_path)
        model = og.Model(config)
        processor = model.create_multimodal_processor()
        tokenizer_stream = processor.create_stream()
        
        # Prepare the prompt in Phi-4 format
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Process images if provided
        images_obj = None
        if images:
            try:
                image_paths = [str(img) for img in images if Path(img).exists()]
                if image_paths:
                    images_obj = og.Images.open(*image_paths)
                    for i in range(len(image_paths)):
                        formatted_prompt = f"<|image_{i+1}|>\n{formatted_prompt}"
            except Exception as e:
                logger.warning(f"Could not process images for Phi-4: {e}")
        
        # Process inputs
        inputs = processor(formatted_prompt, images=images_obj)
        
        # Generate response
        params = og.GeneratorParams(model)
        # Apply speed optimizations: reduced max_length + top_k for faster generation
        params.set_search_options(max_length=1024, top_k=40)
        
        generator = og.Generator(model, params)
        response_tokens = generator.generate(inputs)
        
        # Decode response token by token
        response_text = ""
        for token in response_tokens:
            response_text += tokenizer_stream.decode(token)
        
        result_queue.put({
            'success': True,
            'error': None,
            'response': response_text,
            'generation_time': 0,  # Will be calculated by parent
            'model': 'phi4',
            'images_processed': len(images)
        })
        
    except Exception as e:
        logger.error(f"❌ Phi-4 worker error: {e}")
        result_queue.put({
            'success': False,
            'error': str(e),
            'response': None,
            'generation_time': 0,
            'model': 'phi4'
        })

def qwen_worker(prompt: str, images: List[str], model_path: str, result_queue: Queue):
    """Worker function for Qwen generation in separate process"""
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from PIL import Image as PILImage
        
        logger.info(f"🚀 Qwen worker starting for prompt: '{prompt[:50]}...'")
        
        # Load model and processor - specify GPU explicitly for AWQ
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map={"": 0},  # Explicitly use GPU 0 for AWQ
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Prepare images for Qwen
        image_objects = []
        if images:
            for img_path in images:
                if Path(img_path).exists():
                    try:
                        img = PILImage.open(img_path).convert('RGB')
                        image_objects.append(img)
                    except Exception as e:
                        logger.warning(f"Could not load image {img_path}: {e}")
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add images to message if available
        if image_objects:
            for img in image_objects:
                messages[0]["content"].append({"type": "image", "image": img})
        
        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_objects, return_tensors="pt")
        
        # Move inputs to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        result_queue.put({
            'success': True,
            'error': None,
            'response': response_text,
            'generation_time': 0,  # Will be calculated by parent
            'model': 'qwen',
            'images_processed': len(image_objects)
        })
        
    except Exception as e:
        logger.error(f"❌ Qwen worker error: {e}")
        result_queue.put({
            'success': False,
            'error': str(e),
            'response': None,
            'generation_time': 0,
            'model': 'qwen'
        })

class ParallelGenerator:
    """Manages parallel inference across both models"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info("ParallelGenerator initialized")
    
    def resize_image_if_needed(self, image_path: str, max_size: int = 256) -> str:
        """Resize image if it's larger than max_size to reduce token usage"""
        try:
            with Image.open(image_path) as img:
                # Check if image needs resizing
                if max(img.size) > max_size:
                    # Calculate new size maintaining aspect ratio
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    
                    # Resize image
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Save resized image to temp file
                    temp_path = f"/tmp/resized_{Path(image_path).name}"
                    resized_img.save(temp_path)
                    
                    logger.info(f"Resized image from {img.size} to {new_size}")
                    return temp_path
                else:
                    logger.info(f"Image size {img.size} is within limits, no resizing needed")
                    return image_path
        except Exception as e:
            logger.warning(f"Could not resize image {image_path}: {e}")
            return image_path
    
    def generate_phi4_response(self, prompt: str, images: List[str]) -> Dict[str, Any]:
        """Generate response using Phi-4 ONNX model with proper ONNX Runtime GenAI"""
        logger.info(f"🚀 Starting Phi-4 generation for prompt: '{prompt[:50]}...'")
        
        if not self.model_manager.is_model_loaded('phi4'):
            logger.error("❌ Phi-4 ONNX model not loaded")
            return {
                'success': False,
                'error': 'Phi-4 ONNX model not loaded',
                'response': None,
                'generation_time': 0,
                'model': 'phi4'
            }
        
        try:
            start_time = time.time()
            
            # Use ONNX Runtime GenAI for proper Phi-4 inference
            model_path = str(self.model_manager.phi4_path / "gpu" / "gpu-int4-rtn-block-32")
            
            # Create config and model - use default configuration
            config = og.Config(model_path)
            # Don't clear providers, use the default configuration from the model
            
            model = og.Model(config)
            processor = model.create_multimodal_processor()
            tokenizer_stream = processor.create_stream()
            
            # Prepare the prompt in Phi-4 format
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            
            # Process images if provided
            images_obj = None
            if images:
                try:
                    # Resize images to reduce token usage
                    resized_image_paths = []
                    for img_path in images:
                        if Path(img_path).exists():
                            resized_path = self.resize_image_if_needed(img_path, max_size=256)
                            resized_image_paths.append(resized_path)
                    
                    if resized_image_paths:
                        images_obj = og.Images.open(*resized_image_paths)
                        # Add image tokens to prompt
                        for i in range(len(resized_image_paths)):
                            formatted_prompt = f"<|image_{i+1}|>\n{formatted_prompt}"
                except Exception as e:
                    logger.warning(f"Could not process images for Phi-4: {e}")
            
            # Process inputs
            inputs = processor(formatted_prompt, images=images_obj)
            
            # Generate response
            params = og.GeneratorParams(model)
            # Apply speed optimizations: reduced max_length + top_k for faster generation
            params.set_search_options(max_length=1024, top_k=40)
            
            generator = og.Generator(model, params)
            generator.set_inputs(inputs)
            
            # Generate tokens
            response_tokens = []
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                response_tokens.append(new_token)
            
            # Decode response token by token
            response_text = ""
            for token in response_tokens:
                response_text += tokenizer_stream.decode(token)
            
            # Clean up
            del generator
            del model
            
            generation_time = time.time() - start_time
            
            return {
                'success': True,
                'error': None,
                'response': response_text,
                'generation_time': generation_time,
                'model': 'phi4',
                'images_processed': len(images) if images else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating Phi-4 ONNX response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'generation_time': time.time() - start_time if 'start_time' in locals() else 0,
                'model': 'phi4'
            }
    
    def generate_qwen_response(self, prompt: str, images: List[str]) -> Dict[str, Any]:
        """Generate response using Qwen model"""
        logger.info(f"🚀 Starting Qwen generation for prompt: '{prompt[:50]}...'")
        
        if not self.model_manager.is_model_loaded('qwen'):
            logger.error("❌ Qwen model not loaded")
            return {
                'success': False,
                'error': 'Qwen model not loaded',
                'response': None,
                'generation_time': 0,
                'model': 'qwen'
            }
        
        try:
            start_time = time.time()
            
            # Prepare images for Qwen
            processed_images = []
            for img_path in images:
                if Path(img_path).exists():
                    try:
                        image = Image.open(img_path).convert("RGB")
                        processed_images.append(image)
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path} for Qwen: {e}")
                        continue
            
            # Prepare messages for Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Add images to messages if available
            if processed_images:
                for img_path in images:
                    if Path(img_path).exists():
                        messages[0]["content"].insert(0, {
                            "type": "image", 
                            "image": f"file://{img_path}"
                        })
            
            # Prepare inputs using Qwen's chat template
            text = self.model_manager.qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if processed_images:
                inputs = self.model_manager.qwen_processor(
                    text=[text],
                    images=processed_images,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.model_manager.qwen_processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )
            
            # Move to device
            device_inputs = {}
            for k, v in inputs.items():
                if v is not None:
                    device_inputs[k] = v.to(self.model_manager.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model_manager.qwen_model.generate(
                    **device_inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.model_manager.qwen_processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.model_manager.qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            generation_time = time.time() - start_time
            
            return {
                'success': True,
                'error': None,
                'response': response,
                'generation_time': generation_time,
                'model': 'qwen',
                'images_processed': len(processed_images)
            }
            
        except Exception as e:
            logger.error(f"Qwen generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'generation_time': 0,
                'model': 'qwen'
            }
    
    def generate_parallel_responses(self, prompt: str, images: List[str]) -> Dict[str, Any]:
        """Generate responses from both models sequentially with speed optimizations"""
        logger.info("Starting optimized sequential generation with both models")
        start_time = time.time()
        
        # Check which models are available
        phi4_available = self.model_manager.is_model_loaded('phi4')
        qwen_available = self.model_manager.is_model_loaded('qwen')
        
        if not phi4_available and not qwen_available:
            return {
                'success': False,
                'error': 'No models loaded',
                'phi4_result': None,
                'qwen_result': None,
                'total_time': 0
            }
        
        # Sequential generation with speed optimizations
        results = {}
        if phi4_available:
            logger.info("Running Phi-4 generation with speed optimizations")
            try:
                results['phi4'] = self.generate_phi4_response(prompt, images)
            except Exception as e:
                logger.error(f"Phi-4 generation error: {e}")
                results['phi4'] = {
                    'success': False,
                    'error': str(e),
                    'response': None,
                    'generation_time': 0,
                    'model': 'phi4'
                }
        if qwen_available:
            logger.info("Running Qwen generation")
            try:
                results['qwen'] = self.generate_qwen_response(prompt, images)
            except Exception as e:
                logger.error(f"Qwen generation error: {e}")
                results['qwen'] = {
                    'success': False,
                    'error': str(e),
                    'response': None,
                    'generation_time': 0,
                    'model': 'qwen'
                }
        
        total_time = time.time() - start_time
        
        # Prepare final result
        final_result = {
            'success': any(result.get('success', False) for result in results.values()),
            'phi4_result': results.get('phi4'),
            'qwen_result': results.get('qwen'),
            'total_time': total_time,
            'models_used': list(results.keys())
        }
        
        logger.info(f"Optimized sequential generation completed in {total_time:.2f}s")
        return final_result
    
    def generate_single_model_response(self, model_name: str, prompt: str, images: List[str]) -> Dict[str, Any]:
        """Generate response from a single model"""
        if model_name == 'phi4':
            return self.generate_phi4_response(prompt, images)
        elif model_name == 'qwen':
            return self.generate_qwen_response(prompt, images)
        else:
            return {
                'success': False,
                'error': f'Unknown model: {model_name}',
                'response': None,
                'generation_time': 0,
                'model': model_name
            }
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generation performance"""
        return {
            'executor_max_workers': self.executor._max_workers,
            'models_loaded': self.model_manager.get_loaded_models(),
            'device': self.model_manager.device
        }
    
    def shutdown(self):
        """Shutdown the parallel generator"""
        logger.info("Shutting down ParallelGenerator")
        self.executor.shutdown(wait=True)
