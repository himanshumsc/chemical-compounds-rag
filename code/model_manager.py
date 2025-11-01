#!/usr/bin/env python3
"""
ModelManager: Handles loading and management of both Phi-4 and Qwen models
Part of the modular multimodal RAG system
"""

import torch
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration
import threading
import time
import onnxruntime as ort
import numpy as np

# Suppress AutoAWQ deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and lifecycle of both Phi-4 and Qwen models"""
    
    def __init__(self, phi4_path: Path, qwen_path: Path, device: str = "auto"):
        self.phi4_path = phi4_path
        self.qwen_path = qwen_path
        self.device = self._get_device(device)
        
        # Model instances
        self.phi4_model = None
        self.phi4_processor = None
        self.phi4_onnx_sessions = {}  # ONNX runtime sessions
        self.qwen_model = None
        self.qwen_processor = None
        
        # Loading status
        self.phi4_loaded = False
        self.qwen_loaded = False
        self.loading_lock = threading.Lock()
        
        # Performance tracking
        self.load_times = {}
        self.memory_usage = {}
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine best device for processing"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_phi4_model(self) -> bool:
        """Mark Phi-4 ONNX model as available; defer runtime loading to onnxruntime-genai.

        We no longer create raw onnxruntime InferenceSession(s) here to avoid
        performance warnings from added Memcpy nodes and to keep a single
        execution path via onnxruntime-genai in the generator.
        """
        with self.loading_lock:
            if self.phi4_loaded:
                logger.info("Phi-4 ONNX model already loaded")
                return True
            
            try:
                logger.info(f"Loading Phi-4 ONNX model from {self.phi4_path}")
                start_time = time.time()
                
                # Check if ONNX model exists
                if not self.phi4_path.exists():
                    raise Exception(f"Phi-4 ONNX model not found at {self.phi4_path}")
                
                # Verify expected onnxruntime-genai model dir exists
                onnx_dir = self.phi4_path / "gpu" / "gpu-int4-rtn-block-32"
                if not onnx_dir.exists():
                    raise Exception(f"ONNX model directory not found at {onnx_dir}")
                
                # Do not create raw ORT sessions here; generator will construct
                # an og.Model with its own optimal providers.
                self.phi4_onnx_sessions = {}
                
                # For Phi-4 ONNX, we don't need to load a separate tokenizer
                # ONNX Runtime GenAI handles tokenization internally
                self.phi4_processor = None
                logger.info("Phi-4 ONNX uses built-in tokenization (no separate tokenizer needed)")
                
                load_time = time.time() - start_time
                self.load_times['phi4'] = load_time
                
                # Track memory usage
                if torch.cuda.is_available():
                    self.memory_usage['phi4'] = torch.cuda.memory_allocated() / 1024**3
                
                self.phi4_loaded = True
                logger.info(f"Phi-4 ONNX availability confirmed in {load_time:.2f}s (using onnxruntime-genai at inference time)")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load Phi-4 ONNX model: {e}")
                self.phi4_onnx_sessions = {}
                self.phi4_processor = None
                self.phi4_loaded = False
                return False
    
    def load_qwen_model(self) -> bool:
        """Load Qwen2.5-VL-AWQ model"""
        with self.loading_lock:
            if self.qwen_loaded:
                logger.info("Qwen model already loaded")
                return True
            
            try:
                logger.info(f"Loading Qwen model from {self.qwen_path}")
                start_time = time.time()
                
                # Check if Qwen model exists
                if not self.qwen_path.exists():
                    raise Exception(f"Qwen model not found at {self.qwen_path}")
                
                # Load Qwen model
                self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(self.qwen_path),
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Load processor
                self.qwen_processor = AutoProcessor.from_pretrained(
                    str(self.qwen_path),
                    trust_remote_code=True
                )
                
                load_time = time.time() - start_time
                self.load_times['qwen'] = load_time
                
                # Track memory usage
                if torch.cuda.is_available():
                    self.memory_usage['qwen'] = torch.cuda.memory_allocated() / 1024**3
                
                self.qwen_loaded = True
                logger.info(f"Qwen model loaded successfully in {load_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load Qwen model: {e}")
                self.qwen_model = None
                self.qwen_processor = None
                self.qwen_loaded = False
                return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load both models and return status"""
        logger.info("Loading models...")
        
        # Load both models
        phi4_thread = threading.Thread(target=self.load_phi4_model)
        qwen_thread = threading.Thread(target=self.load_qwen_model)
        
        phi4_thread.start()
        qwen_thread.start()
        
        phi4_thread.join()
        qwen_thread.join()
        
        status = {
            'phi4': self.phi4_loaded,
            'qwen': self.qwen_loaded,
            'both_loaded': self.phi4_loaded and self.qwen_loaded
        }
        
        logger.info(f"Model loading status: {status}")
        return status
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'device': self.device,
            'phi4_loaded': self.phi4_loaded,
            'qwen_loaded': self.qwen_loaded,
            'load_times': self.load_times,
            'memory_usage': self.memory_usage
        }
        
        if torch.cuda.is_available():
            info['current_gpu_memory'] = torch.cuda.memory_allocated() / 1024**3
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
        
        return info
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model to free memory"""
        with self.loading_lock:
            if model_name == 'phi4':
                if self.phi4_loaded:
                    del self.phi4_model
                    del self.phi4_processor
                    self.phi4_model = None
                    self.phi4_processor = None
                    self.phi4_loaded = False
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info("Phi-4 model unloaded")
                    return True
            
            elif model_name == 'qwen':
                if self.qwen_loaded:
                    del self.qwen_model
                    del self.qwen_processor
                    self.qwen_model = None
                    self.qwen_processor = None
                    self.qwen_loaded = False
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info("Qwen model unloaded")
                    return True
            
            return False
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded"""
        if model_name == 'phi4':
            return self.phi4_loaded
        elif model_name == 'qwen':
            return self.qwen_loaded
        return False
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        loaded = []
        if self.phi4_loaded:
            loaded.append('phi4')
        if self.qwen_loaded:
            loaded.append('qwen')
        return loaded
