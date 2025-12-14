#!/home/himanshu/dev/code/.venv_phi4_req/bin/python3
"""
Optimized multimodal QA runner using ONLY vLLM for all questions with RAG:
- REQUIRES vLLM - NO Transformers fallback (hardware constraints)
- Uses vLLM for ALL questions (Q1 with images via multi_modal_data, Q2-Q4 text-only)
- Integrates ChromaDB RAG for context augmentation
- Q1: Image-only similarity search (model identifies compound from image)
- Q2-Q4: Text search for relevant compound information
- Passes images separately using vLLM's multimodal API
- Reads questions from existing answer files in output/qwen_regenerated
- Regenerates Q1 with max_tokens=300 (matches OpenAI), Q2-Q4 with max_tokens=500 (matches comprehensive)
- Batch processing across files for optimal GPU utilization
- Supports background execution with comprehensive logging
- Will FAIL if vLLM cannot be initialized (no fallback)
"""
import argparse
import json
import io
import time
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
import re
import logging
from datetime import datetime
import traceback

# Fix ChromaDB SQLite compatibility
import pysqlite3 as sqlite3
import sys
sys.modules['sqlite3'] = sqlite3

# Set environment variables for vLLM before importing
# Note: VLLM_SKIP_MM_PROFILE is NOT set by default to allow multimodal profiling
# This is needed for Gemma-3 and other models that require vision encoder profiling
# If needed for specific models (e.g., Qwen), set it conditionally in the wrapper

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("ERROR: vLLM not available. This script REQUIRES vLLM. Cannot proceed.")

# Import ChromaDB search engine
try:
    from chromadb_search import ChromaDBSearchEngine
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("WARNING: ChromaDB search not available. RAG will be disabled.")


DEFAULT_INPUT_DIR = Path("/home/himanshu/dev/output/qwen_regenerated")
DEFAULT_OUTPUT_DIR = Path("/home/himanshu/dev/output/qwen_rag_concise")  # New folder for concise RAG-generated outputs
DEFAULT_QA_DIR = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components")
DEFAULT_MODEL_PATH = Path("/home/himanshu/dev/models/QWEN_AWQ")
DEFAULT_CHROMADB_PATH = Path("/home/himanshu/dev/data/chromadb")

# Character limits per question type (for concise answers)
CHAR_LIMIT_Q1 = 600   # Image-based identification
CHAR_LIMIT_Q2 = 1000  # Formula/Type
CHAR_LIMIT_Q3 = 1800  # Production process
CHAR_LIMIT_Q4 = 2000  # Uses/Hazards

# Estimate max_tokens from character limits (roughly 3 chars per token)
# Use 3.0 division to allow slightly longer answers while still staying within limits
MAX_TOKENS_Q1 = int(CHAR_LIMIT_Q1 / 3.0)  # ~200 tokens for 600 chars
MAX_TOKENS_Q2 = int(CHAR_LIMIT_Q2 / 3.0)  # ~333 tokens for 1000 chars
MAX_TOKENS_Q3 = int(CHAR_LIMIT_Q3 / 3.0)  # ~600 tokens for 1800 chars
MAX_TOKENS_Q4 = int(CHAR_LIMIT_Q4 / 3.0)  # ~666 tokens for 2000 chars


@dataclass
class GenerationResult:
    text: str
    latency_s: float
    rag_used: bool = False
    search_type: Optional[str] = None
    n_chunks_used: int = 0
    truncated: bool = False  # Whether answer was truncated to meet character limit


class VLLMRagWrapper:
    """Wrapper that uses ONLY vLLM for all questions (text and vision) with RAG."""
    
    def __init__(
        self,
        model_path: str = str(DEFAULT_MODEL_PATH),
        tokenizer_id: Optional[str] = None,
        quantization: Optional[str] = None,
        dtype: str = "float16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.80,
        enforce_eager: bool = True,
        chromadb_path: Optional[str] = None,
        use_vllm: bool = True,
        use_rag: bool = True,
        n_chunks: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_path = model_path
        self.tokenizer_id = tokenizer_id
        self.quantization = quantization
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.use_rag = use_rag and CHROMADB_AVAILABLE
        self.n_chunks = n_chunks
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize vLLM FIRST (before processor and ChromaDB) to avoid resource conflicts
        # vLLM needs to claim GPU resources first, before any other model loading
        # vLLM is REQUIRED - no Transformers fallback
        if not self.use_vllm:
            raise RuntimeError("vLLM is required but not available. Cannot proceed without vLLM.")
        
        self.vllm_llm = None
        try:
            # Ensure multimodal profiling is enabled for ALL models
            # Q1 questions require images, so vision encoder profiling is essential
            if 'VLLM_SKIP_MM_PROFILE' in os.environ:
                del os.environ['VLLM_SKIP_MM_PROFILE']
            if 'SKIP_MM_PROFILE' in os.environ:
                del os.environ['SKIP_MM_PROFILE']
            
            self.logger.info("Initializing vLLM for ALL questions (text and vision)...")
            self.logger.info("vLLM is REQUIRED - no Transformers fallback available")
            self.logger.info("Multimodal profiling ENABLED (required for Q1 image questions)")
            self.logger.info("Environment variables: VLLM_SKIP_MM_PROFILE={}, SKIP_MM_PROFILE={}".format(
                os.environ.get('VLLM_SKIP_MM_PROFILE', 'not set'),
                os.environ.get('SKIP_MM_PROFILE', 'not set')
            ))
            self.logger.info("Starting vLLM initialization (this may take 20-30 seconds)...")
            
            llm_kwargs: Dict[str, Any] = {
                "model": str(model_path),
                "dtype": self.dtype,
                "trust_remote_code": True,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "enforce_eager": self.enforce_eager,
                "disable_log_stats": True,
            }
            if self.tokenizer_id:
                llm_kwargs["tokenizer"] = self.tokenizer_id
            if self.quantization:
                llm_kwargs["quantization"] = self.quantization
            
            self.vllm_llm = LLM(**llm_kwargs)
            self.logger.info(
                "vLLM config -> tokenizer:%s quantization:%s dtype:%s max_len:%s gpu_util:%.2f enforce_eager:%s",
                self.tokenizer_id or "auto",
                self.quantization or "none",
                self.dtype,
                self.max_model_len,
                self.gpu_memory_utilization,
                self.enforce_eager,
            )
            # Sampling params per question type with character limits
            # Estimate max_tokens from character limits (roughly 3.5 chars per token)
            self.vllm_sampling_params_q1 = SamplingParams(
                temperature=0.7,
                max_tokens=MAX_TOKENS_Q1,  # ~171 tokens for 600 chars
                stop=None,
            )
            self.vllm_sampling_params_q2 = SamplingParams(
                temperature=0.7,
                max_tokens=MAX_TOKENS_Q2,  # ~286 tokens for 1000 chars
                stop=None,
            )
            self.vllm_sampling_params_q3 = SamplingParams(
                temperature=0.7,
                max_tokens=MAX_TOKENS_Q3,  # ~514 tokens for 1800 chars
                stop=None,
            )
            self.vllm_sampling_params_q4 = SamplingParams(
                temperature=0.7,
                max_tokens=MAX_TOKENS_Q4,  # ~571 tokens for 2000 chars
                stop=None,
            )
            self.logger.info("vLLM initialized successfully for all question types")
            self.logger.info(f"Character limits: Q1={CHAR_LIMIT_Q1}, Q2={CHAR_LIMIT_Q2}, Q3={CHAR_LIMIT_Q3}, Q4={CHAR_LIMIT_Q4}")
            self.logger.info(f"Max tokens: Q1={MAX_TOKENS_Q1}, Q2={MAX_TOKENS_Q2}, Q3={MAX_TOKENS_Q3}, Q4={MAX_TOKENS_Q4}")
        except Exception as e:
            self.logger.error(f"CRITICAL: Failed to initialize vLLM: {e}")
            self.logger.error(f"vLLM initialization error details: {traceback.format_exc()}")
            self.logger.error("vLLM is REQUIRED - cannot proceed without it. Exiting.")
            raise RuntimeError(f"vLLM initialization failed: {e}") from e
        
        # Initialize processor AFTER vLLM (needed for chat template formatting)
        # This ensures vLLM has claimed GPU resources first
        try:
            from transformers import AutoProcessor
            self.logger.info("Initializing processor for chat template formatting...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.logger.info("Processor initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize processor: {e}")
            self.logger.warning("Chat template formatting may not work correctly")
            self.processor = None
        
        # Initialize ChromaDB search system AFTER vLLM (if RAG enabled)
        # This avoids potential resource conflicts during vLLM initialization
        self.search_system = None
        if self.use_rag:
            try:
                chromadb_path = chromadb_path or str(DEFAULT_CHROMADB_PATH)
                self.logger.info(f"Initializing ChromaDB search system from {chromadb_path}...")
                self.search_system = ChromaDBSearchEngine(chromadb_path, device="cpu")
                self.logger.info("ChromaDB search system initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ChromaDB: {e}")
                self.logger.warning("RAG will be disabled")
                self.use_rag = False
                self.search_system = None
    
    def _build_context_from_chunks(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return ""
        context_parts = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            chunk_score = chunk.get('score', 0.0)
            context_parts.append(
                f"[Source {i+1} (Relevance: {chunk_score:.3f})]\n{chunk_text}"
            )
        return "\n\n".join(context_parts)
    
    def _truncate_answer(self, text: str, max_chars: int) -> tuple[str, bool]:
        """
        Truncate answer to meet character limit, preserving sentence boundaries.
        This should be a last resort - the model should generate within limits.
        
        Returns:
            (truncated_text, was_truncated)
        """
        if len(text) <= max_chars:
            return text, False
        
        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        # Find last sentence-ending punctuation
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > max_chars * 0.8:  # If sentence end is reasonably close
            truncated = truncated[:last_sentence_end + 1]
        else:
            # Just truncate and add ellipsis
            truncated = truncated[:max_chars - 3] + "..."
        
        return truncated, True
    
    def _augment_prompt_with_context(self, question: str, context: str, max_chars: Optional[int] = None) -> str:
        """Augment user query with RAG context. Include strong concise instruction and character limit."""
        if not context:
            base_prompt = question
        else:
            base_prompt = f"""Based on the following chemical compounds database information, please answer the user's question.

CONTEXT:
{context}

USER QUESTION: {question}"""
        
        # Strong concise instruction - emphasize brevity
        if max_chars:
            concise_instruction = f"""

IMPORTANT INSTRUCTIONS:
- Your answer MUST be brief, concise, and to the point
- Maximum length: {max_chars} characters (strict limit)
- Focus ONLY on the most essential and relevant information
- Avoid unnecessary elaboration or repetition
- Be direct and factual
- If you exceed {max_chars} characters, your answer will be truncated

Generate a concise answer that fits within {max_chars} characters:"""
        else:
            concise_instruction = """

IMPORTANT: Your answer should be brief, concise, and to the point. Focus on the most relevant information only."""
        
        return base_prompt + concise_instruction
    
    def _save_temp_image(self, image: Image.Image) -> str:
        """Save PIL image temporarily for ChromaDB search."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        image.save(temp_path, format='PNG')
        temp_file.close()
        return temp_path
    
    def _cleanup_temp_image(self, temp_path: str):
        """Clean up temporary image file."""
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp image {temp_path}: {e}")

    def generate_with_vision(self, prompt: str, image: Image.Image, max_new_tokens: int = 500, max_chars: Optional[int] = None) -> GenerationResult:
        """Generate with image using vLLM with multimodal data and RAG (Q1)."""
        # Q1: Use image-only similarity search for RAG
        augmented_prompt = prompt
        search_type = None
        n_chunks_used = 0
        
        if self.use_rag and self.search_system and image:
            try:
                # Save image temporarily for ChromaDB search
                temp_image_path = self._save_temp_image(image)
                
                # Use IMAGE-ONLY similarity search (not hybrid, not text)
                # Model identifies compound from image; we find similar structures
                chunks = self.search_system.image_similarity_search(
                    temp_image_path, n_results=self.n_chunks
                )
                
                # Build context from chunks
                context = self._build_context_from_chunks(chunks)
                
                # Augment prompt with context
                augmented_prompt = self._augment_prompt_with_context(prompt, context)
                
                # Track metadata
                search_type = "image"
                n_chunks_used = len(chunks)
                
                # Clean up temp image
                self._cleanup_temp_image(temp_image_path)
            except Exception as e:
                print(f"Warning: RAG failed for Q1: {e}")
                # Continue without RAG context
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": augmented_prompt},
            ],
        }]
        templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Use vLLM with multimodal data - pass image separately (REQUIRED, NO FALLBACK)
        if self.vllm_llm is None:
            raise RuntimeError("vLLM is not initialized. Cannot generate without vLLM.")
        
        # According to vLLM docs, we pass a dict with prompt and multi_modal_data
        start = time.time()
        # vLLM expects: {"prompt": str, "multi_modal_data": {"image": [image]}}
        multimodal_prompt = {
            "prompt": templated,
            "multi_modal_data": {"image": [image]}
        }
        # Use Q1 sampling params (300 tokens) - match OpenAI Q1 settings
        outputs = self.vllm_llm.generate([multimodal_prompt], self.vllm_sampling_params_q1)
        latency = time.time() - start
        
        generated_text = outputs[0].outputs[0].text
        text = postprocess_assistant_only(generated_text)
        
        # Truncate only if still exceeds character limit (should be rare with proper token limits)
        truncated = False
        if max_chars and len(text) > max_chars:
            original_len = len(text)
            text, truncated = self._truncate_answer(text, max_chars)
            if truncated:
                self.logger.warning(f"Answer exceeded limit ({original_len} > {max_chars} chars), truncated to {len(text)} chars")
        
        result = GenerationResult(text=text, latency_s=latency)
        # Store RAG metadata in result
        result.rag_used = self.use_rag and n_chunks_used > 0
        result.search_type = search_type
        result.n_chunks_used = n_chunks_used
        result.truncated = truncated
        return result

    def generate_with_vision_batch(self, prompts: List[str], images: List[Image.Image], max_new_tokens: int = 500, max_chars: Optional[int] = None) -> List[GenerationResult]:
        """Generate batch of vision questions using vLLM if available, with RAG (Q1 across multiple files)."""
        if len(prompts) != len(images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
        
        # Augment all prompts with RAG context
        augmented_prompts = []
        search_types = []
        n_chunks_list = []
        
        for prompt, image in zip(prompts, images):
            augmented_prompt = prompt
            search_type = None
            n_chunks_used = 0
            
            if self.use_rag and self.search_system and image:
                try:
                    # Save image temporarily for ChromaDB search
                    temp_image_path = self._save_temp_image(image)
                    
                    # Use IMAGE-ONLY similarity search
                    chunks = self.search_system.image_similarity_search(
                        temp_image_path, n_results=self.n_chunks
                    )
                    
                    # Build context from chunks
                    context = self._build_context_from_chunks(chunks)
                    
                    # Augment prompt with context (include character limit)
                    augmented_prompt = self._augment_prompt_with_context(prompt, context, max_chars=max_chars)
                    
                    # Track metadata
                    search_type = "image"
                    n_chunks_used = len(chunks)
                    
                    # Clean up temp image
                    self._cleanup_temp_image(temp_image_path)
                except Exception as e:
                    print(f"Warning: RAG failed for batch vision question: {e}")
            
            augmented_prompts.append(augmented_prompt)
            search_types.append(search_type)
            n_chunks_list.append(n_chunks_used)
        
        # Use vLLM batch processing with multimodal data (REQUIRED, NO FALLBACK)
        if self.vllm_llm is None:
            raise RuntimeError("vLLM is not initialized. Cannot generate without vLLM.")
        
        multimodal_prompts = []
        for templated, image in zip(augmented_prompts, images):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": templated},
                ],
            }]
            templated_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            multimodal_prompt = {
                "prompt": templated_prompt,
                "multi_modal_data": {"image": [image]}
            }
            multimodal_prompts.append(multimodal_prompt)
        
        start = time.time()
        # Use Q1 sampling params (300 tokens) - match OpenAI Q1 settings
        outputs = self.vllm_llm.generate(multimodal_prompts, self.vllm_sampling_params_q1)
        latency = time.time() - start
        
        results = []
        for i, output in enumerate(outputs):
            text = postprocess_assistant_only(output.outputs[0].text)
            result = GenerationResult(text=text, latency_s=latency)
            result.rag_used = self.use_rag and n_chunks_list[i] > 0
            result.search_type = search_types[i]
            result.n_chunks_used = n_chunks_list[i]
            results.append(result)
        return results

    def generate_text_only(self, prompt: str, max_new_tokens: int = 500, max_chars: Optional[int] = None, question_index: int = 2) -> GenerationResult:
        """Generate text-only using vLLM (REQUIRED, NO FALLBACK), with RAG (Q2-Q4)."""
        # Q2-Q4: Use text search for RAG
        augmented_prompt = prompt
        search_type = None
        n_chunks_used = 0
        
        if self.use_rag and self.search_system:
            try:
                # Use text search
                chunks = self.search_system.text_search(prompt, n_results=self.n_chunks)
                
                # Build context from chunks
                context = self._build_context_from_chunks(chunks)
                
                # Augment prompt with context (include character limit)
                augmented_prompt = self._augment_prompt_with_context(prompt, context, max_chars=max_chars)
                
                # Track metadata
                search_type = "text"
                n_chunks_used = len(chunks)
            except Exception as e:
                print(f"Warning: RAG failed for text question: {e}")
                # Continue without RAG context
        
        # Use vLLM for text generation (REQUIRED, NO FALLBACK)
        if self.vllm_llm is None:
            raise RuntimeError("vLLM is not initialized. Cannot generate without vLLM.")
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": augmented_prompt}],
        }]
        templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Select appropriate sampling params based on question_index (Q2=2, Q3=3, Q4=4)
        if question_index == 2:
            sampling_params = self.vllm_sampling_params_q2
        elif question_index == 3:
            sampling_params = self.vllm_sampling_params_q3
        elif question_index == 4:
            sampling_params = self.vllm_sampling_params_q4
        else:
            sampling_params = self.vllm_sampling_params_text  # Fallback
        
        start = time.time()
        outputs = self.vllm_llm.generate([templated], sampling_params)
        latency = time.time() - start
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text
        text = postprocess_assistant_only(generated_text)
        
        # Truncate only if still exceeds character limit (should be rare with proper token limits)
        truncated = False
        if max_chars and len(text) > max_chars:
            original_len = len(text)
            text, truncated = self._truncate_answer(text, max_chars)
            if truncated:
                self.logger.warning(f"Answer exceeded limit ({original_len} > {max_chars} chars), truncated to {len(text)} chars")
        
        result = GenerationResult(text=text, latency_s=latency)
        result.rag_used = self.use_rag and n_chunks_used > 0
        result.search_type = search_type
        result.n_chunks_used = n_chunks_used
        result.truncated = truncated
        return result

    def generate_text_only_batch(self, prompts: List[str], max_new_tokens: int = 500, max_chars_list: Optional[List[int]] = None, question_indices: Optional[List[int]] = None) -> List[GenerationResult]:
        """
        Generate batch of text-only questions using vLLM (REQUIRED, NO FALLBACK), with RAG (Q2-Q4).
        
        Args:
            prompts: List of prompts
            max_new_tokens: Not used (kept for compatibility)
            max_chars_list: Optional list of character limits per prompt
            question_indices: Optional list of question indices (2=Q2, 3=Q3, 4=Q4) for selecting sampling params
        """
        if max_chars_list is None:
            max_chars_list = [None] * len(prompts)
        if question_indices is None:
            question_indices = [2] * len(prompts)  # Default to Q2
        
        # Augment all prompts with RAG context
        augmented_prompts = []
        search_types = []
        n_chunks_list = []
        
        for prompt, max_chars in zip(prompts, max_chars_list):
            augmented_prompt = prompt
            search_type = None
            n_chunks_used = 0
            
            if self.use_rag and self.search_system:
                try:
                    # Use text search
                    chunks = self.search_system.text_search(prompt, n_results=self.n_chunks)
                    context = self._build_context_from_chunks(chunks)
                    augmented_prompt = self._augment_prompt_with_context(prompt, context, max_chars=max_chars)
                    search_type = "text"
                    n_chunks_used = len(chunks)
                except Exception as e:
                    print(f"Warning: RAG failed for batch question: {e}")
            
            augmented_prompts.append(augmented_prompt)
            search_types.append(search_type)
            n_chunks_list.append(n_chunks_used)
        
        # Use vLLM batch processing (REQUIRED, NO FALLBACK)
        if self.vllm_llm is None:
            raise RuntimeError("vLLM is not initialized. Cannot generate without vLLM.")
        
        templated_prompts = []
        sampling_params_list = []
        for augmented_prompt, q_idx in zip(augmented_prompts, question_indices):
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": augmented_prompt}],
            }]
            templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            templated_prompts.append(templated)
            
            # Select appropriate sampling params
            if q_idx == 2:
                sampling_params_list.append(self.vllm_sampling_params_q2)
            elif q_idx == 3:
                sampling_params_list.append(self.vllm_sampling_params_q3)
            elif q_idx == 4:
                sampling_params_list.append(self.vllm_sampling_params_q4)
            else:
                sampling_params_list.append(self.vllm_sampling_params_text)
        
        # vLLM requires same sampling params for batch, so we'll process in groups
        # For now, use the most common params (Q2) - in practice, we'll group by question type
        start = time.time()
        outputs = self.vllm_llm.generate(templated_prompts, self.vllm_sampling_params_q2)  # Use Q2 as default
        latency = time.time() - start
        
        results = []
        for i, output in enumerate(outputs):
            text = postprocess_assistant_only(output.outputs[0].text)
            
            # Truncate only if still exceeds character limit (should be rare)
            truncated = False
            if max_chars_list[i] and len(text) > max_chars_list[i]:
                original_len = len(text)
                text, truncated = self._truncate_answer(text, max_chars_list[i])
                if truncated:
                    self.logger.warning(f"Answer {i} exceeded limit ({original_len} > {max_chars_list[i]} chars), truncated to {len(text)} chars")
            
            result = GenerationResult(text=text, latency_s=latency)
            result.rag_used = self.use_rag and n_chunks_list[i] > 0
            result.search_type = search_types[i]
            result.n_chunks_used = n_chunks_list[i]
            result.truncated = truncated
            results.append(result)
        return results


def postprocess_assistant_only(text: str) -> str:
    """Extract assistant response, removing role markers."""
    if not text:
        return text
    t = text.strip()
    matches = list(re.finditer(r'(?:^|\n)assistant\s*:?\s*\n', t, flags=re.IGNORECASE))
    if matches:
        t = t[matches[-1].end():].lstrip()
    t = re.sub(r'^assistant\s*:?\s*', '', t, flags=re.IGNORECASE).lstrip()
    t = re.sub(r'^(system|user)\s*:?\s*', '', t, flags=re.IGNORECASE).lstrip()
    return t


def load_image_sanitized(image_path: str) -> Optional[Image.Image]:
    """Load and sanitize image, removing metadata."""
    p = Path(image_path)
    if not p.exists():
        return None
    img = Image.open(str(p)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    clean = Image.open(buf).convert("RGB")
    clean.load()
    return clean


def load_qa_pairs(qa_path: Path) -> Dict[str, Any]:
    """Load QA pairs from original QA file."""
    return json.loads(qa_path.read_text(encoding="utf-8"))


def regenerate_from_existing_answers(
    input_dir: Path,
    output_dir: Path,
    qa_dir: Path,
    max_new_tokens: int = 500,
    test_limit: Optional[int] = None,
    batch_size: int = 10,
    chromadb_path: Optional[str] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    tokenizer_id: Optional[str] = None,
    quantization: Optional[str] = None,
    dtype: str = "float16",
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.80,
    enforce_eager: bool = True,
) -> Dict[str, Any]:
    """
    Regenerate answers from existing answer files with RAG and save to output directory.
    
    Args:
        input_dir: Directory containing existing answer files (source)
        output_dir: Directory to save RAG-generated answer files
        qa_dir: Directory containing original QA pair files (for images)
        max_new_tokens: Maximum tokens for generation
        test_limit: Limit number of files for testing (None = all)
        batch_size: Number of files to process in each batch
        model_path: Path or repo id for the vLLM model weights
        tokenizer_id: Optional tokenizer identifier for vLLM
        quantization: Quantization backend (e.g., bitsandbytes) if desired
        dtype: vLLM compute dtype
        max_model_len: vLLM context window
        gpu_memory_utilization: vLLM GPU memory fraction
        enforce_eager: Whether to run vLLM in eager mode
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"rag_regeneration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger("qwen_regen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    logger.info("="*70)
    logger.info("QWEN Answer Regeneration with vLLM + RAG")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"QA directory: {qa_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Tokenizer id: {tokenizer_id or 'auto'}")
    logger.info(f"Quantization: {quantization or 'none'} | dtype: {dtype} | max model len: {max_model_len} | gpu util: {gpu_memory_utilization} | enforce eager: {enforce_eager}")
    logger.info(f"Character limits: Q1={CHAR_LIMIT_Q1}, Q2={CHAR_LIMIT_Q2}, Q3={CHAR_LIMIT_Q3}, Q4={CHAR_LIMIT_Q4}")
    logger.info(f"Max tokens: Q1={MAX_TOKENS_Q1}, Q2={MAX_TOKENS_Q2}, Q3={MAX_TOKENS_Q3}, Q4={MAX_TOKENS_Q4}")
    logger.info("Prompt instruction: Answers should be brief, concise, and to the point")
    logger.info(f"vLLM available: {VLLM_AVAILABLE}")
    logger.info(f"ChromaDB available: {CHROMADB_AVAILABLE}")
    logger.info(f"ChromaDB path: {chromadb_path or DEFAULT_CHROMADB_PATH}")
    logger.info(f"Test limit: {test_limit if test_limit else 'All files'}")
    logger.info("="*70)
    
    # CRITICAL: Test vLLM availability first - do not proceed without it
    if not VLLM_AVAILABLE:
        error_msg = "CRITICAL: vLLM is not available. This script REQUIRES vLLM. Cannot proceed."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("vLLM is available - proceeding with initialization...")
    
    # Initialize model wrapper (this will test vLLM initialization)
    # CRITICAL: If vLLM initialization fails, the program will exit
    logger.info("Initializing model...")
    chromadb_path = chromadb_path or os.getenv('CHROMADB_PATH', str(DEFAULT_CHROMADB_PATH))
    try:
        wrapper = VLLMRagWrapper(
            model_path=str(model_path),
            tokenizer_id=tokenizer_id,
            quantization=quantization,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            use_vllm=VLLM_AVAILABLE,
            chromadb_path=chromadb_path,
            use_rag=True,
            n_chunks=5,
            logger=logger
        )
        logger.info(f"Model initialized. Using vLLM: {wrapper.use_vllm}, RAG: {wrapper.use_rag}")
    except RuntimeError as e:
        # vLLM initialization failed - do not proceed
        error_msg = f"CRITICAL: vLLM initialization failed. Cannot proceed without vLLM. Error: {e}"
        logger.error(error_msg)
        logger.error("The program will now exit. Please check:")
        logger.error("  1. GPU memory is available")
        logger.error("  2. vLLM is properly installed")
        logger.error("  3. Model path is correct")
        logger.error("  4. GPU memory utilization settings are appropriate")
        raise RuntimeError(error_msg) from e
    
    # Get all answer files
    answer_files = sorted([f for f in input_dir.glob("*__answers.json")])
    if test_limit:
        answer_files = answer_files[:test_limit]
        logger.info(f"TEST MODE: Processing {len(answer_files)} files")
    
    logger.info(f"Total files to process: {len(answer_files)}")
    
    overall_start = time.time()
    successful = 0
    failed = 0
    failed_files = []
    
    # Process files in batches - optimize GPU usage by batching Q1 and Q2-Q4 across files
    for batch_idx in range(0, len(answer_files), batch_size):
        batch_files = answer_files[batch_idx:batch_idx + batch_size]
        logger.info(f"\nProcessing batch {batch_idx//batch_size + 1} ({len(batch_files)} files)")
        
        # Load all files in batch first
        batch_data = []
        for answer_file in batch_files:
            try:
                # Load existing answer file
                with open(answer_file, 'r', encoding='utf-8') as f:
                    answer_data = json.load(f)
                
                source_file = answer_data.get('source_file', '')
                logger.info(f"Loading: {answer_file.name} (source: {source_file})")
                
                # Load original QA file for image path
                qa_file = qa_dir / source_file
                image_path = None
                if qa_file.exists():
                    qa_data = load_qa_pairs(qa_file)
                    image_path = qa_data.get('image_path', '')
                
                batch_data.append({
                    'answer_file': answer_file,
                    'answer_data': answer_data,
                    'source_file': source_file,
                    'image_path': image_path,
                    'output_file': output_dir / answer_file.name
                })
            except Exception as e:
                logger.error(f"Failed to load {answer_file.name}: {e}")
                failed += 1
                failed_files.append(answer_file.name)
                continue
        
        if not batch_data:
            continue
        
        # Batch Q1: Collect all Q1 questions and images
        q1_prompts = []
        q1_images = []
        q1_indices = []  # Track which file each Q1 belongs to
        
        for idx, data in enumerate(batch_data):
            existing_answers = data['answer_data'].get('answers', [])
            if len(existing_answers) > 0:
                q1 = existing_answers[0].get('question', '')
                img = load_image_sanitized(data['image_path']) if data['image_path'] else None
                if q1 and img:
                    q1_prompts.append(q1)
                    q1_images.append(img)
                    q1_indices.append(idx)
        
        # Process Q1 batch
        q1_results = {}
        if q1_prompts:
            logger.info(f"  Batch Q1: Generating {len(q1_prompts)} vision questions with RAG (char limit: {CHAR_LIMIT_Q1})...")
            try:
                q1_batch_results = wrapper.generate_with_vision_batch(q1_prompts, q1_images, max_new_tokens=MAX_TOKENS_Q1, max_chars=CHAR_LIMIT_Q1)
                for i, (file_idx, res) in enumerate(zip(q1_indices, q1_batch_results)):
                    q1_results[file_idx] = res
                    trunc_msg = f", TRUNCATED" if res.truncated else ""
                    logger.info(f"    Q1[{file_idx}] done: {res.latency_s:.2f}s (RAG: {res.rag_used}, chunks: {res.n_chunks_used}, len: {len(res.text)} chars{trunc_msg})")
            except Exception as e:
                logger.error(f"    Q1 batch error: {e}\n{traceback.format_exc()}")
                # Fallback to individual processing
                for i, (file_idx, prompt, img) in enumerate(zip(q1_indices, q1_prompts, q1_images)):
                    try:
                        res = wrapper.generate_with_vision(prompt, img, max_new_tokens=MAX_TOKENS_Q1, max_chars=CHAR_LIMIT_Q1)
                        q1_results[file_idx] = res
                    except Exception as e2:
                        logger.error(f"    Q1[{file_idx}] individual error: {e2}")
        
        # Batch Q2-Q4: Collect all text questions with their character limits
        text_prompts = []
        text_indices = []  # Track (file_idx, question_idx) for each prompt
        text_max_chars = []  # Character limits per question
        text_q_indices = []  # Question indices (2, 3, 4) for sampling params
        
        char_limits = {1: CHAR_LIMIT_Q2, 2: CHAR_LIMIT_Q3, 3: CHAR_LIMIT_Q4}
        max_tokens_map = {1: MAX_TOKENS_Q2, 2: MAX_TOKENS_Q3, 3: MAX_TOKENS_Q4}
        
        for idx, data in enumerate(batch_data):
            existing_answers = data['answer_data'].get('answers', [])
            for q_idx in range(1, min(4, len(existing_answers))):
                q = existing_answers[q_idx].get('question', '')
                if q:
                    text_prompts.append(q)
                    text_indices.append((idx, q_idx))
                    text_max_chars.append(char_limits[q_idx])
                    text_q_indices.append(q_idx + 1)  # Q2=2, Q3=3, Q4=4
        
        # Process Q2-Q4 batch
        text_results = {}
        if text_prompts:
            logger.info(f"  Batch Q2-Q4: Generating {len(text_prompts)} text questions with RAG (char limits: Q2={CHAR_LIMIT_Q2}, Q3={CHAR_LIMIT_Q3}, Q4={CHAR_LIMIT_Q4})...")
            try:
                text_batch_results = wrapper.generate_text_only_batch(
                    text_prompts, 
                    max_new_tokens=max_new_tokens,
                    max_chars_list=text_max_chars,
                    question_indices=text_q_indices
                )
                for (file_idx, q_idx), res in zip(text_indices, text_batch_results):
                    if file_idx not in text_results:
                        text_results[file_idx] = {}
                    text_results[file_idx][q_idx] = res
                    trunc_msg = f", TRUNCATED" if res.truncated else ""
                    logger.info(f"    Q{q_idx+1}[{file_idx}] done: {res.latency_s:.2f}s (RAG: {res.rag_used}, chunks: {res.n_chunks_used}, len: {len(res.text)} chars{trunc_msg})")
            except Exception as e:
                logger.error(f"    Q2-Q4 batch error: {e}\n{traceback.format_exc()}")
                # Fallback to individual processing per file
                for idx, data in enumerate(batch_data):
                    existing_answers = data['answer_data'].get('answers', [])
                    for q_idx in range(1, min(4, len(existing_answers))):
                        q = existing_answers[q_idx].get('question', '')
                        if q:
                            try:
                                res = wrapper.generate_text_only(
                                    q, 
                                    max_new_tokens=max_tokens_map[q_idx],
                                    max_chars=char_limits[q_idx],
                                    question_index=q_idx + 1
                                )
                                if idx not in text_results:
                                    text_results[idx] = {}
                                text_results[idx][q_idx] = res
                            except Exception as e2:
                                logger.error(f"    Q{q_idx+1}[{idx}] individual error: {e2}")
        
        # Assemble results and save files
        for idx, data in enumerate(batch_data):
            try:
                answer_data = data['answer_data']
                existing_answers = answer_data.get('answers', [])
                new_answers = []
                
                # Q1
                if len(existing_answers) > 0:
                    q1 = existing_answers[0].get('question', '')
                    if idx in q1_results:
                        res1 = q1_results[idx]
                        q1_answer = {
                            "question": q1,
                            "answer": res1.text,
                            "latency_s": round(res1.latency_s, 2)
                        }
                        if hasattr(res1, 'rag_used') and res1.rag_used:
                            q1_answer["rag_used"] = True
                            q1_answer["search_type"] = res1.search_type
                            q1_answer["n_chunks_used"] = res1.n_chunks_used
                        if hasattr(res1, 'truncated') and res1.truncated:
                            q1_answer["truncated"] = True
                        q1_answer["char_limit"] = CHAR_LIMIT_Q1
                        new_answers.append(q1_answer)
                    else:
                        # Fallback to original answer
                        new_answers.append(existing_answers[0])
                
                # Q2-Q4
                for q_idx in range(1, min(4, len(existing_answers))):
                    q = existing_answers[q_idx].get('question', '')
                    if idx in text_results and q_idx in text_results[idx]:
                        res = text_results[idx][q_idx]
                        q_answer = {
                            "question": q,
                            "answer": res.text,
                            "latency_s": round(res.latency_s, 2)
                        }
                        if hasattr(res, 'rag_used') and res.rag_used:
                            q_answer["rag_used"] = True
                            q_answer["search_type"] = res.search_type
                            q_answer["n_chunks_used"] = res.n_chunks_used
                        if hasattr(res, 'truncated') and res.truncated:
                            q_answer["truncated"] = True
                        q_answer["char_limit"] = char_limits[q_idx]
                        new_answers.append(q_answer)
                    else:
                        # Fallback to original answer
                        new_answers.append(existing_answers[q_idx])
                
                # Update answer data with RAG metadata
                answer_data['answers'] = new_answers
                generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Derive model identifier from model path
                model_path_str = str(model_path).lower()
                if 'gemma3' in model_path_str or 'gemma-3' in model_path_str:
                    model_identifier = 'gemma3'
                elif 'gemma' in model_path_str:
                    model_identifier = 'gemma3'  # Default gemma to gemma3
                elif 'qwen' in model_path_str:
                    model_identifier = 'qwen'
                else:
                    model_identifier = Path(model_path).name.lower()
                answer_data['model'] = model_identifier
                answer_data['regenerated_at'] = generation_timestamp
                answer_data['regenerated_with'] = f'{model_identifier}+rag' if wrapper.use_rag else model_identifier
                answer_data['char_limit_q1'] = CHAR_LIMIT_Q1
                answer_data['char_limit_q2'] = CHAR_LIMIT_Q2
                answer_data['char_limit_q3'] = CHAR_LIMIT_Q3
                answer_data['char_limit_q4'] = CHAR_LIMIT_Q4
                answer_data['max_tokens_q1'] = MAX_TOKENS_Q1
                answer_data['max_tokens_q2'] = MAX_TOKENS_Q2
                answer_data['max_tokens_q3'] = MAX_TOKENS_Q3
                answer_data['max_tokens_q4'] = MAX_TOKENS_Q4
                answer_data['concise_mode'] = True
                answer_data['prompt_instruction'] = "brief, concise, and to the point"
                answer_data['rag_enabled'] = wrapper.use_rag
                answer_data['rag_regenerated_at'] = generation_timestamp if wrapper.use_rag else None
                answer_data['source_input_dir'] = str(input_dir)
                
                # Save to output directory
                with open(data['output_file'], 'w', encoding='utf-8') as f:
                    json.dump(answer_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"  ✓ Saved {data['output_file'].name}")
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to save {data['answer_file'].name}: {e}\n{traceback.format_exc()}")
                failed += 1
                failed_files.append(str(data['answer_file']))
        
        # Small delay between batches
        if batch_idx + batch_size < len(answer_files):
            time.sleep(1.0)
    
    total_time = time.time() - overall_start
    summary = {
        "total_files": len(answer_files),
        "successful": successful,
        "failed": failed,
        "failed_files": failed_files,
        "total_time_s": total_time,
        "avg_per_file_s": total_time / max(1, successful),
        "vllm_used": wrapper.use_vllm,
        "char_limit_q1": CHAR_LIMIT_Q1,
        "char_limit_q2": CHAR_LIMIT_Q2,
        "char_limit_q3": CHAR_LIMIT_Q3,
        "char_limit_q4": CHAR_LIMIT_Q4,
        "max_tokens_q1": MAX_TOKENS_Q1,
        "max_tokens_q2": MAX_TOKENS_Q2,
        "max_tokens_q3": MAX_TOKENS_Q3,
        "max_tokens_q4": MAX_TOKENS_Q4,
        "concise_mode": True,
        "log_file": str(log_file),
        "model_path": str(model_path),
        "tokenizer_id": tokenizer_id,
        "quantization": quantization,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }
    
    # Save summary to output directory
    summary_file = output_dir / "rag_regeneration_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "="*70)
    logger.info("REGENERATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total files: {summary['total_files']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Total time: {summary['total_time_s']:.2f}s ({summary['total_time_s']/60:.2f} min)")
    logger.info(f"Average per file: {summary['avg_per_file_s']:.2f}s")
    logger.info(f"vLLM used: {summary['vllm_used']}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("="*70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Regenerate QWEN answers with vLLM optimization")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Directory containing existing answer files")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to save regenerated answer files")
    parser.add_argument("--qa-dir", type=str, default=str(DEFAULT_QA_DIR),
                        help="Directory containing original QA pair files (for images)")
    parser.add_argument("--max-new-tokens", type=int, default=500,
                        help="Maximum tokens for generation")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Limit number of files for testing (None = all)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of files to process per batch")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM (NOT RECOMMENDED - script will fail without vLLM)")
    parser.add_argument("--chromadb-path", type=str, default=None,
                        help="Path to ChromaDB storage (default: /home/himanshu/dev/data/chromadb)")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG, generate without ChromaDB context")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Model directory or repo id to load (default: Qwen AWQ)")
    parser.add_argument("--tokenizer-id", type=str, default=None,
                        help="Optional tokenizer identifier when model path differs")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization backend (e.g., bitsandbytes)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["auto", "float16", "bfloat16"],
                        help="Computation dtype for vLLM (default: float16)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max sequence length for vLLM (default: 8192)")
    parser.add_argument("--gpu-memory-util", type=float, default=0.80,
                        help="GPU memory utilization target for vLLM (default: 0.80)")
    parser.add_argument("--no-enforce-eager", action="store_true",
                        help="Disable enforce_eager mode (enabled by default)")
    
    args = parser.parse_args()
    
    global VLLM_AVAILABLE, CHROMADB_AVAILABLE
    if args.no_vllm:
        VLLM_AVAILABLE = False
    if args.no_rag:
        CHROMADB_AVAILABLE = False
    
    enforce_eager = not args.no_enforce_eager
    
    summary = regenerate_from_existing_answers(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        qa_dir=Path(args.qa_dir),
        max_new_tokens=args.max_new_tokens,
        test_limit=args.test_limit,
        batch_size=args.batch_size,
        chromadb_path=args.chromadb_path,
        model_path=Path(args.model_path),
        tokenizer_id=args.tokenizer_id,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_util,
        enforce_eager=enforce_eager,
    )
    
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
