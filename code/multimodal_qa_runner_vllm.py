#!/usr/bin/env python3
"""
Optimized multimodal QA runner using ONLY vLLM for all questions:
- Uses vLLM for ALL questions (Q1 with images via multi_modal_data, Q2-Q4 text-only)
- Passes images separately using vLLM's multimodal API
- Reads questions from existing answer files in output/qwen_regenerated
- Regenerates all 4 answers with max_tokens=500
- Updates existing answer files in place
- Supports background execution with comprehensive logging
"""
import argparse
import json
import io
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
import re
import logging
from datetime import datetime
import traceback

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("WARNING: vLLM not available. Will use Transformers for all questions.")


DEFAULT_INPUT_DIR = Path("/home/himanshu/dev/output/qwen_regenerated")
DEFAULT_QA_DIR = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components")
DEFAULT_MODEL_PATH = Path("/home/himanshu/dev/models/QWEN_AWQ")


@dataclass
class GenerationResult:
    text: str
    latency_s: float


class QwenVLLMWrapper:
    """Wrapper that uses ONLY vLLM for all questions (text and vision)."""
    
    def __init__(self, model_path: str = str(DEFAULT_MODEL_PATH), use_vllm: bool = True):
        self.model_path = model_path
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        # Initialize processor (needed for chat template formatting)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize vLLM for ALL questions (text and vision)
        self.vllm_llm = None
        if self.use_vllm:
            try:
                print("Initializing vLLM for ALL questions (text and vision)...")
                # Use higher GPU memory since we're only loading one model
                self.vllm_llm = LLM(
                    model=str(model_path),
                    tensor_parallel_size=1,
                    dtype="float16",
                    trust_remote_code=True,
                    max_model_len=8192,  # Increased for image tokens
                    gpu_memory_utilization=0.85,  # Use more GPU since only one model
                )
                self.vllm_sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=500,
                    stop=None,
                )
                print("vLLM initialized successfully for all question types")
            except Exception as e:
                print(f"WARNING: Failed to initialize vLLM: {e}")
                print("Falling back to Transformers for all questions")
                self.use_vllm = False
                self.vllm_llm = None
        
        # Fallback: Initialize Transformers only if vLLM failed
        self.transformers_model = None
        if not self.use_vllm:
            from transformers import AutoModelForImageTextToText
            # Use cuda:0 explicitly to avoid CPU/disk device_map issues with AWQ
            self.transformers_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": "cuda:0"},  # Explicitly use GPU, avoid CPU/disk
                trust_remote_code=True,
            )

    def generate_with_vision(self, prompt: str, image: Image.Image, max_new_tokens: int = 500) -> GenerationResult:
        """Generate with image using vLLM with multimodal data."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        if self.use_vllm and self.vllm_llm is not None:
            # Use vLLM with multimodal data - pass image separately
            # According to vLLM docs, we pass a dict with prompt and multi_modal_data
            start = time.time()
            try:
                # vLLM expects: {"prompt": str, "multi_modal_data": {"image": [image]}}
                multimodal_prompt = {
                    "prompt": templated,
                    "multi_modal_data": {"image": [image]}
                }
                outputs = self.vllm_llm.generate([multimodal_prompt], self.vllm_sampling_params)
                latency = time.time() - start
                
                generated_text = outputs[0].outputs[0].text
                text = postprocess_assistant_only(generated_text)
                return GenerationResult(text=text, latency_s=latency)
            except Exception as e:
                # If vLLM fails with vision, fall back to Transformers
                print(f"vLLM vision generation failed: {e}, falling back to Transformers")
                if self.transformers_model is None:
                    from transformers import AutoModelForImageTextToText
                    # Use cuda:0 explicitly to avoid CPU/disk device_map issues with AWQ
                    self.transformers_model = AutoModelForImageTextToText.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map={"": "cuda:0"},  # Explicitly use GPU, avoid CPU/disk
                        trust_remote_code=True,
                    )
        else:
            # Fallback to Transformers
            if self.transformers_model is None:
                from transformers import AutoModelForImageTextToText
                # Use cuda:0 explicitly to avoid CPU/disk device_map issues with AWQ
                self.transformers_model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map={"": "cuda:0"},  # Explicitly use GPU, avoid CPU/disk
                    trust_remote_code=True,
                )
        
        # Use Transformers fallback
        inputs = self.processor(text=[templated], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.transformers_model.device) for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            outputs = self.transformers_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        latency = time.time() - start
        
        input_len = inputs.get("input_ids").shape[1]
        gen_ids = outputs[0][input_len:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True)
        text = postprocess_assistant_only(text)
        return GenerationResult(text=text, latency_s=latency)

    def generate_text_only(self, prompt: str, max_new_tokens: int = 500) -> GenerationResult:
        """Generate text-only using vLLM if available, else Transformers."""
        if self.use_vllm and self.vllm_llm is not None:
            # Use vLLM for faster inference
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }]
            templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            start = time.time()
            outputs = self.vllm_llm.generate([templated], self.vllm_sampling_params)
            latency = time.time() - start
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            text = postprocess_assistant_only(generated_text)
            return GenerationResult(text=text, latency_s=latency)
        else:
            # Fallback to Transformers
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }]
            templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=[templated], return_tensors="pt")
            inputs = {k: v.to(self.transformers_model.device) for k, v in inputs.items()}
            
            start = time.time()
            with torch.no_grad():
                outputs = self.transformers_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            latency = time.time() - start
            
            input_len = inputs.get("input_ids").shape[1]
            gen_ids = outputs[0][input_len:]
            text = self.processor.decode(gen_ids, skip_special_tokens=True)
            text = postprocess_assistant_only(text)
            return GenerationResult(text=text, latency_s=latency)

    def generate_text_only_batch(self, prompts: List[str], max_new_tokens: int = 500) -> List[GenerationResult]:
        """Generate batch of text-only questions using vLLM if available."""
        if self.use_vllm and self.vllm_llm is not None:
            # Use vLLM batch processing
            templated_prompts = []
            for prompt in prompts:
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }]
                templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                templated_prompts.append(templated)
            
            start = time.time()
            outputs = self.vllm_llm.generate(templated_prompts, self.vllm_sampling_params)
            latency = time.time() - start
            
            results = []
            for output in outputs:
                text = postprocess_assistant_only(output.outputs[0].text)
                results.append(GenerationResult(text=text, latency_s=latency))
            return results
        else:
            # Fallback to Transformers batch
            messages_list = []
            for prompt in prompts:
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }]
                templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                messages_list.append(templated)
            
            proc = self.processor(text=messages_list, padding=True, return_tensors="pt")
            proc = {k: v.to(self.transformers_model.device) for k, v in proc.items()}
            
            start = time.time()
            with torch.no_grad():
                outputs = self.transformers_model.generate(
                    **proc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            latency = time.time() - start
            
            input_ids = proc.get("input_ids")
            attn = proc.get("attention_mask")
            if attn is not None:
                input_lens = attn.sum(dim=1).tolist()
            else:
                input_lens = [input_ids.shape[1]] * input_ids.shape[0]
            
            results = []
            for i in range(outputs.shape[0]):
                gen_ids = outputs[i][int(input_lens[i]):]
                text = self.processor.decode(gen_ids, skip_special_tokens=True)
                text = postprocess_assistant_only(text)
                results.append(GenerationResult(text=text, latency_s=latency))
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
    qa_dir: Path,
    max_new_tokens: int = 500,
    test_limit: Optional[int] = None,
    batch_size: int = 10
) -> Dict[str, Any]:
    """
    Regenerate answers from existing answer files.
    
    Args:
        input_dir: Directory containing existing answer files
        qa_dir: Directory containing original QA pair files (for images)
        max_new_tokens: Maximum tokens for generation
        test_limit: Limit number of files for testing (None = all)
        batch_size: Number of files to process in each batch
    """
    # Setup logging
    logs_dir = input_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"regeneration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    logger.info("QWEN Answer Regeneration with vLLM Optimization")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"QA directory: {qa_dir}")
    logger.info(f"Max tokens: {max_new_tokens}")
    logger.info(f"vLLM available: {VLLM_AVAILABLE}")
    logger.info(f"Test limit: {test_limit if test_limit else 'All files'}")
    logger.info("="*70)
    
    # Initialize model wrapper
    logger.info("Initializing model...")
    wrapper = QwenVLLMWrapper(use_vllm=VLLM_AVAILABLE)
    logger.info(f"Model initialized. Using vLLM for ALL questions: {wrapper.use_vllm}")
    
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
    
    # Process files in batches
    for batch_idx in range(0, len(answer_files), batch_size):
        batch_files = answer_files[batch_idx:batch_idx + batch_size]
        logger.info(f"\nProcessing batch {batch_idx//batch_size + 1} ({len(batch_files)} files)")
        
        for answer_file in batch_files:
            try:
                # Load existing answer file
                with open(answer_file, 'r', encoding='utf-8') as f:
                    answer_data = json.load(f)
                
                source_file = answer_data.get('source_file', '')
                logger.info(f"Processing: {answer_file.name} (source: {source_file})")
                
                # Get questions from existing answers
                existing_answers = answer_data.get('answers', [])
                if len(existing_answers) < 4:
                    logger.warning(f"  Only {len(existing_answers)} answers found, expected 4")
                
                # Load original QA file for image path
                qa_file = qa_dir / source_file
                image_path = None
                if qa_file.exists():
                    qa_data = load_qa_pairs(qa_file)
                    image_path = qa_data.get('image_path', '')
                
                # Regenerate answers
                new_answers = []
                
                # Q1: With image (use vLLM with multimodal data)
                if len(existing_answers) > 0:
                    q1 = existing_answers[0].get('question', '')
                    logger.info(f"  Q1: Generating with image using vLLM...")
                    img = load_image_sanitized(image_path) if image_path else None
                    try:
                        res1 = wrapper.generate_with_vision(q1, img, max_new_tokens=max_new_tokens)
                        new_answers.append({
                            "question": q1,
                            "answer": res1.text,
                            "latency_s": round(res1.latency_s, 2)
                        })
                        logger.info(f"    Q1 done: {res1.latency_s:.2f}s")
                    except Exception as e:
                        logger.error(f"    Q1 error: {e}\n{traceback.format_exc()}")
                        new_answers.append({
                            "question": q1,
                            "answer": existing_answers[0].get('answer', ''),
                            "latency_s": existing_answers[0].get('latency_s', 0.0),
                            "error": str(e)
                        })
                
                # Q2-Q4: Text-only (use vLLM)
                text_questions = []
                for i in range(1, min(4, len(existing_answers))):
                    text_questions.append(existing_answers[i].get('question', ''))
                
                if text_questions:
                    logger.info(f"  Q2-Q4: Generating {len(text_questions)} text-only questions...")
                    try:
                        # Use batch processing for text-only questions
                        text_results = wrapper.generate_text_only_batch(text_questions, max_new_tokens=max_new_tokens)
                        for i, res in enumerate(text_results):
                            q_idx = i + 1
                            if q_idx < len(existing_answers):
                                new_answers.append({
                                    "question": text_questions[i],
                                    "answer": res.text,
                                    "latency_s": round(res.latency_s, 2)
                                })
                                logger.info(f"    Q{q_idx+1} done: {res.latency_s:.2f}s")
                    except Exception as e:
                        logger.error(f"    Q2-Q4 batch error: {e}\n{traceback.format_exc()}")
                        # Fallback to individual generation
                        for i, q in enumerate(text_questions):
                            q_idx = i + 1
                            try:
                                res = wrapper.generate_text_only(q, max_new_tokens=max_new_tokens)
                                new_answers.append({
                                    "question": q,
                                    "answer": res.text,
                                    "latency_s": round(res.latency_s, 2)
                                })
                            except Exception as e2:
                                logger.error(f"    Q{q_idx+1} individual error: {e2}")
                                new_answers.append({
                                    "question": q,
                                    "answer": existing_answers[q_idx].get('answer', ''),
                                    "latency_s": existing_answers[q_idx].get('latency_s', 0.0),
                                    "error": str(e2)
                                })
                
                # Update answer file
                answer_data['answers'] = new_answers
                answer_data['regenerated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                answer_data['regenerated_with'] = 'vllm' if wrapper.use_vllm else 'transformers'
                answer_data['max_tokens'] = max_new_tokens
                
                with open(answer_file, 'w', encoding='utf-8') as f:
                    json.dump(answer_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"  ✓ Updated {answer_file.name}")
                successful += 1
                
            except Exception as e:
                logger.error(f"  ✗ Failed {answer_file.name}: {e}\n{traceback.format_exc()}")
                failed += 1
                failed_files.append(answer_file.name)
        
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
        "max_tokens": max_new_tokens,
        "log_file": str(log_file),
    }
    
    # Save summary
    summary_file = input_dir / "regeneration_summary.json"
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
    parser.add_argument("--qa-dir", type=str, default=str(DEFAULT_QA_DIR),
                        help="Directory containing original QA pair files (for images)")
    parser.add_argument("--max-new-tokens", type=int, default=500,
                        help="Maximum tokens for generation")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Limit number of files for testing (None = all)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of files to process per batch")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM, use Transformers only")
    
    args = parser.parse_args()
    
    global VLLM_AVAILABLE
    if args.no_vllm:
        VLLM_AVAILABLE = False
    
    summary = regenerate_from_existing_answers(
        input_dir=Path(args.input_dir),
        qa_dir=Path(args.qa_dir),
        max_new_tokens=args.max_new_tokens,
        test_limit=args.test_limit,
        batch_size=args.batch_size,
    )
    
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
