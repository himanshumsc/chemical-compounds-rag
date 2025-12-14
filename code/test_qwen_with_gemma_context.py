#!/usr/bin/env python3
"""
Test Qwen-VL with the exact same context/prompt/settings used with Gemma-3.
This helps determine if failures are due to model behavior or retrieval issues.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

# Fix ChromaDB SQLite compatibility
import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sqlite3

# Import vLLM wrapper
try:
    from multimodal_qa_runner_vllm import VLLMRagWrapper, DEFAULT_MODEL_PATH, DEFAULT_QA_DIR
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("ERROR: multimodal_qa_runner_vllm not available")

from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Character limits (same as Gemma-3)
CHAR_LIMIT_Q1 = 600
CHAR_LIMIT_Q2 = 1000
CHAR_LIMIT_Q3 = 1800
CHAR_LIMIT_Q4 = 2000


@dataclass
class TestCase:
    """Represents a single test case from Gemma-3 filtered answer."""
    file: str
    question_idx: int
    question: str
    gemma_answer: str
    context: str
    char_limit: int
    rag_chunks: List[Dict]  # Gemma's retrieved chunks
    image_path: Optional[str] = None
    source_file: Optional[str] = None


def extract_image_path_from_chunks(rag_chunks: List[Dict], qa_dir: Path) -> Optional[str]:
    """
    Extract image path from rag_chunks metadata.
    Falls back to finding image in original QA file.
    """
    # First, try to find image_path in chunks metadata
    for chunk in rag_chunks:
        metadata = chunk.get('metadata', {})
        image_path = metadata.get('image_path', '')
        if image_path and Path(image_path).exists():
            return image_path
    
    # If not found, return None (will be handled by caller)
    return None


def find_original_qa_file(source_file: str, qa_dir: Path) -> Optional[Path]:
    """Find the original QA file based on source_file name."""
    if not source_file:
        return None
    
    # Try exact match
    qa_path = qa_dir / source_file
    if qa_path.exists():
        return qa_path
    
    # Try without extension
    qa_path = qa_dir / f"{source_file}.json"
    if qa_path.exists():
        return qa_path
    
    return None


def load_test_cases(
    input_dir: Path,
    qa_dir: Path,
    test_limit: Optional[int] = None
) -> List[TestCase]:
    """
    Load test cases from Gemma-3 filtered answers.
    """
    json_files = sorted(input_dir.glob("*.json"))
    
    if test_limit:
        json_files = json_files[:test_limit]
    
    test_cases = []
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            source_file = data.get('source_file', '')
            answers = data.get('answers', [])
            
            for idx, answer in enumerate(answers):
                # Only process filtered answers
                if not answer.get('filtered_as_missing_info', False):
                    continue
                
                question_idx = answer.get('question_idx', idx + 1)
                
                # Skip Q1 (image-based questions) - Gemma handled these successfully
                if question_idx == 1:
                    logger.info(f"  Skipping Q1 (image-based) for {json_path.name}")
                    continue
                question = answer.get('question', '')
                gemma_answer = answer.get('answer', '')
                context = answer.get('rag_context_formatted', '')
                char_limit = answer.get('char_limit', None)
                rag_chunks = answer.get('rag_chunks', [])  # Gemma's retrieved chunks
                
                # Extract image path for Q1
                image_path = None
                if question_idx == 1:
                    # Try to get image from chunks
                    image_path = extract_image_path_from_chunks(rag_chunks, qa_dir)
                    
                    # If not found, try original QA file
                    if not image_path:
                        qa_file = find_original_qa_file(source_file, qa_dir)
                        if qa_file:
                            try:
                                with open(qa_file, 'r') as f:
                                    qa_data = json.load(f)
                                    q1_data = qa_data.get('questions', [{}])[0] if qa_data.get('questions') else {}
                                    image_path = q1_data.get('image_path', '')
                                    if image_path and not Path(image_path).exists():
                                        image_path = None
                            except Exception as e:
                                logger.warning(f"Could not load QA file {qa_file}: {e}")
                
                test_case = TestCase(
                    file=json_path.name,
                    question_idx=question_idx,
                    question=question,
                    gemma_answer=gemma_answer,
                    context=context,
                    char_limit=char_limit or (CHAR_LIMIT_Q1 if question_idx == 1 else 
                                             CHAR_LIMIT_Q2 if question_idx == 2 else
                                             CHAR_LIMIT_Q3 if question_idx == 3 else
                                             CHAR_LIMIT_Q4),
                    rag_chunks=rag_chunks,  # Gemma's retrieved chunks
                    image_path=image_path,
                    source_file=source_file
                )
                test_cases.append(test_case)
                
        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return test_cases


def check_answer_quality(answer: str) -> bool:
    """
    Check if answer indicates missing information.
    Returns True if answer seems valid (not missing info), False if missing info.
    """
    if not answer:
        return False
    
    answer_lower = answer.lower()
    
    # Patterns indicating missing information
    missing_patterns = [
        "does not contain",
        "not available",
        "not found",
        "cannot answer",
        "unable to",
        "no information",
        "i am sorry",
        "i cannot",
        "i don't have",
        "i do not have",
        "information is not",
        "provided text does not",
        "provided documents do not"
    ]
    
    for pattern in missing_patterns:
        if pattern in answer_lower:
            return False
    
    # If answer is very short and vague, might be missing info
    if len(answer.strip()) < 20:
        return False
    
    return True


def generate_with_qwen(
    wrapper: VLLMRagWrapper,
    test_case: TestCase,
    gemma_chunks: List[Dict]
) -> Dict[str, Any]:
    """
    Generate answer with Qwen using the EXACT SAME rag_context_formatted from Gemma-3.
    We inject the pre-formatted context into Qwen's RAG prompt augmentation,
    ensuring Qwen uses the same chunks but still goes through RAG processing.
    """
    result = {
        "question": test_case.question,
        "gemma_answer": test_case.gemma_answer,
        "qwen_answer": None,
        "qwen_latency": None,
        "qwen_succeeded": False,
        "qwen_context_used": test_case.context,  # Same context as Gemma
        "chunks_match": True,  # We're using the same chunks
        "error": None
    }
    
    try:
        # Build prompt with the EXACT same context that Gemma-3 used
        # Use wrapper's prompt augmentation method to ensure proper RAG formatting
        augmented_prompt = wrapper._augment_prompt_with_context(
            question=test_case.question,
            context=test_case.context,  # Use Gemma's rag_context_formatted
            max_chars=test_case.char_limit
        )
        
        if test_case.question_idx == 1 and test_case.image_path:
            # Q1: Multimodal with image
            try:
                image = Image.open(test_case.image_path)
                
                # Format messages for multimodal
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": augmented_prompt},
                    ],
                }]
                templated = wrapper.processor.apply_chat_template(messages, add_generation_prompt=True)
                
                # Use vLLM with multimodal data
                multimodal_prompt = {
                    "prompt": templated,
                    "multi_modal_data": {"image": [image]}
                }
                
                # Select sampling params for Q1
                sampling_params = wrapper.vllm_sampling_params_q1
                
                start = time.time()
                outputs = wrapper.vllm_llm.generate([multimodal_prompt], sampling_params)
                latency = time.time() - start
                
                # Extract generated text
                generated_text = outputs[0].outputs[0].text
                from multimodal_qa_runner_vllm import postprocess_assistant_only
                text = postprocess_assistant_only(generated_text)
                
                # Truncate if needed
                if len(text) > test_case.char_limit:
                    text, _ = wrapper._truncate_answer(text, test_case.char_limit)
                
                result["qwen_answer"] = text
                result["qwen_latency"] = latency
                result["qwen_succeeded"] = check_answer_quality(text)
                
            except Exception as e:
                result["error"] = f"Image generation failed: {e}"
                logger.error(f"Error generating Q1 for {test_case.file}: {e}")
        else:
            # Q2-Q4: Text-only
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": augmented_prompt}],
            }]
            templated = wrapper.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Select appropriate sampling params
            if test_case.question_idx == 2:
                sampling_params = wrapper.vllm_sampling_params_q2
            elif test_case.question_idx == 3:
                sampling_params = wrapper.vllm_sampling_params_q3
            elif test_case.question_idx == 4:
                sampling_params = wrapper.vllm_sampling_params_q4
            else:
                sampling_params = wrapper.vllm_sampling_params_text
            
            start = time.time()
            outputs = wrapper.vllm_llm.generate([templated], sampling_params)
            latency = time.time() - start
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            from multimodal_qa_runner_vllm import postprocess_assistant_only
            text = postprocess_assistant_only(generated_text)
            
            # Truncate if needed
            if len(text) > test_case.char_limit:
                text, _ = wrapper._truncate_answer(text, test_case.char_limit)
            
            result["qwen_answer"] = text
            result["qwen_latency"] = latency
            result["qwen_succeeded"] = check_answer_quality(text)
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error generating answer for {test_case.file} Q{test_case.question_idx}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return result


def classify_result(gemma_answer: str, qwen_succeeded: bool, chunks_match: bool = True) -> str:
    """
    Classify the result based on Gemma and Qwen outcomes.
    Since we're using the EXACT same rag_context_formatted, chunks_match is always True.
    """
    gemma_failed = not check_answer_quality(gemma_answer)
    
    if gemma_failed and qwen_succeeded:
        # Same context, Gemma failed, Qwen succeeded → Model issue confirmed
        return "MODEL_FAILURE_CONFIRMED"
    elif gemma_failed and not qwen_succeeded:
        # Same context, both failed → Retrieval issue (wrong chunks retrieved originally)
        return "RETRIEVAL_FAILURE_CONFIRMED"
    elif not gemma_failed and qwen_succeeded:
        return "BOTH_SUCCEEDED"  # Edge case (shouldn't happen for filtered answers)
    else:
        return "UNKNOWN"  # Both succeeded but Gemma was filtered? Edge case


def process_test_cases(
    test_cases: List[TestCase],
    wrapper: VLLMRagWrapper,
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Process test cases in batches.
    """
    results = []
    total = len(test_cases)
    
    for i in range(0, total, batch_size):
        batch = test_cases[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} cases)...")
        
        for j, test_case in enumerate(batch):
            logger.info(f"  [{i+j+1}/{total}] {test_case.file} Q{test_case.question_idx}")
            
            result = generate_with_qwen(wrapper, test_case, test_case.rag_chunks)
            
            # Add metadata
            result["file"] = test_case.file
            result["question_idx"] = test_case.question_idx
            result["source_file"] = test_case.source_file
            result["context_length"] = len(test_case.context)
            result["char_limit"] = test_case.char_limit
            
            # Classify (chunks_match is always True since we use same context)
            result["classification"] = classify_result(
                test_case.gemma_answer,
                result["qwen_succeeded"],
                chunks_match=True  # Always True - using exact same rag_context_formatted
            )
            
            # Comparison
            result["comparison"] = {
                "gemma_answer_length": len(test_case.gemma_answer),
                "qwen_answer_length": len(result["qwen_answer"]) if result["qwen_answer"] else 0,
                "gemma_failed": not check_answer_quality(test_case.gemma_answer),
                "qwen_succeeded": result["qwen_succeeded"],
                "chunks_match": result.get("chunks_match", True),  # Always True - using same context
                "gemma_chunk_count": len(test_case.rag_chunks),
                "same_context_used": True  # We're using the exact same rag_context_formatted
            }
            
            results.append(result)
    
    return results


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics."""
    total = len(results)
    if total == 0:
        return {}
    
    classifications = {}
    by_question_type = {}
    
    for result in results:
        classification = result.get("classification", "UNKNOWN")
        question_idx = result.get("question_idx", 0)
        
        classifications[classification] = classifications.get(classification, 0) + 1
        
        q_type = f"Q{question_idx}"
        if q_type not in by_question_type:
            by_question_type[q_type] = {}
        by_question_type[q_type][classification] = by_question_type[q_type].get(classification, 0) + 1
    
    model_failures_confirmed = classifications.get("MODEL_FAILURE_CONFIRMED", 0)
    retrieval_failures_confirmed = classifications.get("RETRIEVAL_FAILURE_CONFIRMED", 0)
    
    return {
        "total_tested": total,
        "classifications": classifications,
        "by_question_type": by_question_type,
        "model_failure_confirmed_rate": (model_failures_confirmed / total * 100) if total > 0 else 0,
        "retrieval_failure_confirmed_rate": (retrieval_failures_confirmed / total * 100) if total > 0 else 0,
        "qwen_success_rate": sum(1 for r in results if r.get("qwen_succeeded", False)) / total * 100 if total > 0 else 0
    }


def main():
    input_dir = Path("/home/himanshu/dev/output/gemma3_rag_concise_missing_ans")
    qa_dir = DEFAULT_QA_DIR
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test model with Gemma-3 context")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit number of test cases")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Model path")
    parser.add_argument("--tokenizer-id", type=str, default=None, help="Tokenizer ID (for Gemma-3)")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization (bitsandbytes for Gemma-3)")
    parser.add_argument("--dtype", type=str, default="float16", help="Dtype (bfloat16 for Gemma-3)")
    parser.add_argument("--gpu-memory-util", type=float, default=0.80, help="GPU memory utilization")
    parser.add_argument("--enforce-eager", action="store_true", help="Enable enforce_eager (for Gemma-3)")
    args = parser.parse_args()
    
    # Determine output dir based on model
    if "gemma" in args.model_path.lower() or args.quantization:
        output_dir = Path("/home/himanshu/dev/output/gemma3_gemma_context_comparison")
    else:
        output_dir = Path("/home/himanshu/dev/output/qwen_gemma_context_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Load test cases
    logger.info("Loading test cases from Gemma-3 filtered answers...")
    test_cases = load_test_cases(input_dir, qa_dir, test_limit=args.test_limit)
    logger.info(f"Loaded {len(test_cases)} test cases")
    
    if not test_cases:
        logger.error("No test cases found")
        sys.exit(0)
    
    # Initialize model wrapper - we'll inject the exact context from Gemma-3
    # RAG is enabled but we'll bypass retrieval and use Gemma's rag_context_formatted directly
    model_name = "Gemma-3" if "gemma" in args.model_path.lower() or args.quantization else "Qwen-VL"
    logger.info(f"Initializing {model_name} wrapper...")
    try:
        wrapper_kwargs = {
            "model_path": args.model_path,
            "use_rag": True,  # RAG enabled for prompt formatting, but we inject Gemma's context
            "gpu_memory_utilization": args.gpu_memory_util,
            "logger": logger
        }
        
        # Add Gemma-3 specific parameters if provided
        if args.tokenizer_id:
            wrapper_kwargs["tokenizer_id"] = args.tokenizer_id
        if args.quantization:
            wrapper_kwargs["quantization"] = args.quantization
        if args.dtype:
            wrapper_kwargs["dtype"] = args.dtype
        if args.enforce_eager:
            wrapper_kwargs["enforce_eager"] = True
        
        wrapper = VLLMRagWrapper(**wrapper_kwargs)
        logger.info(f"{model_name} wrapper initialized successfully")
        logger.info("Using EXACT same rag_context_formatted from Gemma-3 (concatenated chunks)")
    except Exception as e:
        logger.error(f"Failed to initialize Qwen wrapper: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Process test cases
    logger.info("Processing test cases...")
    results = process_test_cases(test_cases, wrapper, batch_size=args.batch_size)
    
    # Generate summary
    summary = generate_summary(results)
    logger.info(f"\nSummary:")
    logger.info(f"  Total tested: {summary.get('total_tested', 0)}")
    logger.info(f"  Model failures confirmed: {summary.get('classifications', {}).get('MODEL_FAILURE_CONFIRMED', 0)}")
    logger.info(f"  Retrieval failures confirmed: {summary.get('classifications', {}).get('RETRIEVAL_FAILURE_CONFIRMED', 0)}")
    logger.info(f"  Qwen success rate: {summary.get('qwen_success_rate', 0):.1f}%")
    
    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to: {results_path}")
    
    # Generate markdown report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Qwen-VL vs Gemma-3 Context Comparison Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Tested:** {summary.get('total_tested', 0)}\n")
        f.write(f"- **Model Failures Confirmed:** {summary.get('classifications', {}).get('MODEL_FAILURE_CONFIRMED', 0)} ({summary.get('model_failure_confirmed_rate', 0):.1f}%)\n")
        f.write(f"- **Retrieval Failures Confirmed:** {summary.get('classifications', {}).get('RETRIEVAL_FAILURE_CONFIRMED', 0)} ({summary.get('retrieval_failure_confirmed_rate', 0):.1f}%)\n")
        f.write(f"- **Qwen Success Rate:** {summary.get('qwen_success_rate', 0):.1f}%\n")
        f.write(f"- **Same Context Used:** 100% (Qwen uses exact same rag_context_formatted from Gemma-3)\n\n")
        
        f.write("## Breakdown by Question Type\n\n")
        for q_type, counts in summary.get('by_question_type', {}).items():
            f.write(f"### {q_type}\n\n")
            for classification, count in counts.items():
                f.write(f"- {classification}: {count}\n")
            f.write("\n")
        
        f.write("## Sample Results\n\n")
        f.write("### Model Failures Confirmed (Qwen succeeded, Gemma failed)\n\n")
        model_failures = [r for r in results if r.get("classification") == "MODEL_FAILURE_CONFIRMED"][:10]
        for result in model_failures:
            f.write(f"**{result.get('file')} Q{result.get('question_idx')}**\n")
            f.write(f"- Question: {result.get('question', '')[:100]}...\n")
            f.write(f"- Gemma: {result.get('gemma_answer', '')[:150]}...\n")
            f.write(f"- Qwen: {result.get('qwen_answer', '')[:150]}...\n\n")
        
        f.write("### Retrieval Failures Confirmed (Both failed)\n\n")
        retrieval_failures = [r for r in results if r.get("classification") == "RETRIEVAL_FAILURE_CONFIRMED"][:10]
        for result in retrieval_failures:
            f.write(f"**{result.get('file')} Q{result.get('question_idx')}**\n")
            f.write(f"- Question: {result.get('question', '')[:100]}...\n")
            f.write(f"- Gemma: {result.get('gemma_answer', '')[:150]}...\n")
            f.write(f"- Qwen: {result.get('qwen_answer', '')[:150]}...\n\n")
    
    logger.info(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()

