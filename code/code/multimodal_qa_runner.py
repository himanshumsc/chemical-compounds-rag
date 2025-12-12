#!/usr/bin/env python3
"""
Modular multimodal QA runner:
- Choose model: phi4 or qwen
- Feed 10 QA JSONs from dev/test/data/processed/qa_pairs_individual_components
- Q1 with image, Q2-4 text-only
- Save answers under dev/output/<model_name>/ as JSON alongside timing logs
"""
import argparse
import json
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import traceback


DEFAULT_QA_DIR = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components")
DEFAULT_OUTPUT_DIR = Path("/home/himanshu/dev/output")


@dataclass
class GenerationResult:
    text: str
    latency_s: float


class BaseModelWrapper:
    def generate(self, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: int = 128) -> GenerationResult:
        raise NotImplementedError


class Phi4Wrapper(BaseModelWrapper):
    def __init__(self, model_path: str = "/home/himanshu/dev/models/PHI4_ONNX"):
        from transformers import AutoModel, AutoProcessor, AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        cfg._attn_implementation = "sdpa"
        self.model = AutoModel.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def generate(self, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: int = 128) -> GenerationResult:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt") if image else self.processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        latency = time.time() - start
        # Decode only newly generated tokens (exclude prompt)
        input_len = inputs.get("input_ids").shape[1]
        gen_ids = outputs[0][input_len:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True)
        text = postprocess_assistant_only(text)
        return GenerationResult(text=text, latency_s=latency)


class QwenWrapper(BaseModelWrapper):
    def __init__(self, model_path: str = "/home/himanshu/dev/models/QWEN_AWQ"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def generate(self, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: int = 128) -> GenerationResult:
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            templated = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(text=[templated], images=[image], return_tensors="pt")
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            templated = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(text=[templated], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        latency = time.time() - start
        # Decode only newly generated tokens (exclude chat template tokens)
        input_len = inputs.get("input_ids").shape[1]
        gen_ids = outputs[0][input_len:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True)
        text = postprocess_assistant_only(text)
        return GenerationResult(text=text, latency_s=latency)

    def generate_batch(self, prompts: List[str], images: Optional[List[Optional[Image.Image]]] = None, max_new_tokens: int = 128) -> List[GenerationResult]:
        # images: list aligned to prompts; None entries mean text-only
        messages_list = []
        if images is None:
            images = [None] * len(prompts)
        batched_images: List[Optional[Image.Image]] = []
        for p, img in zip(prompts, images):
            if img is not None:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": p},
                    ],
                }]
                templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                messages_list.append(templated)
                batched_images.append(img)
            else:
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": p}],
                }]
                templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                messages_list.append(templated)
                batched_images.append(None)

        # Processor requires consistent list for images: replace None with a dummy 0-length list
        has_any_image = any(img is not None for img in batched_images)
        if has_any_image:
            proc = self.processor(text=messages_list, images=[img for img in batched_images], padding=True, return_tensors="pt")
        else:
            proc = self.processor(text=messages_list, padding=True, return_tensors="pt")
        proc = {k: v.to(self.model.device) for k, v in proc.items()}
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **proc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        latency = time.time() - start
        # Per-sample input lengths may differ in batched chat templates; use attention_mask to infer
        input_ids = proc.get("input_ids")
        attn = proc.get("attention_mask")
        if attn is not None:
            input_lens = attn.sum(dim=1).tolist()
        else:
            input_lens = [input_ids.shape[1]] * input_ids.shape[0]
        results: List[GenerationResult] = []
        for i in range(outputs.shape[0]):
            gen_ids = outputs[i][int(input_lens[i]):]
            text = self.processor.decode(gen_ids, skip_special_tokens=True)
            text = postprocess_assistant_only(text)
            results.append(GenerationResult(text=text, latency_s=latency))
        return results


def postprocess_assistant_only(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    # Find the last occurrence of an assistant role marker and keep content after it
    matches = list(re.finditer(r'(?:^|\n)assistant\s*:?\s*\n', t, flags=re.IGNORECASE))
    if matches:
        t = t[matches[-1].end():].lstrip()
    # Remove any leading explicit role labels like "Assistant:"
    t = re.sub(r'^assistant\s*:?\s*', '', t, flags=re.IGNORECASE).lstrip()
    # Also strip any lingering system/user role headers if present at start
    t = re.sub(r'^(system|user)\s*:?\s*', '', t, flags=re.IGNORECASE).lstrip()
    return t


def load_qa_pairs(qa_path: Path) -> Dict[str, Any]:
    return json.loads(qa_path.read_text(encoding="utf-8"))


def load_image_sanitized(image_path: str) -> Optional[Image.Image]:
    """
    Open an image and strip any filename/EXIF metadata by re-encoding in-memory.
    Returns a fresh PIL.Image without path/EXIF. If missing, returns None.
    """
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


def run_sample(model_name: str, limit: int, qa_dir: Path, output_dir: Path, max_new_tokens: int = 128) -> Dict[str, Any]:
    logger = _ensure_logger(model_name.lower(), output_dir)
    if model_name.lower() == "phi4":
        wrapper: BaseModelWrapper = Phi4Wrapper()
        model_tag = "phi4"
    elif model_name.lower() == "qwen":
        wrapper = QwenWrapper()
        model_tag = "qwen"
    else:
        raise ValueError("model_name must be one of: phi4, qwen")

    out_dir = output_dir / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_files = sorted([p for p in qa_dir.glob("*_*.json")])[:limit]

    overall_start = time.time()
    results_summary: List[Dict[str, Any]] = []

    for qa_file in qa_files:
        logger.info(f"Processing set file={qa_file.name}")
        data = load_qa_pairs(qa_file)
        questions = [pair.get("question", "").strip() for pair in data.get("qa_pairs", [])]
        image_path = data.get("image_path", "")

        answers: List[Dict[str, Any]] = []

        # Q1: with image only (do not leak compound name via prompt)
        q1 = questions[0] if questions else ""
        img = load_image_sanitized(image_path) if image_path else None
        logger.info(f"Q1 start file={qa_file.name} image={'yes' if img is not None else 'no'}")
        try:
            res1 = wrapper.generate(q1, image=img, max_new_tokens=max_new_tokens)
            logger.info(f"Q1 done  file={qa_file.name} latency={round(res1.latency_s,2)}s")
            answers.append({"question": q1, "answer": res1.text, "latency_s": round(res1.latency_s, 2)})
        except Exception as e:
            logger.error(f"Q1 error file={qa_file.name}: {e}\n{traceback.format_exc()}")
            answers.append({"question": q1, "answer": "", "latency_s": 0.0, "error": str(e)})

        # Q2..: text only
        for q in questions[1:4]:
            qn = questions.index(q) + 1 if q in questions else 0
            logger.info(f"Q{qn} start file={qa_file.name}")
            try:
                res = wrapper.generate(q, image=None, max_new_tokens=max_new_tokens)
                logger.info(f"Q{qn} done  file={qa_file.name} latency={round(res.latency_s,2)}s")
                answers.append({"question": q, "answer": res.text, "latency_s": round(res.latency_s, 2)})
            except Exception as e:
                logger.error(f"Q{qn} error file={qa_file.name}: {e}\n{traceback.format_exc()}")
                answers.append({"question": q, "answer": "", "latency_s": 0.0, "error": str(e)})

        # Save per-file output
        out = {
            "source_file": qa_file.name,
            "model": model_tag,
            "image_used_for_q1": bool(img is not None),
            "answers": answers,
        }
        out_path = out_dir / qa_file.name.replace(".json", f"__answers.json")
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

        results_summary.append({
            "file": qa_file.name,
            "q_count": len(answers),
            "q1_latency_s": answers[0]["latency_s"],
            "avg_latency_s": sum(a["latency_s"] for a in answers) / max(1, len(answers)),
        })

    total_time = time.time() - overall_start
    summary = {
        "model": model_tag,
        "processed": len(qa_files),
        "total_time_s": total_time,
        "avg_per_set_s": total_time / max(1, len(qa_files)),
        "details": results_summary,
    }

    # Save summary
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("RUN REPORT: " + json.dumps(summary))
    return summary


def run_sample_batched(model_name: str, qa_dir: Path, output_dir: Path, max_new_tokens: int, total_sets: int, batch_size: int) -> Dict[str, Any]:
    logger = _ensure_logger(model_name.lower(), output_dir)
    """
    Batched runner that processes QA sets in mixed batches.
    Each batch contains B QA sets; for each set we include its 4 questions
    in-order: [Q1 with image, Q2 text, Q3 text, Q4 text]. This avoids bursts
    of image-only or text-only work and mirrors real usage.
    """
    model_tag = model_name.lower()
    out_dir = output_dir / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize a single wrapper
    if model_tag == "qwen":
        wrapper: BaseModelWrapper = QwenWrapper()
    elif model_tag == "phi4":
        wrapper = Phi4Wrapper()
    else:
        raise ValueError("model_name must be one of: phi4, qwen")

    qa_files = sorted([p for p in qa_dir.glob("*_*.json")])[:total_sets]

    overall_start = time.time()
    per_file_answers: Dict[str, List[Dict[str, Any]]] = {}

    # Process QA files in chunks of batch_size sets
    for i in range(0, len(qa_files), batch_size):
        chunk = qa_files[i:i + batch_size]
        logger.info(f"Batch start idx={i} size={len(chunk)}")

        # Build a mixed batch of prompts/images preserving per-set order (Q1..Q4)
        batch_prompts: List[str] = []
        batch_images: List[Optional[Image.Image]] = []
        batch_sources: List[str] = []
        batch_qidx: List[int] = []

        for qa_file in chunk:
            data = load_qa_pairs(qa_file)
            questions = [pair.get("question", "").strip() for pair in data.get("qa_pairs", [])]
            image_path = data.get("image_path", "")
            img = load_image_sanitized(image_path) if image_path else None

            # Q1 (image)
            if len(questions) > 0:
                batch_prompts.append(questions[0])
                batch_images.append(img)
                batch_sources.append(qa_file.name)
                batch_qidx.append(0)
            # Q2..Q4 (text)
            for q_i in [1, 2, 3]:
                if len(questions) > q_i:
                    batch_prompts.append(questions[q_i])
                    batch_images.append(None)
                    batch_sources.append(qa_file.name)
                    batch_qidx.append(q_i)

        # Run one batched generation for this mixed batch: split to avoid None images
        logger.info(f"Batch generate start prompts={len(batch_prompts)} (mixed image/text)")
        idx_with_img = [j for j, im in enumerate(batch_images) if im is not None]
        idx_text_only = [j for j, im in enumerate(batch_images) if im is None]

        results_by_index: Dict[int, GenerationResult] = {}

        if idx_with_img:
            sub_prompts = [batch_prompts[j] for j in idx_with_img]
            sub_images = [batch_images[j] for j in idx_with_img]
            try:
                if isinstance(wrapper, QwenWrapper):
                    sub_results = wrapper.generate_batch(sub_prompts, images=sub_images, max_new_tokens=max_new_tokens)
                else:
                    sub_results = [wrapper.generate(p, image=im, max_new_tokens=max_new_tokens) for p, im in zip(sub_prompts, sub_images)]
            except Exception as e:
                logger.error(f"Batch image-subgroup error: {e}\n{traceback.format_exc()}")
                sub_results = [GenerationResult(text="", latency_s=0.0) for _ in sub_prompts]
            for j, res in zip(idx_with_img, sub_results):
                results_by_index[j] = res

        if idx_text_only:
            sub_prompts = [batch_prompts[j] for j in idx_text_only]
            try:
                if isinstance(wrapper, QwenWrapper):
                    sub_results = wrapper.generate_batch(sub_prompts, images=None, max_new_tokens=max_new_tokens)
                else:
                    sub_results = [wrapper.generate(p, image=None, max_new_tokens=max_new_tokens) for p in sub_prompts]
            except Exception as e:
                logger.error(f"Batch text-subgroup error: {e}\n{traceback.format_exc()}")
                sub_results = [GenerationResult(text="", latency_s=0.0) for _ in sub_prompts]
            for j, res in zip(idx_text_only, sub_results):
                results_by_index[j] = res

        logger.info("Batch generate done")

        # Scatter results back to per-file buckets preserving batch order
        for j in range(len(batch_prompts)):
            res = results_by_index[j]
            src = batch_sources[j]
            qi = batch_qidx[j]
            prompt = batch_prompts[j]
            per_file_answers.setdefault(src, []).append({
                "question": prompt,
                "answer": res.text,
                "latency_s": round(res.latency_s, 2),
                "q_index": qi,
            })

    # Save outputs per source file
    for src, answers in per_file_answers.items():
        answers_sorted = sorted(answers, key=lambda x: x["q_index"])
        out = {
            "source_file": src,
            "model": model_tag,
            "image_used_for_q1": True if any(a["q_index"] == 0 for a in answers_sorted) else False,
            "answers": [
                {"question": a["question"], "answer": a["answer"], "latency_s": a["latency_s"]}
                for a in answers_sorted
            ],
        }
        out_path = out_dir / src.replace(".json", f"__answers.json")
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    total_time = time.time() - overall_start
    total_questions = len(qa_files) * 4
    summary = {
        "model": model_tag,
        "batched": True,
        "batch_size": batch_size,
        "total_sets": len(qa_files),
        "total_questions": total_questions,
        "total_time_s": total_time,
        "avg_per_question_s": total_time / max(1, total_questions),
    }
    (out_dir / "summary_batched.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("RUN REPORT: " + json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Modular multimodal QA runner")
    parser.add_argument("--model", choices=["phi4", "qwen"], required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--qa-dir", default=str(DEFAULT_QA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--multithread", action="store_true", help="Enable multithreaded question-level execution")
    parser.add_argument("--threads", type=int, default=2, help="Number of threads for multithreaded mode")
    parser.add_argument("--total-questions", type=int, default=20, help="Total questions to process across threads")
    parser.add_argument("--total-sets", type=int, default=0, help="Total compound sets to process (each set = up to 4 questions)")
    parser.add_argument("--batched", action="store_true", help="Enable GPU-friendly batched execution (preferred)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for batched execution")
    args = parser.parse_args()

    if args.batched:
        summary = run_sample_batched(
            model_name=args.model,
            qa_dir=Path(args.qa_dir),
            output_dir=Path(args.output_dir),
            max_new_tokens=args.max_new_tokens,
            total_sets=max(1, args.total_sets or 10),
            batch_size=max(1, args.batch_size),
        )
        print(json.dumps(summary, indent=2))
    elif args.multithread:
        summary = run_sample_multithread(
            model_name=args.model,
            qa_dir=Path(args.qa_dir),
            output_dir=Path(args.output_dir),
            max_new_tokens=args.max_new_tokens,
            num_threads=max(1, args.threads),
            total_questions=max(1, args.total_questions),
            total_sets=max(0, args.total_sets),
        )
        print(json.dumps(summary, indent=2))
    else:
        summary = run_sample(
            model_name=args.model,
            limit=args.limit,
            qa_dir=Path(args.qa_dir),
            output_dir=Path(args.output_dir),
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(summary, indent=2))


# -------------------- Multithreaded question-level runner --------------------

_thread_local = threading.local()
_shared_wrapper_lock = threading.Lock()
_shared_wrapper: Optional[BaseModelWrapper] = None
_generate_lock = threading.Lock()
_logger: Optional[logging.Logger] = None


def _ensure_logger(model_tag: str, output_dir: Path) -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    logs_dir = (output_dir / model_tag / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(f"qa_runner_{model_tag}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _logger = logger
    return logger


def _get_thread_wrapper(model_name: str) -> BaseModelWrapper:
    global _shared_wrapper
    # Fast path if already initialized
    if _shared_wrapper is not None:
        return _shared_wrapper
    # Initialize once under lock
    with _shared_wrapper_lock:
        if _shared_wrapper is None:
            if model_name.lower() == "qwen":
                _shared_wrapper = QwenWrapper()
            else:
                _shared_wrapper = Phi4Wrapper()
    return _shared_wrapper


def _run_question_task(task: Dict[str, Any]) -> Dict[str, Any]:
    wrapper = _get_thread_wrapper(task["model"])  # lazy init per thread
    question = task["question"]
    image_path = task.get("image_path")
    image = None
    if image_path:
        image = load_image_sanitized(image_path)
    # Guard model.generate with a lock to avoid non-thread-safe concurrent calls on GPU
    with _generate_lock:
        res = wrapper.generate(question, image=image, max_new_tokens=task["max_new_tokens"])
    return {
        "source_file": task["source_file"],
        "q_index": task["q_index"],
        "question": task["question"],
        "answer": res.text,
        "latency_s": round(res.latency_s, 2),
        "model": task["model"],
    }


def run_sample_multithread(model_name: str, qa_dir: Path, output_dir: Path, max_new_tokens: int, num_threads: int, total_questions: int, total_sets: int = 0) -> Dict[str, Any]:
    logger = _ensure_logger(model_name.lower(), output_dir)
    model_tag = model_name.lower()
    out_dir = output_dir / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_files = sorted([p for p in qa_dir.glob("*_*.json")])
    tasks: List[Dict[str, Any]] = []

    # Build tasks based on total_sets (preferred) or total_questions fallback
    if total_sets and total_sets > 0:
        selected_files = qa_files[:total_sets]
        for qa_file in selected_files:
            data = load_qa_pairs(qa_file)
            questions = [pair.get("question", "").strip() for pair in data.get("qa_pairs", [])]
            image_path = data.get("image_path", "")
            for idx, q in enumerate(questions[:4]):
                tasks.append({
                    "model": model_tag,
                    "source_file": qa_file.name,
                    "q_index": idx,
                    "question": q,
                    "image_path": image_path if idx == 0 else None,
                    "max_new_tokens": max_new_tokens,
                })
    else:
        # Build question tasks up to total_questions cap
        for qa_file in qa_files:
            if len(tasks) >= total_questions:
                break
            data = load_qa_pairs(qa_file)
            questions = [pair.get("question", "").strip() for pair in data.get("qa_pairs", [])]
            image_path = data.get("image_path", "")
            for idx, q in enumerate(questions[:4]):
                if len(tasks) >= total_questions:
                    break
                tasks.append({
                    "model": model_tag,
                    "source_file": qa_file.name,
                    "q_index": idx,
                    "question": q,
                    "image_path": image_path if idx == 0 else None,
                    "max_new_tokens": max_new_tokens,
                })

    overall_start = time.time()
    results: List[Dict[str, Any]] = []
    logger.info(f"ThreadPool start workers={num_threads} tasks={len(tasks)}")
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = [ex.submit(_run_question_task, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                r = fut.result(timeout=180)
                logger.info(f"Task done file={r['source_file']} q_index={r['q_index']} latency={r['latency_s']}s")
                results.append(r)
            except Exception as e:
                logger.error(f"Task error: {e}\n{traceback.format_exc()}")

    # Aggregate per source_file
    per_file: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        per_file.setdefault(r["source_file"], []).append(r)

    # Save per-file outputs (partial if not all 4 questions processed)
    for src, answers in per_file.items():
        answers_sorted = sorted(answers, key=lambda x: x["q_index"])
        out = {
            "source_file": src,
            "model": model_tag,
            "image_used_for_q1": True if any(a["q_index"] == 0 for a in answers_sorted) else False,
            "answers": [
                {
                    "question": a["question"],
                    "answer": a["answer"],
                    "latency_s": a["latency_s"],
                }
                for a in answers_sorted
            ],
        }
        out_path = out_dir / src.replace(".json", f"__answers.json")
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    total_time = time.time() - overall_start
    summary = {
        "model": model_tag,
        "threads": num_threads,
        "total_questions": len(tasks),
        "total_sets": total_sets,
        "completed": len(results),
        "total_time_s": total_time,
        "avg_per_question_s": total_time / max(1, len(results)),
    }
    (out_dir / "summary_mt.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("RUN REPORT: " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()


