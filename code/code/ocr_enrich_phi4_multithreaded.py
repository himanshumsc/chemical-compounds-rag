#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import queue

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


@dataclass
class EnrichmentConfig:
    chunks_in: Path
    images_dir: Path
    model_dir: Path
    output_path: Path
    min_text_len: int
    limit: Optional[int]
    max_new_tokens: int
    temperature: float
    log_interval: int
    small_run_timeout: int
    batch_size: int
    num_workers: int


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def should_enrich_text(text: Optional[str], min_len: int) -> bool:
    if not text:
        return True
    stripped = text.strip()
    return len(stripped) < min_len


def load_phi4_model_and_processor(model_dir: Path, device: torch.device):
    print(f"[LOAD] Starting model load from {model_dir}")
    print(f"[LOAD] Target device: {device}")
    
    # Trust remote code since Phi-4 MM ships custom processors/modeling
    print("[LOAD] Testing attention implementations for Tesla P4...")
    
    # First try: Enable FlashAttention (may fail on Tesla P4)
    try:
        os.environ.pop("TRANSFORMERS_NO_FLASH_ATTENTION", None)
        os.environ.pop("DISABLE_FLASH_ATTN", None)
        os.environ.pop("FLASH_ATTENTION_FORCE_DISABLE", None)
        print("[LOAD] Attempting FlashAttention (may fail on Tesla P4)")
    except:
        pass

    print("[LOAD] Loading config...")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    print(f"[LOAD] Loaded config model_type={getattr(cfg, 'model_type', None)}")
    
    # Try different attention implementations (skip FlashAttention2 for Tesla P4)
    print("[LOAD] Testing attention implementations for Tesla P4...")
    attention_implementations = ["sdpa", "eager"]  # Skip flash_attention_2 for Tesla P4
    
    for attn_impl in attention_implementations:
        try:
            print(f"[LOAD] Trying {attn_impl}...")
            setattr(cfg, "attn_implementation", attn_impl)
            setattr(cfg, "_attn_implementation_internal", attn_impl)
            print(f"[LOAD] Successfully set {attn_impl}")
            break
        except Exception as e:
            print(f"[LOAD] {attn_impl} failed: {e}")
            continue

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[LOAD] Using dtype: {dtype}")

    def try_load(mdir: Path):
        print(f"[LOAD] Attempting to load model from {mdir}")
        # Optimized approach: Use more GPU memory, reduce CPU offloading
        if device.type == "cuda":
            print("[LOAD] Loading model with optimized CUDA device mapping...")
            print("[LOAD] Config: device_map=auto, max_memory GPU=20GB, CPU=30GB")
            model_local = AutoModelForCausalLM.from_pretrained(
                mdir,
                trust_remote_code=True,
                config=cfg,
                torch_dtype=torch.float16,
                device_map="auto",  # Let transformers handle device placement
                low_cpu_mem_usage=True,
                max_memory={0: "20GB", "cpu": "30GB"},  # Use more GPU memory
            )
            print("[LOAD] Model loaded with optimized GPU memory usage")
        else:
            print("[LOAD] Loading model for CPU...")
            model_local = AutoModelForCausalLM.from_pretrained(
                mdir,
                trust_remote_code=True,
                config=cfg,
                torch_dtype=dtype,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            print("[LOAD] Model loaded for CPU")
        print("[LOAD] Loading processor...")
        proc_local = AutoProcessor.from_pretrained(
            mdir, trust_remote_code=True, use_fast=False
        )
        print("[LOAD] Processor loaded")
        
        # Force slow tokenizer to avoid fast tokenizers JSON parsing issues
        print("[LOAD] Loading slow tokenizer...")
        try:
            slow_tok = AutoTokenizer.from_pretrained(
                mdir, trust_remote_code=True, use_fast=False
            )
            if hasattr(proc_local, "tokenizer"):
                proc_local.tokenizer = slow_tok
            elif hasattr(proc_local, "text_tokenizer"):
                setattr(proc_local, "text_tokenizer", slow_tok)
            print("[LOAD] Slow tokenizer loaded and assigned")
        except Exception as e:
            print(f"[LOAD] Warning: Could not load slow tokenizer: {e}")
        print("[LOAD] Model and processor loading completed")
        return model_local, proc_local

    # First attempt: load directly
    print("[LOAD] Attempting direct load...")
    try:
        result = try_load(model_dir)
        print("[LOAD] Direct load successful!")
        return result
    except Exception as e:
        print(f"[LOAD] Direct load failed: {e}")
        print("[LOAD] Attempting slow-tokenizer staging...")

    # Fallback: stage a view of the model dir without tokenizer.json to force slow tokenizer
    stage_root = Path.home() / "dev" / "models" / ".phi4_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="phi4_", dir=str(stage_root)))
    print(f"[LOAD] Creating staged model directory: {tmpdir}")
    
    for item in model_dir.iterdir():
        if item.name == "tokenizer.json":
            print(f"[LOAD] Skipping {item.name}")
            continue
        target = tmpdir / item.name
        try:
            # Create symlink to avoid copying large shards
            os.symlink(item, target)
            print(f"[LOAD] Linked {item.name}")
        except FileExistsError:
            pass
    print(f"[LOAD] Staged model without tokenizer.json at: {tmpdir}")
    return try_load(tmpdir)


def move_inputs_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            # Only move tensors; no detach or grad changes
            inputs[k] = v.to(device, non_blocking=True)
    return inputs


def run_phi4_ocr_batch(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    images: List[Image.Image],
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    """Process multiple images in a single batch for better GPU utilization."""
    print(f"[OCR_BATCH] Processing {len(images)} images in batch")
    
    # Manual construction of Phi-4 multimodal prompt
    user_prompt = f"<|user|>\n{prompt}\n<|image_1|>\n<|end|>\n"
    assistant_prompt = "<|assistant|>"
    full_prompt = user_prompt + assistant_prompt

    # Ensure model is in eval mode
    model.eval()
    
    with torch.no_grad():
        # Process all images in the batch
        print(f"[OCR_BATCH] Processing inputs with processor for {len(images)} images...")
        model_inputs = processor(
            text=[full_prompt] * len(images),  # Same prompt for all images
            images=images,
            return_tensors="pt",
        )
        print(f"[OCR_BATCH] Processor output keys: {list(model_inputs.keys())}")
        
        # Move inputs to device
        primary_device = next(model.parameters()).device if hasattr(model, 'parameters') else device
        print(f"[OCR_BATCH] Primary device: {primary_device}")
        
        model_inputs = move_inputs_to_device(model_inputs, primary_device)
        print("[OCR_BATCH] Inputs moved to device")

        # Generate for all images in batch
        actual_max_tokens = min(max_new_tokens, 128)
        print(f"[OCR_BATCH] Starting batch generation with max_tokens={actual_max_tokens}")
        
        generate_ids = model.generate(
            **model_inputs,
            max_new_tokens=actual_max_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )
        print(f"[OCR_BATCH] Batch generation completed, output shape: {generate_ids.shape}")
        
        # Decode all results
        print("[OCR_BATCH] Decoding generated tokens...")
        full_decoded = processor.batch_decode(generate_ids, skip_special_tokens=True)
        
        # Extract assistant responses
        results = []
        for i, decoded_text in enumerate(full_decoded):
            if assistant_prompt in decoded_text:
                out_text = decoded_text.split(assistant_prompt)[-1].strip()
            else:
                # Fallback: slice after input length
                input_len = model_inputs.get("input_ids", torch.tensor([])).shape[1]
                out_text = processor.decode(generate_ids[i, input_len:], skip_special_tokens=True).strip()
                if not out_text:
                    prompt_encoded = processor.tokenizer.encode(full_prompt)
                    prompt_len = len(prompt_encoded)
                    out_text = processor.decode(generate_ids[i, prompt_len:], skip_special_tokens=True).strip()
            results.append(out_text)
        
        print(f"[OCR_BATCH] Extracted {len(results)} results")
        
        # Cleanup
        if primary_device.type == "cuda":
            del model_inputs, generate_ids
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        return results


def run_phi4_ocr(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image: Image.Image,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Single image OCR processing."""
    results = run_phi4_ocr_batch(model, processor, [image], device, prompt, max_new_tokens, temperature)
    return results[0]


def generate_extracted_info(ocr_text: str, base_text: str, image_name: str) -> str:
    """Generate structured extracted_info field with summary and analysis."""
    import json
    
    # Analyze content types
    content_types = []
    if "chemical" in ocr_text.lower() or "compound" in ocr_text.lower():
        content_types.append("chemical compound information")
    if "formula" in ocr_text.lower() or any(char in ocr_text for char in ["C", "H", "O", "N", "S"]):
        content_types.append("chemical formulas")
    if "table" in ocr_text.lower() or "|" in ocr_text:
        content_types.append("tabular data")
    if "diagram" in ocr_text.lower() or "structure" in ocr_text.lower():
        content_types.append("diagram")
    if "reference" in ocr_text.lower() or "bibliography" in ocr_text.lower():
        content_types.append("reference")
    
    # Generate summary
    summary = f"Page contains "
    if content_types:
        summary += ", ".join(content_types)
    else:
        summary += "text content"
    summary += f" ({len(ocr_text)} characters)"
    
    # Generate parsed_tables (extract any table-like content)
    parsed_tables = ""
    if "|" in ocr_text:
        # Try to extract table content
        lines = ocr_text.split('\n')
        table_lines = [line for line in lines if '|' in line]
        if table_lines:
            parsed_tables = "\n".join(table_lines)
    
    # Generate image descriptions
    image_descriptions = f"The image {image_name} was processed via Phi-4 multimodal OCR for text extraction."
    if content_types:
        image_descriptions += f" Detected content types: {', '.join(content_types)}."
    
    extracted_info = {
        "summary": summary,
        "parsed_tables": parsed_tables,
        "image_descriptions": image_descriptions,
        "content_types": content_types,
        "text_length": len(ocr_text),
        "processing_method": "phi4_multimodal_ocr",
        "has_original_text": bool(base_text.strip()),
        "has_tables": bool(parsed_tables),
        "table_count": len([line for line in ocr_text.split('\n') if '|' in line])
    }
    
    return json.dumps(extracted_info, ensure_ascii=False)


def transform_tables_format(row: Dict[str, Any]) -> None:
    """Transform tables to the expected format with table_data structure."""
    tables = row.get("tables", [])
    if not tables:
        return
    
    # Convert string tables to the expected table_data format
    transformed_tables = []
    for i, table_text in enumerate(tables):
        if isinstance(table_text, str) and table_text.strip():
            # Parse the table text into cells
            lines = table_text.strip().split('\n')
            cells = []
            
            for line in lines:
                if line.strip():
                    # Split by common delimiters or treat as single cell
                    if '|' in line:
                        # Already in table format
                        cell_row = [cell.strip() for cell in line.split('|') if cell.strip()]
                    else:
                        # Treat as single cell
                        cell_row = [line.strip()]
                    cells.append(cell_row)
            
            # Create table_data structure
            table_data = {
                "cells": cells,
                "position": f"Page {row.get('page_range', 'unknown')}, Block {i+1}"
            }
            
            transformed_tables.append({
                "table_data": [table_data]  # Wrap in list as per spec
            })
    
    row["tables"] = transformed_tables


def process_chunk_batch_threaded(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    device: torch.device,
    chunk_batch: List[Dict[str, Any]],
    cfg: EnrichmentConfig,
    images_dir: Path,
    thread_id: int,
) -> List[Dict[str, Any]]:
    """Process a batch of chunks using threading with GPU memory management."""
    print(f"[THREAD-{thread_id}] Processing {len(chunk_batch)} chunks")
    
    # Prepare images for batch processing
    images = []
    valid_chunks = []
    image_paths = []
    
    for chunk in chunk_batch:
        images_list = chunk.get("images", []) or []
        image_path = None
        for candidate in images_list:
            p = Path(candidate)
            if not p.is_absolute():
                p = images_dir / p
            if p.exists():
                image_path = p
                break
        
        if image_path:
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    # Resize for better batch processing
                    max_size = 512
                    if max(img.size) > max_size:
                        ratio = max_size / max(img.size)
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    images.append(img)
                    valid_chunks.append(chunk)
                    image_paths.append(image_path)
            except Exception as e:
                print(f"[THREAD-{thread_id}] Failed to load image {image_path}: {e}")
                chunk["ocr_failed"] = str(e)
                valid_chunks.append(chunk)
        else:
            print(f"[THREAD-{thread_id}] No image found for chunk")
            valid_chunks.append(chunk)
    
    if not images:
        return valid_chunks
    
    # Process images in batch
    prompt = (
        "Extract all text from this image, including tables and labels, and output as structured text. "
        "Preserve chemical formulas and names."
    )
    
    try:
        t0 = time.time()
        ocr_texts = run_phi4_ocr_batch(
            model, processor, images, device, prompt, cfg.max_new_tokens, cfg.temperature
        )
        dt = time.time() - t0
        
        # Update chunks with OCR results
        for i, chunk in enumerate(valid_chunks):
            if i < len(ocr_texts):
                base_text = chunk.get("text") or ""
                sep = "\n\nOCR Extracted: " if base_text else ""
                chunk["text"] = f"{base_text}{sep}{ocr_texts[i]}" if base_text else ocr_texts[i]
                chunk["enriched_via_ocr"] = True
                
                # Generate extracted_info
                if i < len(image_paths):
                    extracted_info = generate_extracted_info(ocr_texts[i], base_text, image_paths[i].name)
                    chunk["extracted_info"] = extracted_info
                
                # Transform tables format
                transform_tables_format(chunk)
        
        print(f"[THREAD-{thread_id}] Processed {len(images)} images in {dt:.1f}s ({dt/len(images):.1f}s per image)")
        
    except Exception as e:
        print(f"[THREAD-{thread_id}] Failed to process batch: {e}")
        for chunk in valid_chunks:
            chunk["ocr_failed"] = str(e)
    
    return valid_chunks


def enrich_chunks(cfg: EnrichmentConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Monitor GPU memory
    if device.type == "cuda":
        def print_gpu_memory():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        def print_gpu_memory():
            pass

    # Preload sparse indices
    print("Scanning for sparse chunks...")
    sparse_indices: List[int] = []
    cached_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(cfg.chunks_in)):
        cached_rows.append(row)
        text = row.get("text")
        if should_enrich_text(text, cfg.min_text_len):
            sparse_indices.append(idx)
    print(f"Found {len(sparse_indices)} sparse chunks (text < {cfg.min_text_len} chars)")

    if cfg.limit is not None:
        sparse_indices = sparse_indices[: cfg.limit]
        print(f"Limiting enrichment to first {len(sparse_indices)} sparse chunks for this run")

    if not sparse_indices:
        print("No sparse chunks to enrich. Copying input to output unchanged.")
        write_jsonl(cfg.output_path, cached_rows)
        return

    # Load model
    print("[MAIN] Loading Phi-4 model + processor...")
    model, processor = load_phi4_model_and_processor(cfg.model_dir, device)
    print("[MAIN] Model loaded, setting to eval mode...")
    model.eval()
    torch.set_grad_enabled(False)
    print("[MAIN] Model setup complete, checking memory...")
    print_gpu_memory()

    # Process chunks in batches with multi-threading
    updated_rows: List[Dict[str, Any]] = cached_rows
    start_total = time.time()
    
    # Create batches of chunks
    chunk_batches = []
    for i in range(0, len(sparse_indices), cfg.batch_size):
        batch_indices = sparse_indices[i:i + cfg.batch_size]
        batch_chunks = [updated_rows[idx] for idx in batch_indices]
        chunk_batches.append((batch_indices, batch_chunks))
    
    print(f"[MAIN] Created {len(chunk_batches)} batches of size {cfg.batch_size}")
    print(f"[MAIN] Using {cfg.num_workers} worker threads")
    
    # Process batches with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        # Submit all batches to thread pool
        future_to_batch = {}
        for batch_idx, (batch_indices, batch_chunks) in enumerate(chunk_batches, 1):
            future = executor.submit(
                process_chunk_batch_threaded,
                model, processor, device, batch_chunks, cfg, cfg.images_dir, batch_idx
            )
            future_to_batch[future] = (batch_idx, batch_indices, batch_chunks)
        
        # Process completed batches
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_idx, batch_indices, batch_chunks = future_to_batch[future]
            try:
                processed_chunks = future.result()
                
                # Update rows
                for i, chunk in enumerate(processed_chunks):
                    updated_rows[batch_indices[i]] = chunk
                
                completed_batches += 1
                print_gpu_memory()
                
                # Logging
                if (cfg.limit is not None and cfg.limit <= 10) or (completed_batches % max(cfg.log_interval, 1) == 0):
                    enriched_count = sum(1 for chunk in processed_chunks if chunk.get('enriched_via_ocr', False))
                    print(f"[BATCH {batch_idx}/{len(chunk_batches)}] Enriched {enriched_count}/{len(processed_chunks)} chunks")
                    
            except Exception as e:
                print(f"[BATCH {batch_idx}/{len(chunk_batches)}] Failed to process batch: {e}")
                for chunk in batch_chunks:
                    chunk["ocr_failed"] = str(e)
                    updated_rows[batch_indices[batch_idx]] = chunk

    total_dt = time.time() - start_total
    print(f"Enrichment finished in {total_dt/60:.1f} min; writing output -> {cfg.output_path}")
    
    # Ensure all chunks have consistent format
    print("Ensuring consistent format for all chunks...")
    for row in updated_rows:
        if not row.get("enriched_via_ocr", False):
            # Add basic extracted_info for non-OCR chunks
            if "extracted_info" not in row:
                base_text = row.get("text", "")
                tables = row.get("tables", [])
                page_range = row.get("page_range", "unknown")
                
                # Generate parsed_tables in markdown format
                parsed_tables = ""
                if tables:
                    table_markdowns = []
                    for table in tables:
                        if "table_data" in table and table["table_data"]:
                            cells = table["table_data"][0]["cells"]
                            if cells:
                                # Create markdown table
                                if len(cells) > 0:
                                    # Header row
                                    header = "| " + " | ".join(cells[0]) + " |"
                                    separator = "| " + " | ".join(["---"] * len(cells[0])) + " |"
                                    table_markdown = header + "\n" + separator + "\n"
                                    
                                    # Data rows
                                    for row_cells in cells[1:]:
                                        data_row = "| " + " | ".join(row_cells) + " |"
                                        table_markdown += data_row + "\n"
                                    
                                    table_markdowns.append(table_markdown.strip())
                    
                    parsed_tables = "\n\n".join(table_markdowns)
                
                # Generate image descriptions
                images = row.get("images", [])
                image_descriptions = []
                for img_path in images:
                    img_name = Path(img_path).name if img_path else "unknown"
                    image_descriptions.append(f"The image {img_name} contains the page content extracted via PyMuPDF.")
                
                # Create summary
                content_types = []
                if "chemical" in base_text.lower() or "compound" in base_text.lower():
                    content_types.append("chemical compound information")
                if "formula" in base_text.lower():
                    content_types.append("chemical formulas")
                if tables:
                    content_types.append("tabular data")
                
                summary = f"Page {page_range} contains "
                if content_types:
                    summary += ", ".join(content_types)
                else:
                    summary += "text content"
                summary += f" ({len(base_text)} characters)"
                
                extracted_info = {
                    "summary": summary,
                    "parsed_tables": parsed_tables,
                    "image_descriptions": "; ".join(image_descriptions),
                    "content_types": content_types,
                    "text_length": len(base_text),
                    "processing_method": "pymupdf_native",
                    "has_tables": len(tables) > 0,
                    "table_count": len(tables)
                }
                row["extracted_info"] = json.dumps(extracted_info, ensure_ascii=False)
            
            # Transform tables format for non-OCR chunks
            transform_tables_format(row)
    
    write_jsonl(cfg.output_path, updated_rows)


def parse_args() -> EnrichmentConfig:
    home = Path(os.path.expanduser("~"))
    default_chunks = home / "dev" / "data" / "processed" / "chemical-compounds_chunks.jsonl"
    default_images = home / "dev" / "data" / "processed" / "pdf_extracted_images"
    default_model = home / "dev" / "models" / "PHI4"
    default_out = home / "dev" / "data" / "processed" / "chemical-compounds_chunks_enriched.jsonl"

    p = argparse.ArgumentParser()
    p.add_argument("--chunks-in", type=Path, default=default_chunks)
    p.add_argument("--images-dir", type=Path, default=default_images)
    p.add_argument("--model-dir", type=Path, default=default_model)
    p.add_argument("--output", type=Path, default=default_out)
    p.add_argument("--min-text-len", type=int, default=50)
    p.add_argument("--limit", type=int, default=10, help="Limit sparse pages to process (None for all)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--log-interval", type=int, default=100, help="Log every N pages for full runs")
    p.add_argument("--batch-size", type=int, default=4, help="Number of images to process in each batch")
    p.add_argument("--num-workers", type=int, default=4, help="Number of worker threads")
    p.add_argument(
        "--small-run-timeout",
        type=int,
        default=180,
        help="Abort if limit<=10 and total processing exceeds this many seconds",
    )
    args = p.parse_args()

    limit_val: Optional[int] = None if args.limit is None or args.limit < 0 else int(args.limit)
    return EnrichmentConfig(
        chunks_in=args.chunks_in,
        images_dir=args.images_dir,
        model_dir=args.model_dir,
        output_path=args.output,
        min_text_len=int(args.min_text_len),
        limit=limit_val,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        log_interval=int(args.log_interval),
        small_run_timeout=int(args.small_run_timeout),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )


def main() -> None:
    cfg = parse_args()
    for p in [cfg.chunks_in, cfg.images_dir, cfg.model_dir]:
        if not p.exists():
            print(f"Missing path: {p}")
            sys.exit(1)
    enrich_chunks(cfg)


if __name__ == "__main__":
    main()
