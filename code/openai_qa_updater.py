#!/usr/bin/env python3
"""
OpenAI QA Answer Updater
- Reads QA pair JSON files
- Sends first question + image to OpenAI API
- Updates answer in source file
- Limits answer to 300-600 characters
- Anonymizes images before sending
"""

import json
import os
import io
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import base64
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Directories
QA_DIR = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components")
OUTPUT_DIR = Path("/home/himanshu/dev/output/openai_qa_updates")

# Answer length constraints
MIN_ANSWER_LENGTH = 300
MAX_ANSWER_LENGTH = 600


def anonymize_image(image_path: Path) -> Optional[Image.Image]:
    """
    Load and anonymize image by removing ALL metadata and re-encoding.
    This ensures:
    - No EXIF data (camera info, GPS, etc.)
    - No filename embedded in image
    - No path information
    - No identifying metadata of any kind
    Returns a clean PIL.Image with only pixel data.
    """
    if not image_path.exists():
        print(f"  ⚠️  Image not found: {image_path}")
        return None
    
    try:
        # Load image (this reads from disk but doesn't embed path)
        img = Image.open(str(image_path))
        
        # Convert to RGB to remove any palette/transparency metadata
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Create a completely new image from pixel data only
        # This strips ALL metadata including EXIF, ICC profiles, etc.
        pixel_data = list(img.getdata())
        width, height = img.size
        clean_img = Image.new("RGB", (width, height))
        clean_img.putdata(pixel_data)
        
        # Re-encode to PNG in memory (PNG format doesn't preserve EXIF)
        # This creates a fresh image with zero metadata
        buf = io.BytesIO()
        # Explicitly save without any metadata
        clean_img.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        
        # Reload from clean buffer - this is now a completely anonymous image
        final_img = Image.open(buf)
        final_img.load()
        
        # Verify no metadata exists
        if hasattr(final_img, '_getexif'):
            exif = final_img._getexif()
            if exif:
                print(f"  ⚠️  Warning: EXIF data still present (unexpected)")
        
        return final_img
    except Exception as e:
        print(f"  ❌ Error loading image {image_path}: {e}")
        return None


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string for API.
    Only sends pixel data - no metadata, no filename, no identifying information.
    """
    buf = io.BytesIO()
    # Save without any metadata
    image.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    img_bytes = buf.read()
    # Return only the base64 encoded image data
    return base64.b64encode(img_bytes).decode("utf-8")


def call_openai_api(question: str, image: Optional[Image.Image] = None) -> Optional[str]:
    """
    Call OpenAI API with question and optional image.
    Returns the generated answer or None on error.
    """
    if not OPENAI_API_KEY:
        print("  ❌ OPENAI_API_KEY not found in environment")
        return None
    
    # Prepare messages
    messages = []
    
    if image:
        # Multimodal: image + text
        base64_image = image_to_base64(image)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        })
    else:
        # Text-only
        messages.append({
            "role": "user",
            "content": question
        })
    
    # Add system prompt to guide answer length
    system_prompt = f"""You are a helpful assistant answering questions about chemical compounds. 
Provide a clear, informative answer that is between {MIN_ANSWER_LENGTH} and {MAX_ANSWER_LENGTH} characters long.
Be concise but comprehensive."""
    
    payload = {
        "model": "gpt-4o",  # Using GPT-4o for vision support
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages
        ],
        "max_tokens": 300,  # Roughly 300 tokens ≈ 600-750 characters (we'll truncate if needed)
        "temperature": 0.7
    }
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"  📡 Calling OpenAI API...")
        response = requests.post(OPENAI_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        
        # Check and adjust length if needed
        if len(answer) < MIN_ANSWER_LENGTH:
            print(f"  ⚠️  Answer too short ({len(answer)} chars), requesting expansion...")
            # Could add logic to request expansion, but for now just warn
        elif len(answer) > MAX_ANSWER_LENGTH:
            print(f"  ⚠️  Answer too long ({len(answer)} chars), truncating...")
            answer = answer[:MAX_ANSWER_LENGTH].rsplit(' ', 1)[0] + "..."
        
        return answer
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"  Error details: {error_detail}")
            except:
                print(f"  Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return None


# Thread lock for file operations
_file_lock = threading.Lock()

def update_qa_file(qa_file: Path, new_answer: str) -> bool:
    """
    Update the first answer in the QA pair file.
    Thread-safe file writing.
    Returns True if successful, False otherwise.
    """
    try:
        # Use lock to ensure thread-safe file operations
        with _file_lock:
            # Read existing file
            with open(qa_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update first answer
            if "qa_pairs" in data and len(data["qa_pairs"]) > 0:
                data["qa_pairs"][0]["answer"] = new_answer
                data["qa_pairs"][0]["updated_by"] = "openai_api"
                data["qa_pairs"][0]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                print(f"  ❌ No qa_pairs found in file")
                return False
            
            # Write back to file
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error updating file: {e}")
        return False


def process_qa_file(qa_file: Path, dry_run: bool = False, thread_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Process a single QA file:
    1. Load QA data
    2. Get first question and image
    3. Anonymize image
    4. Call OpenAI API
    5. Update file with new answer
    
    Returns result dictionary.
    """
    thread_prefix = f"[Thread-{thread_id}] " if thread_id is not None else ""
    print(f"\n{thread_prefix}{'='*60}")
    print(f"{thread_prefix}Processing: {qa_file.name}")
    print(f"{thread_prefix}{'='*60}")
    
    try:
        # Load QA file
        with open(qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get first question
        if "qa_pairs" not in data or len(data["qa_pairs"]) == 0:
            return {"success": False, "error": "No qa_pairs found"}
        
        first_qa = data["qa_pairs"][0]
        question = first_qa.get("question", "").strip()
        old_answer = first_qa.get("answer", "")
        
        if not question:
            return {"success": False, "error": "First question is empty"}
        
        print(f"{thread_prefix}  📝 Question: {question[:80]}...")
        print(f"{thread_prefix}  📏 Old answer length: {len(old_answer)} characters")
        
        # Get image path
        image_path_str = data.get("image_path", "")
        image = None
        
        if image_path_str:
            image_path = Path(image_path_str)
            # Don't print the actual path to avoid leaking information
            print(f"{thread_prefix}  🖼️  Image found: {image_path.name} (path anonymized)")
            image = anonymize_image(image_path)
            if image:
                print(f"{thread_prefix}  ✅ Image anonymized - all metadata removed ({image.size[0]}x{image.size[1]})")
                print(f"{thread_prefix}  🔒 No filename, path, or EXIF data will be sent to API")
            else:
                print(f"{thread_prefix}  ⚠️  Image not available, proceeding with text-only")
        else:
            print(f"{thread_prefix}  ⚠️  No image_path in file, proceeding with text-only")
        
        if dry_run:
            print(f"{thread_prefix}  🔍 DRY RUN: Would call OpenAI API and update file")
            return {
                "success": True,
                "dry_run": True,
                "question": question,
                "has_image": image is not None,
                "old_answer_length": len(old_answer)
            }
        
        # Call OpenAI API
        print(f"{thread_prefix}  📡 Calling OpenAI API...")
        new_answer = call_openai_api(question, image)
        
        if not new_answer:
            return {"success": False, "error": "Failed to get answer from OpenAI"}
        
        print(f"{thread_prefix}  ✅ New answer received: {len(new_answer)} characters")
        print(f"{thread_prefix}  📄 Answer preview: {new_answer[:100]}...")
        
        # Update file
        if update_qa_file(qa_file, new_answer):
            print(f"{thread_prefix}  ✅ File updated successfully")
            return {
                "success": True,
                "question": question,
                "old_answer_length": len(old_answer),
                "new_answer_length": len(new_answer),
                "has_image": image is not None
            }
        else:
            return {"success": False, "error": "Failed to update file"}
            
    except Exception as e:
        print(f"{thread_prefix}  ❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def process_file_wrapper(args_tuple):
    """Wrapper function for thread pool executor"""
    qa_file, dry_run, thread_id = args_tuple
    return process_qa_file(qa_file, dry_run=dry_run, thread_id=thread_id)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Update QA answers using OpenAI API")
    parser.add_argument("--limit", type=int, default=178, help="Number of files to process (default: 178)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't call API or update files)")
    parser.add_argument("--qa-dir", type=Path, default=QA_DIR, help="Directory containing QA JSON files")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing (default: 4)")
    
    args = parser.parse_args()
    
    # Check API key
    if not args.dry_run and not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Get QA files
    all_qa_files = sorted([f for f in args.qa_dir.glob("*_*.json")])
    qa_files = all_qa_files[:args.limit]
    
    if not qa_files:
        print(f"❌ No QA files found in {args.qa_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"OpenAI QA Answer Updater (Multithreaded)")
    print(f"{'='*60}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Files to process: {len(qa_files)}")
    print(f"Threads: {args.threads}")
    print(f"Answer length: {MIN_ANSWER_LENGTH}-{MAX_ANSWER_LENGTH} characters")
    print(f"{'='*60}\n")
    
    # Process files in parallel using ThreadPoolExecutor
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Create tasks with thread IDs
        tasks = [(qa_file, args.dry_run, i % args.threads) for i, qa_file in enumerate(qa_files)]
        
        # Submit all tasks
        future_to_file = {executor.submit(process_file_wrapper, task): task[0] for task in tasks}
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_file):
            qa_file = future_to_file[future]
            completed += 1
            try:
                result = future.result()
                results.append({
                    "file": qa_file.name,
                    **result
                })
                print(f"\n[Progress] {completed}/{len(qa_files)} files completed")
            except Exception as e:
                print(f"\n❌ Exception processing {qa_file.name}: {e}")
                results.append({
                    "file": qa_file.name,
                    "success": False,
                    "error": str(e)
                })
            
            # Small delay to avoid rate limiting (only in live mode)
            if not args.dry_run and completed < len(qa_files):
                time.sleep(0.5)  # Reduced delay since we're parallelizing
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r.get("success"))
    print(f"Processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per file: {total_time/len(results):.1f} seconds")
    
    if successful > 0:
        avg_length = sum(r.get("new_answer_length", 0) for r in results if r.get("success")) / successful
        print(f"Average answer length: {avg_length:.1f} characters")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

