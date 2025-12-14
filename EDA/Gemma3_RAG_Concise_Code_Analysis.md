# Gemma3 RAG Concise Code Analysis

## Overview

This document explains which code generated `dev/output/output/gemma3_rag_concise` and where images for Q1 generation are located.

---

## Code That Generates `gemma3_rag_concise`

### Main Script

**File:** `dev/code/code/multimodal_qa_runner_gemma3.py`

This is the entry point script that generates the `gemma3_rag_concise` output.

**Key Details:**
- **Output Directory:** `/home/himanshu/dev/output/gemma3_rag_concise` (line 21)
- **Model Path:** `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED` (line 19)
- **Tokenizer:** `google/gemma-3-12b-it` (line 20)
- **Base Function:** Calls `regenerate_from_existing_answers()` from `multimodal_qa_runner_vllm.py`

### Core Implementation

**File:** `dev/code/code/multimodal_qa_runner_vllm.py`

**Function:** `regenerate_from_existing_answers()` (lines 661-1128)

**Key Parameters:**
- `input_dir`: Source directory with existing answer files (default: `/home/himanshu/dev/output/qwen_regenerated`)
- `output_dir`: Where to save new answers (default: `/home/himanshu/dev/output/gemma3_rag_concise`)
- `qa_dir`: Directory with original QA files containing image paths (default: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components`)

---

## Image Location for Q1 Generation

### Where Images Are Loaded From

**Primary Source:** QA JSON files in `qa_dir`

**Default Path:** `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components/`

**Code Location:** `multimodal_qa_runner_vllm.py`, lines 800-805

```python
# Load original QA file for image path
qa_file = qa_dir / source_file
image_path = None
if qa_file.exists():
    qa_data = load_qa_pairs(qa_file)
    image_path = qa_data.get('image_path', '')
```

### Image Path Structure

Each QA JSON file (e.g., `3_22'-Dichlorodiethyl_Sulfide_Mustard_Gas.json`) contains:
```json
{
  "image_path": "/path/to/image.png",
  "qa_pairs": [...]
}
```

The `image_path` field in the QA file points to the actual image file used for Q1.

### Image Loading Function

**Function:** `load_image_sanitized()` (lines 642-653)

**Purpose:** 
- Loads image from path
- Sanitizes it (removes EXIF, re-encodes) to prevent data leakage
- Returns PIL Image object

**Code:**
```python
def load_image_sanitized(image_path: str) -> Optional[Image.Image]:
    """Load and sanitize image (remove EXIF, re-encode)."""
    p = Path(image_path)
    if not p.exists():
        return None
    # ... sanitization code ...
```

### Q1 Processing Flow

1. **Load Answer File** (line 794)
   - Reads existing `*__answers.json` from `input_dir`
   - Extracts `source_file` name

2. **Load QA File** (lines 801-805)
   - Constructs path: `qa_dir / source_file`
   - Loads QA JSON to get `image_path`

3. **Load Image** (line 832)
   - Calls `load_image_sanitized(image_path)`
   - Sanitizes image to prevent metadata leakage

4. **Batch Q1 Processing** (lines 823-857)
   - Collects all Q1 questions and images
   - Processes in batch using `generate_with_vision_batch()`

5. **RAG Image Search** (lines 391-468)
   - Saves image temporarily for ChromaDB search
   - Performs image similarity search
   - Retrieves similar compound chunks
   - Augments prompt with context

---

## Path Structure (VM vs Current Location)

### Original VM Paths (as seen in code)

**Input Directory:**
- `/home/himanshu/dev/output/qwen_regenerated`

**Output Directory:**
- `/home/himanshu/dev/output/gemma3_rag_concise`

**QA Directory:**
- `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components`

**ChromaDB Path:**
- `/home/himanshu/dev/data/chromadb`

**Model Path:**
- `/home/himanshu/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED`

### Current Location (After Copy from VM)

**Base Path Changed:** `/home/himanshu/MSC_FINAL/` instead of `/home/himanshu/`

**Updated Paths:**
- Input: `/home/himanshu/MSC_FINAL/dev/output/qwen_regenerated`
- Output: `/home/himanshu/MSC_FINAL/dev/output/output/gemma3_rag_concise` (note: extra `output/` level)
- QA: `/home/himanshu/MSC_FINAL/dev/test/test/data/processed/qa_pairs_individual_components`
- ChromaDB: `/home/himanshu/MSC_FINAL/dev/data/data/chromadb`
- Model: `/home/himanshu/MSC_FINAL/dev/models/GEMMA3_QAT_Q4_0_UNQUANTIZED`

**Note:** The actual output is at `dev/output/output/gemma3_rag_concise` (with double `output/`), suggesting the code may have been run with a different `output_dir` parameter or the directory structure changed.

---

## Image Path Examples

Based on the JSON structure, images are typically stored in:

### Possible Image Locations

1. **QA Generation Output:**
   - `/home/himanshu/MSC_FINAL/dev/qa_generation/qa_generation/output/`
   - Images generated during QA pair creation

2. **Extracted PDF Images:**
   - `/home/himanshu/MSC_FINAL/dev/data/data/processed/pdf_extracted_images/`
   - Images extracted from the PDF

3. **Test Images:**
   - `/home/himanshu/MSC_FINAL/dev/test/test/extracted_images/images/`
   - Test/extracted images

4. **Input Images:**
   - `/home/himanshu/MSC_FINAL/dev/input_img/input_img/`
   - Manual input images

### Finding Actual Image Paths

**Example from QA file:**
```json
{
  "image_path": "/home/himanshu/dev/test/extracted_images/renders/page_0056_2,2'-Dichlorodiethyl_Sulfide_(Mustard_Gas)_chapter_start.png"
}
```

**Original VM Path (in QA files):** `/home/himanshu/dev/test/extracted_images/renders/`

**Actual Current Location (found):** `/home/himanshu/MSC_FINAL/dev/test/test/extracted_images/renders/`

**⚠️ Path Mismatch:** 
- QA files reference: `/home/himanshu/dev/test/extracted_images/renders/`
- Actual location: `/home/himanshu/MSC_FINAL/dev/test/test/extracted_images/renders/`

**Images Found:**
- `page_0056_2,2'-Dichlorodiethyl_Sulfide_(Mustard_Gas)_chapter_start.png`
- `page_0057_2,2'-Dichlorodiethyl_Sulfide_(Mustard_Gas)_following_page.png`

**Note:** The code will fail to load images if paths in QA files are not updated or if a path mapping/fallback is not implemented.

**To find actual image locations:**

```bash
# Find QA files
find /home/himanshu/MSC_FINAL/dev -name "*22'-Dichlorodiethyl*" -type f

# Check image_path in QA file
cat /home/himanshu/MSC_FINAL/dev/test/test/data/processed/qa_pairs_individual_components/3_22'-Dichlorodiethyl_Sulfide_Mustard_Gas.json | python3 -c "import json, sys; print(json.load(sys.stdin).get('image_path'))"

# Search for actual image file
find /home/himanshu/MSC_FINAL/dev -name "*Dichlorodiethyl*" -name "*.png" 2>/dev/null
```

---

## Running the Code

### Command to Generate gemma3_rag_concise

```bash
cd /home/himanshu/MSC_FINAL/dev/code/code
source .venv_phi4_req/bin/activate  # or llama-env

python multimodal_qa_runner_gemma3.py \
    --input-dir /home/himanshu/MSC_FINAL/dev/output/qwen_regenerated \
    --output-dir /home/himanshu/MSC_FINAL/dev/output/output/gemma3_rag_concise \
    --qa-dir /home/himanshu/MSC_FINAL/dev/test/test/data/processed/qa_pairs_individual_components \
    --chromadb-path /home/himanshu/MSC_FINAL/dev/data/data/chromadb \
    --batch-size 10
```

### Shell Scripts

**File:** `dev/code/code/run_gemma3_qa_background.sh`

This script runs the Gemma3 QA generation in the background.

---

## Key Code Sections

### 1. Image Loading (multimodal_qa_runner_vllm.py:800-832)

```python
# Load original QA file for image path
qa_file = qa_dir / source_file
image_path = None
if qa_file.exists():
    qa_data = load_qa_pairs(qa_file)
    image_path = qa_data.get('image_path', '')

# Later...
img = load_image_sanitized(data['image_path']) if data['image_path'] else None
```

### 2. Q1 Batch Processing (multimodal_qa_runner_vllm.py:823-857)

```python
# Batch Q1: Collect all Q1 questions and images
q1_prompts = []
q1_images = []
q1_indices = []

for idx, data in enumerate(batch_data):
    existing_answers = data['answer_data'].get('answers', [])
    if len(existing_answers) > 0:
        q1 = existing_answers[0].get('question', '')
        img = load_image_sanitized(data['image_path']) if data['image_path'] else None
        if q1 and img:
            q1_prompts.append(q1)
            q1_images.append(img)
            q1_indices.append(idx)
```

### 3. RAG Image Search (multimodal_qa_runner_vllm.py:391-468)

```python
def generate_with_vision_batch(...):
    # Save image temporarily for ChromaDB search
    temp_image_path = self._save_temp_image(image)
    
    # Use IMAGE-ONLY similarity search
    chunks = self.search_system.image_similarity_search(
        temp_image_path, n_results=self.n_chunks
    )
    
    # Build context from chunks
    context = self._build_context_from_chunks(chunks)
    
    # Augment prompt with context
    augmented_prompt = self._augment_prompt_with_context(prompt, context)
```

---

## Summary

1. **Code:** `multimodal_qa_runner_gemma3.py` → calls `multimodal_qa_runner_vllm.py::regenerate_from_existing_answers()`

2. **Images:** Loaded from `image_path` field in QA JSON files located in `qa_dir` (default: `qa_pairs_individual_components/`)

3. **Path Changes:** Code uses `/home/himanshu/dev/` but actual location is `/home/himanshu/MSC_FINAL/dev/` (and output has extra `output/` level)

4. **Q1 Flow:** Load QA file → Extract `image_path` → Load & sanitize image → Batch process with RAG image similarity search

---

## Finding Actual Image Locations

To find where images are actually stored for a specific compound:

```bash
# 1. Find the QA file
QA_FILE="/home/himanshu/MSC_FINAL/dev/test/test/data/processed/qa_pairs_individual_components/3_22'-Dichlorodiethyl_Sulfide_Mustard_Gas.json"

# 2. Extract image_path
cat "$QA_FILE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('image_path', 'NOT FOUND'))"

# 3. Check if image exists
IMAGE_PATH=$(cat "$QA_FILE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data.get('image_path', ''))")
if [ -f "$IMAGE_PATH" ]; then
    echo "✅ Image found at: $IMAGE_PATH"
else
    echo "❌ Image not found. Searching..."
    find /home/himanshu/MSC_FINAL -name "*$(basename "$IMAGE_PATH")" 2>/dev/null
fi
```

