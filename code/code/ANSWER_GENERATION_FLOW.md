# Complete Answer Generation Flow

## Overview
This document explains the complete flow from loading source questions to generating final answers for both image-based (Q1) and text-based (Q2-Q4) questions.

## High-Level Flow

```
Source Files → Load Questions → Batch Processing → RAG Retrieval → Prompt Augmentation → vLLM Generation → Answer Assembly → Save Results
```

---

## Step-by-Step Detailed Flow

### 1. **Initialization Phase**

**Location:** `regenerate_from_existing_answers()` function (line 652)

1. **Load source files:**
   - Scans `input_dir` for `*__answers.json` files (e.g., `/home/himanshu/dev/output/qwen_regenerated/`)
   - Each file contains existing questions and answers from previous generation
   - Sorts files alphabetically
   - Applies `test_limit` if specified (for testing)

2. **Initialize components (in order):**
   - **vLLM FIRST** - Loads model, claims GPU resources (lines 120-197)
   - **Processor AFTER vLLM** - For chat template formatting (lines 199-209)
   - **ChromaDB LAST** - RAG search system on CPU (lines 211-223)

---

### 2. **Batch Processing Setup**

**Location:** Lines 784-818

For each batch (default batch_size=6):
1. Load all answer files in the batch
2. For each file:
   - Read existing `*__answers.json` file
   - Extract `source_file` name (e.g., "138_Pyridoxine.json")
   - Load corresponding QA file from `qa_dir` to get `image_path`
   - Store in `batch_data` structure

---

### 3. **Q1 (Image-Based Questions) - Batch Processing**

**Location:** Lines 823-857

#### 3.1 Collect Q1 Questions
- Extract Q1 question from each file's `answers[0]`
- Load and sanitize image from `image_path`
- Collect into `q1_prompts[]` and `q1_images[]` arrays

#### 3.2 RAG Retrieval for Q1
**Location:** `generate_with_vision_batch()` (lines 391-468)

For each Q1 question:
1. **Save image temporarily** (line 409)
   - Save PIL image to temp PNG file
   - Needed for ChromaDB image search

2. **Image Similarity Search** (line 412)
   ```python
   chunks = self.search_system.image_similarity_search(
       temp_image_path, n_results=5
   )
   ```
   - Uses ChromaDB to find 5 most similar images/compounds
   - Returns chunks with text descriptions and similarity scores

3. **Build Context** (line 417)
   ```python
   context = self._build_context_from_chunks(chunks)
   ```
   - Formats chunks as:
     ```
     [Source 1 (Relevance: 0.923)]
     Pyridoxine (vitamin B6) is a water-soluble vitamin...
     
     [Source 2 (Relevance: 0.891)]
     ...
     ```

4. **Augment Prompt** (line 419)
   ```python
   augmented_prompt = self._augment_prompt_with_context(prompt, context)
   ```
   - Creates final prompt:
     ```
     You are provided with relevant information from a chemical compounds database...
     
     CONTEXT INFORMATION:
     [Source 1 (Relevance: 0.923)]
     Pyridoxine (vitamin B6)...
     
     USER QUESTION: Look at the molecular structure diagram...
     
     IMPORTANT: The context above contains the information needed...
     ```

5. **Clean up temp image** (line 423)

#### 3.3 vLLM Generation for Q1
**Location:** Lines 425-468

1. **Format messages** (lines 425-432)
   ```python
   messages = [{
       "role": "user",
       "content": [
           {"type": "image", "image": image},
           {"type": "text", "text": augmented_prompt}
       ]
   }]
   ```

2. **Apply chat template** (line 433)
   ```python
   templated = self.processor.apply_chat_template(messages, add_generation_prompt=True)
   ```
   - Formats according to model's chat template (Gemma3/Qwen format)

3. **Prepare multimodal input** (lines 438-441)
   ```python
   multimodal_prompt = {
       "prompt": templated,
       "multi_modal_data": {"image": [image]}
   }
   ```

4. **Generate with vLLM** (line 442)
   ```python
   outputs = self.vllm_llm.generate([multimodal_prompt], self.vllm_sampling_params_q1)
   ```
   - Uses Q1 sampling params: `max_tokens=200`, `temperature=0.7`
   - vLLM processes image + text together

5. **Post-process** (line 445)
   - Extract generated text
   - Clean assistant-only response
   - Truncate if exceeds 600 character limit

6. **Return result** with metadata:
   - `text`: Generated answer
   - `latency_s`: Generation time
   - `rag_used`: True/False
   - `search_type`: "image"
   - `n_chunks_used`: 5
   - `truncated`: True/False

---

### 4. **Q2-Q4 (Text-Based Questions) - Batch Processing**

**Location:** Lines 858-913

#### 4.1 Collect Q2-Q4 Questions
- Extract Q2, Q3, Q4 from each file's `answers[1]`, `answers[2]`, `answers[3]`
- Collect into `text_prompts[]` array
- Track character limits: Q2=1000, Q3=1800, Q4=2000
- Track question indices for sampling params

#### 4.2 RAG Retrieval for Q2-Q4
**Location:** `generate_text_only_batch()` (lines 538-627)

For each text question:
1. **Text Search** (line 566)
   ```python
   chunks = self.search_system.text_search(prompt, n_results=5)
   ```
   - Uses ChromaDB text embedding search
   - Searches for 5 most relevant text chunks
   - Query is the full question text

2. **Build Context** (line 567)
   - Same format as Q1: `[Source 1 (Relevance: 0.923)]...`

3. **Augment Prompt** (line 568)
   ```python
   augmented_prompt = self._augment_prompt_with_context(
       prompt, context, max_chars=max_chars
   )
   ```
   - Includes character limit in instructions
   - Same format as Q1 but with char limit emphasis

#### 4.3 vLLM Generation for Q2-Q4
**Location:** Lines 570-627

1. **Format messages** (lines 570-573)
   ```python
   messages = [{
       "role": "user",
       "content": [{"type": "text", "text": augmented_prompt}]
   }]
   ```

2. **Apply chat template** (line 574)

3. **Select sampling params** (lines 576-584)
   - Q2: `max_tokens=333`
   - Q3: `max_tokens=600`
   - Q4: `max_tokens=666`

4. **Generate with vLLM** (line 586)
   ```python
   outputs = self.vllm_llm.generate([templated], sampling_params)
   ```
   - Text-only generation (no images)

5. **Post-process** (lines 589-595)
   - Extract and clean text
   - Truncate if exceeds character limit

6. **Return result** with same metadata as Q1

---

### 5. **Answer Assembly and Saving**

**Location:** Lines 915-1000

For each file in batch:
1. **Assemble Q1 answer** (lines 922-942)
   - Use generated Q1 result if available
   - Fallback to original answer if generation failed
   - Add metadata: `rag_used`, `search_type`, `n_chunks_used`, `char_limit`

2. **Assemble Q2-Q4 answers** (lines 944-970)
   - Use generated results for Q2, Q3, Q4
   - Fallback to original if generation failed
   - Add same metadata

3. **Update answer data** (lines 972-1000)
   - Set `model`: "gemma3" or "qwen"
   - Set `regenerated_at`: timestamp
   - Set `regenerated_with`: "gemma3+rag"
   - Update character limits and max tokens
   - Set `concise_mode`: true

4. **Save to output directory** (lines 1001-1010)
   - Write to `output_dir/*__answers.json`
   - Same filename as input file
   - JSON format with indentation

---

## Key Differences: Q1 vs Q2-Q4

| Aspect | Q1 (Image-Based) | Q2-Q4 (Text-Based) |
|--------|------------------|---------------------|
| **Input** | Image + Text prompt | Text prompt only |
| **RAG Search** | Image similarity search | Text embedding search |
| **vLLM Input** | Multimodal (image + text) | Text only |
| **Max Tokens** | 200 | 333/600/666 |
| **Char Limit** | 600 | 1000/1800/2000 |
| **Search Type** | "image" | "text" |

---

## RAG Context Format

Both Q1 and Q2-Q4 use the same context format:

```
[Source 1 (Relevance: 0.923)]
Pyridoxine (vitamin B6) is a water-soluble vitamin that plays a crucial role in amino acid metabolism...

[Source 2 (Relevance: 0.891)]
Vitamin B6 exists in several forms including pyridoxine, pyridoxal, and pyridoxamine...

[Source 3 (Relevance: 0.856)]
...
```

Relevance scores are converted from ChromaDB distances: `score = 1 - distance`

---

## Prompt Augmentation Details

**Function:** `_augment_prompt_with_context()` (lines 266-298)

**Structure:**
1. Base prompt with context
2. User question
3. Explicit instruction to use context
4. Character limit instructions
5. Concise mode instructions

**Key Instruction Added:**
> "IMPORTANT: The context above contains the information needed to answer the question. Please use the information from the context to provide your answer. If the information is in the context, use it. Do not say the information is not available if it appears in the context above."

This was added specifically to help Gemma3 use context more reliably.

---

## Error Handling

- **RAG failures:** Continue without RAG context (warning logged)
- **Generation failures:** Fallback to original answer
- **File load failures:** Skip file, increment failed counter
- **Batch failures:** Fallback to individual processing

---

## Performance Optimizations

1. **Batch Processing:** Process multiple files simultaneously
2. **Separate Q1 and Q2-Q4 batching:** Optimize GPU usage
3. **Temporary image cleanup:** Remove temp files after use
4. **Character limit enforcement:** Prevent excessive generation
5. **Token limits:** Match character limits to reduce truncation

---

## Output File Structure

```json
{
  "source_file": "138_Pyridoxine.json",
  "model": "gemma3",
  "image_used_for_q1": true,
  "answers": [
    {
      "question": "...",
      "answer": "...",
      "latency_s": 22.56,
      "rag_used": true,
      "search_type": "image",
      "n_chunks_used": 5,
      "char_limit": 600
    },
    ...
  ],
  "regenerated_at": "2025-11-27 16:44:01",
  "regenerated_with": "gemma3+rag",
  ...
}
```

