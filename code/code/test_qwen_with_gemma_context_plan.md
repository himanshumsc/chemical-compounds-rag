# Plan: Test Qwen-VL with Gemma-3 Context

## Objective
Use the exact same `rag_context_formatted`, questions, prompts, and settings from Gemma-3's filtered answers to generate answers with Qwen-VL. This will help determine:
1. **If Qwen succeeds** → Confirms MODEL_FAILURE (Gemma failed to use context)
2. **If Qwen also fails** → Confirms RETRIEVAL_FAILURE (wrong chunks retrieved)

## Input Data
- Filtered JSON files from `/home/himanshu/dev/output/gemma3_rag_concise_missing_ans/`
- Each file contains answers with `filtered_as_missing_info: true`
- Each answer has:
  - `question`: The exact question asked
  - `rag_context_formatted`: The exact context used
  - `char_limit`: Character limit used
  - `question_idx`: Question index (Q2, Q3, or Q4)
  - Image path (for Q1, if applicable)

## Key Requirements

### 1. Use Exact Same Context
- Extract `rag_context_formatted` from Gemma-3 answers
- Use this EXACT context (no re-retrieval from ChromaDB)
- This ensures we're testing with the same chunks that Gemma saw

### 2. Use Exact Same Prompt Template
- The prompt template used for Gemma-3 was:
  ```
  You are provided with relevant information from a chemical compounds database. Use this information to answer the user's question.

  CONTEXT INFORMATION:
  {context}

  USER QUESTION: {question}

  IMPORTANT: The context above contains the information needed to answer the question. Please use the information from the context to provide your answer. If the information is in the context, use it. Do not say the information is not available if it appears in the context above.
  ```
- Use this EXACT template (not the one currently in code)

### 3. Use Same Settings
- Same `char_limit` per question type
- Same `max_tokens` (derived from char_limit)
- Same temperature and other sampling parameters
- Same batch size (if applicable)

### 4. Handle Q1 (Image Questions)
- For Q1, we need the original image path
- Extract from `rag_chunks` or original QA file
- Use same image + context combination

## Implementation Steps

### Step 1: Load Filtered Gemma-3 Answers
- Read JSON files from `gemma3_rag_concise_missing_ans/`
- Filter for answers with `filtered_as_missing_info: true`
- Extract: question, context, char_limit, question_idx, image_path (if Q1)

### Step 2: Reconstruct Exact Prompt
- Use the Gemma-3 prompt template (explicit version)
- Format: `{context}` + `{question}` + instructions
- Add character limit instructions if `char_limit` exists

### Step 3: Generate with Qwen-VL
- Initialize Qwen-VL using `VLLMRagWrapper` (but disable RAG - use provided context)
- For Q1: Use multimodal generation with image
- For Q2-Q4: Use text-only generation
- Use same sampling parameters as Gemma-3 run

### Step 4: Compare Results
- Compare Qwen's answer with Gemma's answer
- Classify:
  - **Qwen succeeds, Gemma failed** → MODEL_FAILURE confirmed
  - **Both fail** → RETRIEVAL_FAILURE confirmed
  - **Both succeed** → Edge case (shouldn't happen for filtered answers)

### Step 5: Generate Report
- Save Qwen's answers alongside Gemma's answers
- Add comparison metadata
- Generate statistics:
  - How many cases Qwen succeeded where Gemma failed
  - How many cases both failed
  - Breakdown by question type

## Output Structure

### Per-Answer Comparison:
```json
{
  "file": "2_Ibuprofen__answers.json",
  "question_idx": 2,
  "question": "...",
  "gemma_answer": "...",
  "gemma_context": "...",
  "qwen_answer": "...",
  "qwen_succeeded": true/false,
  "classification": "MODEL_FAILURE_CONFIRMED" | "RETRIEVAL_FAILURE_CONFIRMED",
  "comparison": {
    "qwen_used_context": true/false,
    "qwen_answer_length": 123,
    "gemma_answer_length": 456
  }
}
```

### Summary Statistics:
- Total tested
- Qwen successes (where Gemma failed)
- Both failures
- Success rate by question type

## Technical Considerations

### 1. Disable RAG in Qwen Wrapper
- We're providing context directly, not retrieving
- Need to modify `VLLMRagWrapper` to accept pre-formatted context
- Or create a simpler wrapper that just uses vLLM with provided context

### 2. Image Handling for Q1
- Need to find original image paths
- May need to check original QA files or `rag_chunks` metadata
- Load images and pass to Qwen-VL

### 3. Prompt Template
- Must use EXACT Gemma-3 template (explicit version)
- Not the current template in `multimodal_qa_runner_vllm.py`
- This is critical for fair comparison

### 4. Sampling Parameters
- Use same parameters as Gemma-3 run
- Check what was used in `multimodal_qa_runner_gemma3.py`

## Files to Create

1. `test_qwen_with_gemma_context.py` - Main test program
2. Output directory: `/home/himanshu/dev/output/qwen_gemma_context_comparison/`
3. Report: `comparison_report.json` and `comparison_report.md`

## Validation

- Spot-check: Manually verify 5-10 cases
- Ensure Qwen is using the exact same context
- Verify prompt template matches Gemma-3's
- Check that image paths are correct for Q1

