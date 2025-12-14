# Test Qwen-VL with Gemma-3 Context

## Overview

This program tests Qwen-VL using the **exact same context, prompts, and settings** that were used with Gemma-3 for filtered "missing information" answers. This helps determine whether failures are due to:

1. **Model behavior** (MODEL_FAILURE_CONFIRMED): Qwen succeeds where Gemma failed → Gemma failed to use context
2. **Retrieval issues** (RETRIEVAL_FAILURE_CONFIRMED): Both fail → Wrong chunks retrieved

## How It Works

### 1. Load Gemma-3 Filtered Answers
- Reads JSON files from `/home/himanshu/dev/output/gemma3_rag_concise_missing_ans/`
- Extracts answers with `filtered_as_missing_info: true`
- Gets exact `rag_context_formatted` that Gemma-3 saw

### 2. Reconstruct Exact Prompt
- Uses the **exact Gemma-3 prompt template** (explicit version):
  ```
  You are provided with relevant information from a chemical compounds database. Use this information to answer the user's question.

  CONTEXT INFORMATION:
  {context}

  USER QUESTION: {question}

  IMPORTANT: The context above contains the information needed to answer the question. Please use the information from the context to provide your answer. If the information is in the context, use it. Do not say the information is not available if it appears in the context above.
  ```
- Adds character limit instructions (same as Gemma-3)

### 3. Generate with Qwen-VL
- Initializes Qwen-VL wrapper with **RAG disabled** (we provide context directly)
- For Q1: Uses multimodal generation with image
- For Q2-Q4: Uses text-only generation
- Uses same character limits and sampling parameters

### 4. Compare Results
- Checks if Qwen's answer indicates missing information
- Classifies:
  - **MODEL_FAILURE_CONFIRMED**: Gemma failed, Qwen succeeded
  - **RETRIEVAL_FAILURE_CONFIRMED**: Both failed
  - **BOTH_SUCCEEDED**: Edge case (shouldn't happen for filtered answers)

## Usage

### Basic Usage
```bash
cd /home/himanshu/dev
python3 code/test_qwen_with_gemma_context.py
```

### With Options
```bash
# Test with limited cases (for quick testing)
python3 code/test_qwen_with_gemma_context.py --test-limit 10

# Adjust batch size
python3 code/test_qwen_with_gemma_context.py --batch-size 5

# Use different Qwen model
python3 code/test_qwen_with_gemma_context.py --model-path /path/to/qwen
```

## Output

### 1. JSON Results (`comparison_results.json`)
Contains:
- **Summary statistics**: Total tested, classification counts, success rates
- **Detailed results**: Per-answer comparison with Gemma and Qwen answers

### 2. Markdown Report (`comparison_report.md`)
Contains:
- Summary statistics
- Breakdown by question type
- Sample results (first 10 of each classification)

## Output Structure

### Per-Answer Result:
```json
{
  "file": "2_Ibuprofen__answers.json",
  "question_idx": 2,
  "question": "What is the chemical formula...",
  "gemma_answer": "Information not found...",
  "qwen_answer": "Formula: C13H18O2...",
  "qwen_succeeded": true,
  "classification": "MODEL_FAILURE_CONFIRMED",
  "comparison": {
    "gemma_answer_length": 150,
    "qwen_answer_length": 45,
    "gemma_failed": true,
    "qwen_succeeded": true
  }
}
```

## Key Features

1. **Exact Context Replication**: Uses the exact `rag_context_formatted` from Gemma-3 answers
2. **Exact Prompt Template**: Uses the same explicit prompt template used for Gemma-3
3. **Same Settings**: Same character limits, sampling parameters
4. **Image Handling**: Extracts and uses original images for Q1 questions
5. **Quality Checking**: Automatically detects if answers indicate missing information

## Interpretation

### MODEL_FAILURE_CONFIRMED
- **Meaning**: Qwen succeeded with the same context that Gemma failed on
- **Implication**: Gemma failed to use the provided context
- **Action**: Improve Gemma's prompt or model behavior

### RETRIEVAL_FAILURE_CONFIRMED
- **Meaning**: Both models failed with the same context
- **Implication**: The retrieved chunks don't contain the answer
- **Action**: Improve ChromaDB search or query formulation

## Limitations

1. **Answer Quality Detection**: Uses pattern matching to detect "missing information" - may have false positives/negatives
2. **Image Path Extraction**: May not find images for all Q1 questions if paths are missing
3. **Prompt Template**: Assumes the explicit Gemma-3 template was used (may need adjustment if different)

## Example Output

```
Summary:
  Total tested: 150
  Model failures confirmed: 120 (80.0%)
  Retrieval failures confirmed: 30 (20.0%)
  Qwen success rate: 80.0%
```

This indicates that **80% of failures are due to Gemma not using context**, while **20% are due to retrieval issues**.

