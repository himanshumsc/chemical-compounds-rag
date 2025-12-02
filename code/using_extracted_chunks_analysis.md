# Using Extracted Chunks as Input to Qwen and Gemma-3

## Overview

The extracted chunks from the filtered JSON files can be used as input to both Qwen and Gemma-3 models. This document explains what data is available and how it can be utilized.

## What You Have

### 1. `rag_chunks` Array
Each answer in the filtered JSON files contains a `rag_chunks` array with structured chunk objects:

```json
{
  "id": "chunk_0",
  "score": 0.7254985570907593,
  "text": "The actual chunk text content...",
  "metadata": {
    "text_content": "...",
    "chunk_id": "chunk_0",
    "image_path": "",
    "has_image": false,
    "page_range": "1-1",
    "text_length": 162
  },
  "search_type": "text"
}
```

**Key Fields:**
- `text`: The actual chunk content (the information retrieved)
- `score`: Relevance score (0.0-1.0, higher = more relevant)
- `metadata`: Additional information (page range, chunk_id, text_length, etc.)

### 2. `rag_context_formatted` String
A pre-formatted string in the exact same format that was sent to the model during generation:

```
[Source 1 (Relevance: 0.725)]
{chunk text}

[Source 2 (Relevance: 0.724)]
{chunk text}

[Source 3 (Relevance: 0.721)]
{chunk text}
...
```

This is the **exact context** that was provided to Gemma-3 when it generated the answer.

## How You Can Use These Chunks

### 1. Regenerate Answers with Same Context
- **Purpose**: Feed the same `rag_context_formatted` to Qwen or Gemma-3
- **Use Case**: Compare how each model handles the exact same context
- **Benefit**: Verify if the information is actually present in the chunks

**Example Workflow:**
1. Extract `rag_context_formatted` from a filtered JSON file
2. Use the same question from that file
3. Feed both to Qwen (or Gemma-3 again)
4. Compare the responses

### 2. Test Model Behavior
- **Purpose**: Determine if Gemma-3 was correct in saying "information not found"
- **Use Case**: Check if the chunks actually contain the answer
- **Benefit**: Identify whether the issue is:
  - Model not finding information that exists
  - Model correctly identifying missing information
  - Model misunderstanding the context

**Example:**
For the naphthalene question where Gemma-3 said "naphthalene is not mentioned in the provided context":
- Extract the `rag_context_formatted` from the JSON
- Manually search the chunks for "naphthalene"
- Feed the same context to Qwen with the same question
- Compare responses

### 3. Controlled Experiments
- **Purpose**: Same question + same chunks → different models
- **Use Case**: Isolate whether differences come from:
  - Model interpretation capabilities
  - Prompt understanding
  - Context utilization strategies
- **Benefit**: Fair comparison between models using identical RAG context

**Experimental Setup:**
```
Question: "What is the chemical formula and molecular weight of naphthalene?"
Context: [Same rag_context_formatted from JSON]
Model A: Qwen → Answer A
Model B: Gemma-3 → Answer B
Compare: Answer A vs Answer B
```

### 4. Debugging and Analysis
- **Purpose**: Understand what was retrieved vs what the model claimed was missing
- **Use Case**: Identify if the issue is:
  - **Retrieval Quality**: Wrong chunks retrieved (low relevance scores)
  - **Model Behavior**: Model not using provided context effectively
  - **Prompt Effectiveness**: Prompt not guiding model to use context
- **Benefit**: Pinpoint the root cause of "missing information" responses

**Debugging Checklist:**
- [ ] Check if answer is actually in `rag_chunks[].text`
- [ ] Verify relevance scores (are chunks actually relevant?)
- [ ] Compare chunk content with model's answer
- [ ] Test same chunks with different prompt formulations
- [ ] Test same chunks with different models

## Example Use Case: Naphthalene Analysis

**Scenario:**
- Question: "What is the chemical formula and molecular weight of naphthalene?"
- Gemma-3 Answer: "I am sorry, but naphthalene is not mentioned in the provided context..."
- Chunks Retrieved: 5 chunks with scores 0.725, 0.724, 0.721, 0.739, 0.738

**Analysis Steps:**
1. Extract `rag_context_formatted` from `/home/himanshu/dev/output/gemma3_rag_concise_missing_ans/97_Naphthalene__answers.json`
2. Search chunks for "naphthalene", "C10H8", "128.17" (formula/weight)
3. If found: Gemma-3 failed to use context → prompt/model issue
4. If not found: Retrieval issue → ChromaDB search problem
5. Feed same context to Qwen → compare responses

## Implementation Notes

### Accessing the Data
```python
import json

# Load filtered JSON
with open('97_Naphthalene__answers.json', 'r') as f:
    data = json.load(f)

# Get Q2 answer (example)
q2_answer = data['answers'][1]  # Index 1 = Q2

# Access chunks
chunks = q2_answer['rag_chunks']  # Array of chunk objects
formatted_context = q2_answer['rag_context_formatted']  # Ready-to-use string

# Use formatted_context directly in prompt
prompt = f"""Based on the following chemical compounds database information, please answer the user's question.

CONTEXT:
{formatted_context}

USER QUESTION: {q2_answer['question']}"""
```

### Key Observations
- **Q1 chunks are empty**: Q1 (image-based) chunks were skipped as requested
- **Q2-Q4 have chunks**: All text-based questions have extracted chunks
- **Same format as original**: `rag_context_formatted` matches what was sent to models
- **Metadata preserved**: Full chunk details available for analysis

## Benefits of This Approach

1. **Reproducibility**: Exact same context can be reused
2. **Fair Comparison**: Models tested with identical inputs
3. **Debugging**: Can verify if information exists in chunks
4. **Analysis**: Understand model behavior differences
5. **Experimentation**: Test different prompts with same context

## Next Steps

1. **Verify Chunk Quality**: Check if retrieved chunks actually contain answers
2. **Compare Models**: Feed same chunks to Qwen and Gemma-3
3. **Analyze Patterns**: Identify common issues in chunk retrieval or model usage
4. **Improve Prompts**: Use findings to refine prompt engineering
5. **Optimize Retrieval**: Adjust ChromaDB search if chunks are irrelevant

