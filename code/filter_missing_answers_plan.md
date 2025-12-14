# Plan: Filter Missing Answers and Extract Chunks

## Objective
Filter QA sets from `dev/output/gemma3_rag_concise` that contain answers indicating missing information, copy them to `dev/output/gemma3_rag_concise_missing_ans`, and extract the ChromaDB chunks that were used for each answer.

## Steps

### 1. Filter Criteria
Check all answers (Q1-Q4) in each JSON file for phrases (case-insensitive, partial match):
- "text does not contain information"
- "information is not present in the provided text"
- "provided documents do not contain"
- "not mentioned in the provided context"
- "not found in provided"
- "not available in the provided"
- "not present in the provided"
- "not mentioned in provided"
- "not found in provided sources"
- "not available in provided context"

**Action**: If ANY answer matches, copy the entire JSON file.

### 2. File Operations
- Source: `dev/output/gemma3_rag_concise/*.json`
- Target: `dev/output/gemma3_rag_concise_missing_ans/`
- Create target directory if it doesn't exist
- Copy matching JSON files (preserve filename)

### 3. Extract Chunks from ChromaDB
For each copied JSON file:

#### 3.1 Load Original QA File
- Read `source_file` from JSON
- Read `source_input_dir` from JSON (e.g., `/home/himanshu/dev/output/qwen_regenerated`)
- Construct path: `{source_input_dir}/{source_file}`
- Load JSON to get `image_path`

#### 3.2 Extract Chunks for Each Answer
- **Q1 (image-based)**:
  - Use `image_similarity_search(image_path, n_results=5)`
  - Format: `[Source 1 (Relevance: 0.923)]\n{chunk_text}`
  
- **Q2-Q4 (text-based)**:
  - Use `text_search(question_text, n_results=5)`
  - Format: `[Source 1 (Relevance: 0.923)]\n{chunk_text}`

#### 3.3 Add Chunks to JSON
For each answer, add:
```json
{
  "question": "...",
  "answer": "...",
  "rag_chunks": [
    {
      "id": "...",
      "text": "...",
      "score": 0.923,
      "metadata": {...}
    },
    ...
  ],
  "rag_context_formatted": "[Source 1 (Relevance: 0.923)]\n...\n\n[Source 2 (Relevance: 0.912)]\n..."
}
```

### 4. Implementation Details

#### 4.1 ChromaDB Setup
- Path: `/home/himanshu/dev/data/chromadb` (DEFAULT_CHROMADB_PATH)
- Collection: `chemical_compounds_multimodal`
- Use `ChromaDBSearchEngine` from `chromadb_search.py`

#### 4.2 Chunk Formatting
- Use same logic as `_build_context_from_chunks()` in `multimodal_qa_runner_vllm.py`
- Format: `[Source {i+1} (Relevance: {score:.3f})]\n{text}`
- Join multiple chunks with `\n\n`

#### 4.3 Error Handling
- If original QA file not found, skip chunk extraction for Q1
- If ChromaDB search fails, log error and continue
- If image_path missing, skip Q1 chunk extraction

### 5. Output Structure
Each filtered JSON will have:
- Original structure preserved
- New field `rag_chunks_extracted: true`
- Each answer will have:
  - `rag_chunks`: List of chunk dicts (id, text, score, metadata)
  - `rag_context_formatted`: Formatted context string (same as sent to model)

### 6. Logging
- Count of files scanned
- Count of files matching filter
- Count of files with chunks extracted
- Errors/warnings for missing files or failed searches

