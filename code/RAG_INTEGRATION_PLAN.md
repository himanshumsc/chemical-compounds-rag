# Plan: Integrate ChromaDB RAG into multimodal_qa_runner_vllm.py

## Objective
Modify `multimodal_qa_runner_vllm.py` to use ChromaDB RAG for ALL questions:
- **Q1**: Use **image-only similarity search** to find similar molecular structures
  - Model identifies compound from image itself (no name leakage)
  - Image similarity finds chunks with similar molecular structures
  - Provides additional context for properties without revealing compound name
- **Q2-Q4**: Use text search to retrieve relevant compound information

## Current State Analysis

### Image Sanitization ✅
- **Status**: Images ARE already anonymized
- **Method**: `load_image_sanitized()` function (lines 289-300)
  - Re-encodes image in-memory using PNG format
  - Strips all EXIF metadata and filename information
  - Returns clean PIL.Image object
- **Conclusion**: No changes needed for image handling

### Current Flow (multimodal_qa_runner_vllm.py)
- Q1: Direct generation with image (no RAG)
- Q2-Q4: Direct generation with text only (no RAG)
- **No ChromaDB integration currently**

### RAG Flow (from modular_multimodal_rag.py)
1. Retrieve chunks from ChromaDB using `ChromaDBSearchEngine`
2. Build context from retrieved chunks
3. Augment prompt with context
4. Generate with augmented prompt

## Implementation Plan

### Step 1: Add ChromaDB Integration
- Import `ChromaDBSearchEngine` from `chromadb_search`
- Initialize ChromaDB search system in `QwenVLLMWrapper.__init__()`
- Add ChromaDB path parameter (default: `/home/himanshu/dev/data/chromadb`)

### Step 2: Add Context Building Functions
- Create `_build_context_from_chunks()` method similar to `rag_orchestrator._prepare_context_for_generation()`
- Format: `[Source 1 (Relevance: score)]\nchunk_text\n\n[Source 2...]`
- Create `_retrieve_chunks_for_q1()` for hybrid search (text + image)
- Create `_retrieve_chunks_for_text()` for text-only search (Q2-Q4)
- Augment prompt with context for all questions

### Step 3: Modify Generation Flow
- **Q1**: 
  - Use **image-only similarity search** (NOT hybrid, NOT text)
  - Rationale: Model identifies compound from image; we don't want to leak compound name
  - Retrieve chunks using `image_similarity_search()` (default: 5 chunks)
  - Finds chunks with similar molecular structure images
  - Build context from chunks (may contain same compound or related compounds)
  - Augment prompt with context
  - Generate with augmented prompt + image
  - Model identifies compound from image AND uses context for comprehensive properties
- **Q2-Q4**: 
  - Retrieve chunks using `text_search()` (default: 5 chunks)
  - Build context from chunks
  - Augment prompt with context
  - Generate with augmented prompt

### Step 4: Add Configuration Options
- `--n-chunks`: Number of chunks to retrieve (default: 5)
- `--use-rag`: Enable/disable RAG (default: True for Q2-Q4)
- `--chromadb-path`: Path to ChromaDB (default: `/home/himanshu/dev/data/chromadb`)

### Step 5: Update Output Metadata
- Add `rag_used: true/false` to answer metadata
- Add `n_chunks_retrieved` for Q2-Q4
- Add `context_length` (character count)

### Step 6: Regeneration Strategy
- Read existing answer files from `qwen_regenerated/`
- Regenerate ALL questions (Q1-Q4) with RAG context
- Q1: Use hybrid search (text + image similarity)
- Q2-Q4: Use text search
- Update answer files in place
- Add metadata: `rag_regenerated_at`, `n_chunks_used`, `search_type`, `rag_regenerated_at`

## Code Structure Changes

### New Imports Needed
```python
import sys
import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sqlite3  # ChromaDB compatibility
from chromadb_search import ChromaDBSearchEngine
```

### Modified Class: QwenVLLMWrapper
```python
class QwenVLLMWrapper:
    def __init__(self, model_path, chromadb_path, use_rag=True, n_chunks=5):
        # ... existing initialization ...
        
        # Initialize ChromaDB if RAG enabled
        self.use_rag = use_rag
        self.n_chunks = n_chunks
        self.search_system = None
        if use_rag:
            self.search_system = ChromaDBSearchEngine(chromadb_path, device="cpu")
    
    def _build_context_from_chunks(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Source {i+1} (Relevance: {chunk['score']:.3f})]\n{chunk['text']}"
            )
        return "\n\n".join(context_parts)
    
    def _save_temp_image(self, image: Image.Image) -> str:
        """Save PIL image temporarily for ChromaDB search"""
        import tempfile
        temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        image.save(temp_path, format='PNG')
        return temp_path
    
    def _cleanup_temp_image(self, temp_path: str):
        """Clean up temporary image file"""
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def _augment_prompt_with_context(self, query: str, context: str) -> str:
        """Augment user query with RAG context"""
        return f"""Based on the following chemical compounds database information, please answer the user's question comprehensively and accurately.

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a detailed, accurate response based on the provided context. Include relevant chemical formulas, properties, uses, and any safety information when available."""
```

### Modified Generation Flow
```python
# Q1 generation with RAG (image-only similarity search)
if self.use_rag and self.search_system and image:
    # Save image temporarily for ChromaDB search
    temp_image_path = self._save_temp_image(image)
    # Use IMAGE-ONLY similarity search (not hybrid, not text)
    # Model identifies compound from image; we find similar structures
    chunks = self.search_system.image_similarity_search(
        temp_image_path, n_chunks=self.n_chunks
    )
    context = self._build_context_from_chunks(chunks)
    # Augment prompt with context (context may contain compound info)
    augmented_prompt = self._augment_prompt_with_context(question, context)
    # Generate with augmented prompt + image
    # Model identifies compound from image AND uses context for properties
    res = self.generate_with_vision(augmented_prompt, image)
    # Clean up temp image
    self._cleanup_temp_image(temp_image_path)

# Q2-Q4 generation with RAG (text only)
elif self.use_rag and self.search_system:
    # Use text search
    chunks = self.search_system.text_search(question, self.n_chunks)
    context = self._build_context_from_chunks(chunks)
    augmented_prompt = self._augment_prompt_with_context(question, context)
    # Use augmented_prompt instead of question
else:
    # Use original question (fallback)
    augmented_prompt = question
```

## Files to Modify

1. **multimodal_qa_runner_vllm.py**
   - Add ChromaDB integration
   - Add RAG context building
   - Modify Q2-Q4 generation to use RAG
   - Keep Q1 unchanged

## Testing Strategy

1. **Test with 1-3 files first**
   - Verify ChromaDB retrieval works
   - Check context is properly formatted
   - Verify Q1 remains unchanged
   - Verify Q2-Q4 use RAG context

2. **Compare outputs**
   - Check if RAG answers are more comprehensive
   - Verify metadata is saved correctly

3. **Full regeneration**
   - Run for all 178 files
   - Monitor logs for errors
   - Check disk space

## Expected Benefits

1. **Better Answer Quality**: ALL answers will have context from ChromaDB
2. **Q1 Enhancement**: 
   - Image similarity search finds chunks with similar molecular structures
   - Model identifies compound from image (no name leakage)
   - Retrieved context provides additional properties and information
   - May find related compounds with similar structures
3. **Q2-Q4 Enhancement**: Text search retrieves relevant compound information
4. **More Comprehensive**: Answers can reference specific database information
5. **Traceability**: Metadata shows which chunks were used for each question
6. **No Name Leakage**: Q1 uses image-only search, avoiding compound name in search query

## Risks & Considerations

1. **Token Limits**: RAG context adds tokens - may need to adjust `max_tokens`
2. **Latency**: ChromaDB retrieval adds time (should be minimal)
3. **GPU Memory**: ChromaDB uses CPU, shouldn't affect GPU memory
4. **Context Length**: Need to ensure context fits within model limits

## Default Configuration

- **n_chunks**: 5 (same as modular_multimodal_rag.py)
- **Q1 Search type**: **image-only** (image similarity search)
  - Rationale: Model identifies compound from image; image similarity finds similar structures
  - Avoids leaking compound name through text search
  - Finds chunks with similar molecular structure images
- **Q2-Q4 Search type**: text
- **RAG enabled**: True (for all questions)
- **Image handling**: Images are sanitized (metadata removed) before use

## Q1 RAG Strategy Rationale

**Why image-only search for Q1?**
1. Model identifies compound from image itself (no name needed)
2. Question text doesn't contain compound name: "Look at the molecular structure diagram..."
3. Image similarity finds chunks with similar molecular structures
4. Retrieved chunks may contain:
   - Information about the same compound (if image matches)
   - Information about related compounds (if structures are similar)
5. Model uses context to provide more comprehensive properties
6. No compound name leakage through text search

## Next Steps

1. ✅ Plan created
2. ⏳ Review plan
3. ⏳ Implement changes
4. ⏳ Test with sample files
5. ⏳ Run full regeneration

