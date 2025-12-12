# Plan: Regenerate OpenAI Q2-Q4 Answers with Character Limits

## Objective
Regenerate OpenAI answers for Q2-Q4 with character limits matching Qwen/Gemma, while preserving Q1 answers.

## Requirements

### 1. Character Limits (Matching Qwen/Gemma)
- **Q2:** 1,000 characters
- **Q3:** 1,800 characters  
- **Q4:** 2,000 characters
- **Q1:** Preserved (not regenerated)

### 2. Context Source
- Use **full comprehensive_text** (as before)
- Same system prompt and user prompt structure

### 3. Rate Limit Handling
- If rate limit hit, wait 1 minute and retry
- Maximum retries: 3 attempts
- Log rate limit occurrences

### 4. Output
- Update existing files in `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive`
- Preserve Q1 answers (copy from current)
- Regenerate Q2-Q4 with character limits
- Add metadata indicating character limits were used

## Implementation Plan

### Script: `generate_qa_pairs_comprehensive_update_with_limits.py`

**Key Changes from Original:**
1. Add character limit enforcement (Q2: 1000, Q3: 1800, Q4: 2000)
2. Add truncation logic (similar to Qwen/Gemma)
3. Add rate limit handling with retry logic
4. Update prompt to include character limit instructions
5. Preserve Q1 answers exactly as they are

### Prompt Updates
- **System Prompt:** Changed to "You are a helpful assistant." (matches Qwen/Gemma)
- **User Prompt Structure:** Matches Qwen/Gemma format:
  ```
  Based on the following chemical compounds database information, please answer the user's question.
  
  CONTEXT:
  {comprehensive_text}
  
  USER QUESTION: {question}
  
  IMPORTANT INSTRUCTIONS:
  - Your answer MUST be brief, concise, and to the point
  - Maximum length: {char_limit} characters (strict limit)
  - Focus ONLY on the most essential and relevant information
  - Avoid unnecessary elaboration or repetition
  - Be direct and factual
  - If you exceed {char_limit} characters, your answer will be truncated
  
  Generate a concise answer that fits within {char_limit} characters:
  ```
- Uses comprehensive_text as CONTEXT (instead of RAG chunks)
- Same concise instructions as Qwen/Gemma

### Rate Limit Handling
```python
def regenerate_with_retry(question, compound_name, comprehensive_text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return regenerate_answer(...)
        except RateLimitError:
            if attempt < max_retries - 1:
                wait 60 seconds
                continue
            else:
                raise
```

### Truncation Logic
- If answer exceeds character limit, truncate at sentence boundary
- Add "..." if truncated
- Similar to Qwen/Gemma truncation logic

## Execution
1. Create the script
2. Test on a few files first
3. Run full regeneration in background
4. Monitor progress and rate limits

## Expected Output
- Updated JSON files with:
  - Q1: Preserved (unchanged)
  - Q2-Q4: Regenerated with character limits
  - Metadata: Character limits used, regeneration timestamp

