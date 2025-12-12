# Generation Settings Comparison: QWEN vs OpenAI

## Summary

Both QWEN and OpenAI used **similar generation settings**, with the main difference being token limits and model-specific parameters.

---

## QWEN Generation Settings

**File**: `/home/himanshu/dev/code/multimodal_qa_runner.py`  
**Model**: Qwen2.5-VL-AWQ (local)

### Settings Used

| Parameter | Value | Location |
|-----------|-------|----------|
| **Temperature** | `0.7` | Line 121, 172 |
| **do_sample** | `True` | Line 120, 171 |
| **max_new_tokens** | `128` (default) | Line 119, 170, 453 |
| **pad_token_id** | `processor.tokenizer.eos_token_id` | Line 122, 173 |
| **use_cache** | `True` (default) | Not explicitly set |

### Code Reference

```python
# Line 117-123: Single generation
outputs = self.model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,  # 128
    do_sample=True,
    temperature=0.7,
    pad_token_id=self.processor.tokenizer.eos_token_id,
)

# Line 168-174: Batch generation (same settings)
outputs = self.model.generate(
    **proc,
    max_new_tokens=max_new_tokens,  # 128
    do_sample=True,
    temperature=0.7,
    pad_token_id=self.processor.tokenizer.eos_token_id,
)
```

### Additional QWEN Settings

- **Model Type**: Vision-Language (multimodal)
- **Device**: Auto (CUDA if available)
- **Dtype**: `torch.float16`
- **Chat Template**: Applied via `processor.apply_chat_template()`
- **Image Processing**: Images converted to base64 and included in messages

---

## OpenAI Generation Settings

**File**: `/home/himanshu/dev/code/openai_qa_updater.py`  
**Model**: GPT-4o (via API)

### Settings Used

| Parameter | Value | Location |
|-----------|-------|----------|
| **Temperature** | `0.7` | Line 152 |
| **max_tokens** | `300` | Line 151 |
| **Model** | `gpt-4o` | Line 113 |
| **System Prompt** | Custom (length guidance) | Line 108-110 |

### Code Reference

```python
# Line 112-120: Payload construction
payload = {
    "model": "gpt-4o",  # Using GPT-4o for vision support
    "messages": [
        {"role": "system", "content": system_prompt},
        *messages  # User message with image + text or text only
    ],
    "max_tokens": 300,  # Roughly 300 tokens ≈ 600-750 characters
    "temperature": 0.7
}
```

### System Prompt

```python
system_prompt = f"""You are a helpful assistant answering questions about chemical compounds. 
Provide a clear, informative answer that is between {MIN_ANSWER_LENGTH} and {MAX_ANSWER_LENGTH} characters long.
Be concise but comprehensive."""
```

Where:
- `MIN_ANSWER_LENGTH = 300`
- `MAX_ANSWER_LENGTH = 600`

### Additional OpenAI Settings

- **Image Format**: Base64 encoded PNG
- **Image URL Format**: `data:image/png;base64,{base64_image}`
- **Multimodal Support**: GPT-4o with vision capabilities
- **Answer Length Control**: System prompt guides 300-600 character range
- **Truncation**: Manual truncation if answer exceeds 600 chars

---

## Side-by-Side Comparison

| Setting | QWEN | OpenAI | Notes |
|---------|------|--------|-------|
| **Temperature** | 0.7 | 0.7 | ✅ Same |
| **Sampling** | `do_sample=True` | N/A (always samples) | ✅ Similar behavior |
| **Token Limit** | 128 tokens | 300 tokens | ⚠️ OpenAI allows more |
| **Character Limit** | ~400-500 chars | 300-600 chars | ⚠️ Different ranges |
| **Model** | Qwen2.5-VL-AWQ | GPT-4o | Different models |
| **System Prompt** | None | Yes (length guidance) | ⚠️ OpenAI has guidance |
| **Image Format** | PIL Image → Tensor | Base64 PNG | Different formats |
| **Chat Template** | Yes (QWEN format) | Yes (OpenAI format) | Model-specific |

---

## Detailed Parameter Explanations

### Temperature: 0.7 (Both Models)

**What it means:**
- Controls randomness in generation
- **0.0**: Deterministic (always same output)
- **0.7**: Moderate creativity (used by both)
- **1.0+**: High randomness

**Impact:**
- Both models use same temperature → similar randomness level
- Answers will have some variation but not too random
- Good balance between consistency and diversity

### Token Limits

**QWEN**: 128 tokens
- Roughly 400-500 characters
- Answers often truncated
- Explains incomplete sentences

**OpenAI**: 300 tokens
- Roughly 600-750 characters (but constrained to 300-600 chars)
- More room for complete answers
- System prompt guides length

### Sampling

**QWEN**: `do_sample=True`
- Enables probabilistic sampling
- Uses temperature for randomness
- Not deterministic

**OpenAI**: Always samples (default behavior)
- Uses temperature for randomness
- Similar behavior to QWEN

---

## Impact on Answer Quality

### Temperature (0.7) - Same for Both
- ✅ **Fair comparison**: Same randomness level
- ✅ **Consistent**: Both models have similar variation
- ✅ **Appropriate**: Good balance for QA tasks

### Token Limits - Different
- ⚠️ **QWEN**: 128 tokens may be too restrictive
- ⚠️ **OpenAI**: 300 tokens allows more complete answers
- ⚠️ **Unfair comparison**: QWEN truncated, OpenAI complete

### System Prompt - OpenAI Only
- ⚠️ **OpenAI advantage**: Length guidance helps consistency
- ⚠️ **QWEN disadvantage**: No length guidance
- ⚠️ **Impact**: OpenAI answers more consistent in length

---

## Recommendations for Fair Comparison

### Option 1: Match Token Limits
```bash
# Re-run QWEN with higher token limit
python multimodal_qa_runner.py \
  --model qwen \
  --batched \
  --total-sets 178 \
  --max-new-tokens 300  # Match OpenAI's 300 tokens
```

### Option 2: Match Character Limits
- Set QWEN to generate ~300-600 characters
- Or truncate both to same length for comparison

### Option 3: Add System Prompt to QWEN
- Add length guidance to QWEN prompts
- Make both models aware of desired answer length

---

## Code Locations

### QWEN Settings
- **File**: `multimodal_qa_runner.py`
- **Lines**: 56-66 (Phi4), 88-130 (QWEN), 132-189 (QWEN batch)
- **Default**: Line 453 (`--max-new-tokens` default=128)

### OpenAI Settings
- **File**: `openai_qa_updater.py`
- **Lines**: 112-120 (payload construction)
- **System Prompt**: Lines 108-110

---

## Summary Table

| Aspect | QWEN | OpenAI |
|--------|------|--------|
| **Temperature** | 0.7 | 0.7 ✅ |
| **Sampling** | do_sample=True | Yes (default) ✅ |
| **Token Limit** | 128 | 300 ⚠️ |
| **Character Target** | None | 300-600 ⚠️ |
| **System Prompt** | No | Yes ⚠️ |
| **Image Support** | Yes | Yes ✅ |
| **Model** | Qwen2.5-VL-AWQ | GPT-4o |

**Key Finding**: Both use **temperature=0.7**, but QWEN has **lower token limit (128 vs 300)** and **no system prompt guidance**, which may have affected answer completeness and BLEU scores.

---

**Document Created**: 2025-01-07

