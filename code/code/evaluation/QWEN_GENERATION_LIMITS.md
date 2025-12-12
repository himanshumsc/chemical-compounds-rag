# QWEN Answer Generation Limits

## Summary

**YES, there was a limit set in the code for QWEN answer generation.**

### Key Limit Found

**`max_new_tokens = 128`** (default value)

This limit was set in the code and explains why QWEN answers are truncated.

---

## Code Location

**File**: `/home/himanshu/dev/code/multimodal_qa_runner.py`

### Default Limit Setting

```python
# Line 453
parser.add_argument("--max-new-tokens", type=int, default=128)
```

### Where It's Used

The `max_new_tokens=128` parameter is passed to the QWEN model's `generate()` function:

```python
# Line 88-130: QwenWrapper.generate()
def generate(self, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: int = 128):
    # ...
    outputs = self.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # ← 128 tokens limit
        do_sample=True,
        temperature=0.7,
        pad_token_id=self.processor.tokenizer.eos_token_id,
    )
```

---

## Evidence of Truncation

### 1. Answer Length Analysis
- **QWEN average**: 524 characters
- **OpenAI average**: 356 characters
- **Note**: Despite the 128 token limit, QWEN answers are longer because:
  - Tokens ≠ characters (1 token ≈ 3-4 characters typically)
  - 128 tokens ≈ 400-500 characters
  - Some answers hit the limit and are truncated

### 2. Truncated Answers in Files

**Verified**: Many answers are cut off mid-sentence due to the 128 token limit:

**Examples of Truncated Answers**:
- `1_13-Butadiene__answers.json`:
  - Q1: Ends with "...as steam cracking of petroleum" (no period)
  - Q3: Ends with "...temperatures (around 700°C to" (incomplete)
  - Q4: Ends with "...omer (EPDM)**: EPDM is another" (incomplete)
  
- `100_Nicotine__answers.json`:
  - Q1: Ends with "...ing, such as nicotine gums and" (incomplete)
  - Q3: Ends with "...used in the extraction process" (no period)
  - Q4: Ends with "...iation with lung cancer, heart" (incomplete)

- `124_Polyurethane__answers.json`:
  - Q1: Ends with "...ed in insulation materials for" (incomplete)
  - Q2: Ends with "... \text{O}-\text{R} - \text{O}-" (mid-formula)
  - Q3: Ends with "...h of the hydrogen atoms on the" (incomplete)
  - Q4: Ends with "... resistance to deformation. It" (incomplete)

**Pattern**: Answers that don't end with proper punctuation (`.`, `!`, `?`) are likely truncated.

### 3. Summary File Confirmation

From `summary_batched.json`:
```json
{
  "model": "qwen",
  "batched": true,
  "batch_size": 2,
  "total_sets": 178,
  "total_questions": 712
}
```
- All 178 files were processed
- No file limit was set (all files processed)
- But **token limit was 128** for each answer

---

## Impact of the 128 Token Limit

### What It Means

- **128 tokens** ≈ **400-500 characters** (rough estimate)
- Answers are **truncated** when they exceed this limit
- This explains why some answers end mid-sentence

### Why Answers Are Still Long (524 chars average)

1. **Token-to-character ratio**: Not all tokens are single characters
   - Chemical formulas: "C4H6" = 4 tokens
   - Technical terms: "polymerization" = 1 token
   - Average: ~3-4 chars per token

2. **Some answers complete within limit**: Not all answers hit 128 tokens

3. **Truncation happens mid-sentence**: Answers stop at token limit, not character limit

---

## Command Used (Inferred)

Based on the summary file, the command was likely:

```bash
python multimodal_qa_runner.py \
  --model qwen \
  --batched \
  --total-sets 178 \
  --batch-size 2 \
  --max-new-tokens 128  # ← This limit was used (default)
```

**Note**: If `--max-new-tokens` wasn't specified, it defaulted to **128**.

---

## Comparison with OpenAI

| Aspect | QWEN | OpenAI |
|--------|------|--------|
| **Token Limit** | 128 tokens | ~200 tokens (300-600 chars) |
| **Character Limit** | ~400-500 chars (estimated) | 300-600 chars (explicit) |
| **Truncation** | Yes (mid-sentence) | Yes (at word boundary) |
| **Average Length** | 524 chars | 356 chars |

### Why QWEN Answers Are Longer Despite Lower Token Limit?

1. **Different tokenization**: QWEN's tokenizer may tokenize differently
2. **More technical terms**: Chemical formulas and technical terms are tokenized efficiently
3. **Denser content**: More information packed into fewer tokens

---

## Code References

### Default Value
```python
# Line 453 in multimodal_qa_runner.py
parser.add_argument("--max-new-tokens", type=int, default=128)
```

### Function Signature
```python
# Line 88
def generate(self, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: int = 128):
```

### Usage in Generation
```python
# Line 119
outputs = self.model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,  # ← 128 tokens
    do_sample=True,
    temperature=0.7,
    pad_token_id=self.processor.tokenizer.eos_token_id,
)
```

---

## Impact on BLEU Scores

The 128 token limit may have affected BLEU scores:

1. **Truncated answers**: Some QWEN answers are incomplete
2. **Different completeness**: OpenAI answers are complete (within 300-600 char limit)
3. **Comparison fairness**: Comparing complete vs truncated answers may skew results

### Recommendation

For fair comparison, consider:
1. Re-running QWEN with higher `max_new_tokens` (e.g., 256 or 512)
2. Truncating both sets to same length for comparison
3. Analyzing only complete answers (excluding truncated ones)

---

## How to Change the Limit

To generate longer QWEN answers, modify the command:

```bash
python multimodal_qa_runner.py \
  --model qwen \
  --batched \
  --total-sets 178 \
  --batch-size 2 \
  --max-new-tokens 256  # ← Increase from 128 to 256 or higher
```

Or modify the default in code:
```python
# Line 453: Change default
parser.add_argument("--max-new-tokens", type=int, default=256)  # Changed from 128
```

---

## Summary

✅ **Limit Found**: `max_new_tokens = 128` (default)  
✅ **Location**: `multimodal_qa_runner.py` line 453  
✅ **Impact**: Answers are truncated at ~128 tokens (~400-500 chars)  
✅ **Evidence**: Truncated answers in output files, mid-sentence cuts  
✅ **All 178 files processed**: No file limit, only token limit per answer  

**The 128 token limit explains why QWEN answers are sometimes incomplete and may have affected the BLEU score comparison.**

---

**Document Created**: 2025-01-07  
**Code File**: `/home/himanshu/dev/code/multimodal_qa_runner.py`

