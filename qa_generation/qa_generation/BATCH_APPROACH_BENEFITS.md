# Batch Approach: Sending comprehensive_text Once for All 3 Questions

## Overview

Instead of sending `comprehensive_text` 3 times (once per question), we send it **once** and ask all 3 questions together in a single API call.

## Benefits

### 1. Token Usage Reduction: ~65% Savings

**Current Approach (3 separate calls):**
- Call 1: System prompt + Q2 + comprehensive_text = ~32,050 tokens
- Call 2: System prompt + Q3 + comprehensive_text = ~32,050 tokens  
- Call 3: System prompt + Q4 + comprehensive_text = ~32,050 tokens
- **Total: ~96,150 tokens**

**Batch Approach (1 call):**
- Single call: System prompt + all 3 questions + comprehensive_text = ~33,166 tokens
- **Total: ~33,166 tokens**

**Savings: 62,884 tokens (65.5% reduction)**

### 2. API Call Reduction: 66.7% Fewer Calls

- **Current**: 3 API calls per file
- **Batch**: 1 API call per file
- **Reduction**: 2 fewer calls per file (66.7% reduction)

### 3. Rate Limit Impact

**For Acetylsalicylic Acid (124K chars):**
- **Current**: 320.5% of rate limit (exceeds by 3.2x)
- **Batch**: 110.6% of rate limit (exceeds by 1.1x)
- **Improvement**: Much closer to fitting within limit

**For average files (5K chars):**
- **Current**: ~10-15% of rate limit per file
- **Batch**: ~3-5% of rate limit per file
- **Result**: Can process many more files per minute

### 4. Processing Speed

- **Current**: 3 API calls × ~2-3 seconds each = ~6-9 seconds per file
- **Batch**: 1 API call × ~3-4 seconds = ~3-4 seconds per file
- **Speedup**: ~2x faster processing

### 5. Cost Reduction

- **Current**: 3 API calls per file
- **Batch**: 1 API call per file
- **Cost savings**: ~66% reduction in API calls (though token cost is similar)

## Implementation

### New Script
`generate_qa_pairs_comprehensive_update_batch.py`

### Key Changes

1. **Single API Call**: Sends all 3 questions together
2. **JSON Response Format**: Forces structured JSON output
3. **Batch Processing**: Processes Q2, Q3, Q4 in one go
4. **Max Tokens**: 1500 (3 answers × 500 tokens each)

### System Prompt
Modified to request JSON format with 3 answers:
```json
{
  "answer_1": "Answer to question 1",
  "answer_2": "Answer to question 2", 
  "answer_3": "Answer to question 3"
}
```

### User Prompt Format
```
Please answer the following 3 questions about {compound_name} based on the comprehensive information provided below.

Question 1: {Q2}
Question 2: {Q3}
Question 3: {Q4}

Comprehensive Compound Information for {compound_name}:
{comprehensive_text}

Provide answers to all 3 questions based on the comprehensive information above.
```

## Token Usage Examples

### Small File (3K chars, ~800 tokens)
- **Current**: 3 × (200 + 800) = 3,000 tokens
- **Batch**: 200 + 100 + 800 = 1,100 tokens
- **Savings**: 1,900 tokens (63%)

### Medium File (10K chars, ~2,600 tokens)
- **Current**: 3 × (200 + 2,600) = 8,400 tokens
- **Batch**: 200 + 100 + 2,600 = 2,900 tokens
- **Savings**: 5,500 tokens (65%)

### Large File (124K chars, ~32K tokens) - Acetylsalicylic Acid
- **Current**: 3 × (200 + 32,000) = 96,600 tokens
- **Batch**: 200 + 100 + 32,000 = 32,300 tokens
- **Savings**: 64,300 tokens (66%)

## Rate Limit Analysis

### Current Approach
- 178 files × 3 calls = 534 API calls
- Average: ~96K tokens per large file
- Many files exceed rate limit

### Batch Approach
- 178 files × 1 call = 178 API calls
- Average: ~33K tokens per large file
- Fewer rate limit issues
- Can process ~3-4 large files per minute (vs 1 file per 3 minutes)

## Recommendations

1. ✅ **Use batch approach** for all future updates
2. ✅ **Retry failed files** using batch approach
3. ✅ **Monitor token usage** to ensure it stays within limits
4. ✅ **Add exponential backoff** for any remaining rate limit errors

## Testing

Test with a few files first:
```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate
cd ../qa_generation
python generate_qa_pairs_comprehensive_update_batch.py
```

Then process all files if test is successful.

