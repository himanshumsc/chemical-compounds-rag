# Root Cause Analysis: 17 Failed Files

## Root Cause: OpenAI API Rate Limiting (HTTP 429)

All 17 failed files encountered **OpenAI API rate limit errors** during processing.

### Error Details

**Error Type**: `429 Too Many Requests`  
**Error Code**: `rate_limit_exceeded`  
**Limit Type**: Tokens Per Minute (TPM)  
**Rate Limit**: 30,000 tokens per minute  
**Issue**: Script exceeded the rate limit during processing

### Example Error Messages

```
Error code: 429 - Rate limit reached for gpt-4o in organization 
org-v2azWyowjxBzM4GMyOFwwNov on tokens per min (TPM): 
Limit 30000, Used 24900, Requested 5219. 
Please try again in 238ms.
```

### Failed Files Analysis

All failed files have:
- ✅ Valid compound files
- ✅ `comprehensive_text` present
- ✅ File matching successful
- ❌ Hit rate limit during API call

### Failed Files List

1. **113_Petroleum.json**
   - Comprehensive text length: 19,637 chars
   - Error: Rate limit hit on Q2

2. **140_Riboflavin.json**
   - Comprehensive text length: 6,560 chars
   - Error: Rate limit hit on Q3 (after Q2 succeeded)

3. **151_Sodium_Cyclamate.json**
   - Comprehensive text length: 3,245 chars
   - Error: Rate limit hit on Q3 (after Q2 succeeded)

4. **154_Sodium_Hypochlorite.json**
   - Comprehensive text length: 7,181 chars
   - Error: Rate limit hit on Q2

5. **156_Sodium_Phosphate.json**
   - Comprehensive text length: 5,985 chars
   - Error: Rate limit hit on Q2

6. **172_Toluene.json**
   - Comprehensive text length: 9,459 chars
   - Error: Rate limit hit on Q2

7. **22_Benzene.json**
   - Comprehensive text length: 15,478 chars
   - Error: Rate limit hit on Q2

8. **32_Calcium_Oxide.json**
   - Comprehensive text length: 6,621 chars
   - Error: Rate limit hit on Q2

9. **40_Cellulose.json**
   - Comprehensive text length: 7,316 chars
   - Error: Rate limit hit on Q3 (after Q2 succeeded)

10. **44_Chlorophyll.json**
    - Comprehensive text length: 3,673 chars
    - Error: Rate limit hit on Q3 (after Q2 succeeded)

(And 7 more files with similar rate limit errors)

## Why Rate Limits Were Hit

### Contributing Factors:

1. **High Request Volume**
   - 534 API calls total (178 files × 3 questions)
   - Processing 178 files sequentially
   - 0.3s delay between questions, 0.5s between files

2. **Large comprehensive_text**
   - Some files have very large comprehensive_text (e.g., Petroleum: 19,637 chars)
   - Larger text = more tokens in request
   - More tokens = faster rate limit consumption

3. **Rate Limit Window**
   - OpenAI tracks tokens per minute in a rolling window
   - When processing many files quickly, the rolling window fills up
   - Even with delays, the cumulative token usage exceeded 30,000 TPM

4. **Retry Logic**
   - Script attempted retries when hitting rate limits
   - Retries also consumed tokens, making the situation worse
   - Some retries happened too quickly (within milliseconds)

## Current Script Behavior

The script has:
- ✅ 0.3s delay between questions
- ✅ 0.5s delay between files
- ❌ No exponential backoff for rate limits
- ❌ No token usage tracking
- ❌ No adaptive rate limiting

## Solution Options

### Option 1: Add Exponential Backoff (Recommended)
- Implement exponential backoff when hitting rate limits
- Wait longer between retries (e.g., 60 seconds, then 120 seconds)
- Track token usage and slow down when approaching limits

### Option 2: Increase Delays
- Increase delay between files from 0.5s to 2-3 seconds
- Increase delay between questions from 0.3s to 1 second
- This will slow down processing but reduce rate limit hits

### Option 3: Batch Processing with Pauses
- Process files in batches (e.g., 50 files)
- Pause for 1 minute between batches
- Allows rate limit window to reset

### Option 4: Retry Failed Files
- Create a retry script for the 17 failed files
- Run it after rate limits have reset
- Add better rate limit handling

## Recommendation

**Immediate Action**: Retry the 17 failed files with improved rate limit handling:
1. Add exponential backoff (wait 60s, then 120s on retries)
2. Track token usage per minute
3. Add adaptive delays when approaching limits

**Long-term**: Implement token usage tracking and adaptive rate limiting for future runs.

## Impact

- **Success Rate**: 90.5% (161/178 files)
- **Failed Files**: 17 files (9.5%)
- **Data Quality**: All failed files have valid data - just need to retry with better rate limit handling
- **No Data Issues**: All compound files exist and have comprehensive_text

