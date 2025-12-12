# Repeated Text Analysis: Failed Files vs Successful Files

## Key Finding: **NOT Repeated Text - It's File Size!**

### Summary

The 17 failed files do **NOT** have significantly more repeated text. In fact, they have **less repetition** than successful files. The real issue is **file size**.

## Comparison Statistics

### Failed Files (17 files)
- **Average text length**: 14,766 characters
- **Average repetition ratio**: 1.01%
- **Average timeline references**: 88

### Successful Files (sample of 17)
- **Average text length**: 5,026 characters  
- **Average repetition ratio**: 1.38%
- **Average timeline references**: ~30-40 (estimated)

### Difference
- **Length**: Failed files are **+9,740 chars** (almost **3x larger**)
- **Repetition**: Failed files have **-0.36% less repetition**

## Analysis Results

### Repetition Patterns in Failed Files

Most failed files have **minimal repetition**:
- **0% repetition**: 11 out of 17 files (65%)
- **1-2% repetition**: 4 files
- **3-5% repetition**: 2 files (Benzene: 5.56%, Sodium Hypochlorite: 4.35%)

### Top Failed Files by Size

1. **Acetylsalicylic Acid**: 124,356 chars (0.26% repetition)
   - This is an **extreme outlier** - 25x larger than average successful file!
   - 983 timeline references
   - Very low repetition (0.26%)

2. **Petroleum**: 19,637 chars (1.16% repetition)
   - 78 timeline references
   - Minimal repetition

3. **Benzene**: 15,478 chars (5.56% repetition)
   - 103 timeline references
   - Highest repetition ratio among failed files
   - Some repeated sentences about "Gray sticks indicate double bonds"

4. **Ethyl Alcohol**: 14,645 chars (1.52% repetition)
   - 80 timeline references

5. **Toluene**: 9,459 chars (1.61% repetition)
   - 34 timeline references

## Root Cause: File Size, Not Repetition

### Why Large Files Hit Rate Limits

1. **Token Consumption**
   - Larger `comprehensive_text` = more tokens in API request
   - Example: Acetylsalicylic Acid (124K chars) ≈ ~31,000 tokens per request
   - With 3 questions per file, that's ~93,000 tokens for one file
   - Rate limit: 30,000 tokens/minute
   - **Result**: Single large file can consume entire minute's quota

2. **Timeline References**
   - Failed files average 88 timeline references
   - Successful files average ~30-40 timeline references
   - Timeline sections add significant length

3. **Compound Name Frequency**
   - Failed files have higher compound name occurrences
   - Example: Benzene appears 114 times in its comprehensive_text
   - This is normal for comprehensive text, not repetition

## Detailed Findings

### Files with Some Repetition

**Benzene** (5.56% repetition):
- Repeated sentences about molecular structure diagrams
- "Gray sticks indicate double bonds" (3x)
- "Striped sticks indicate a benzene ring" (3x)
- This is legitimate - describing the same diagram structure

**Sodium Hypochlorite** (4.35% repetition):
- Some repeated sentences about bleaching fabrics
- Minimal overall impact

### Files with No Repetition

11 out of 17 failed files have **0% repetition**:
- Acetic acid
- Sodium Cyclamate
- Sodium Phosphate
- Calcium Oxide
- Cellulose
- Chlorophyll
- And 5 more...

## Conclusion

### The Real Problem

**NOT repeated text** - Failed files actually have less repetition (1.01% vs 1.38%)

**YES file size** - Failed files are 3x larger on average:
- Large comprehensive_text = more tokens per API call
- More tokens = faster rate limit consumption
- Rate limit: 30,000 tokens/minute
- Large files consume quota quickly

### Why Rate Limits Were Hit

1. **Large comprehensive_text files** (average 14,766 chars)
2. **Many timeline references** (average 88 per file)
3. **High token consumption per request** (large text = many tokens)
4. **Cumulative effect** - Processing many large files in sequence

### Solution

The issue is **file size**, not repetition. Solutions:

1. **Increase delays** for large files
2. **Track token usage** and slow down when approaching limits
3. **Process large files separately** with longer delays
4. **Implement exponential backoff** for rate limit errors

## Recommendations

1. ✅ **No need to deduplicate** - repetition is not the issue
2. ✅ **Focus on rate limiting** - handle large files better
3. ✅ **Add token usage tracking** - monitor consumption
4. ✅ **Implement adaptive delays** - longer delays for larger files

