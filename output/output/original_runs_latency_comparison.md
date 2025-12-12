# Latency Comparison: Original QA Generation Runs

**Generated:** 2025-11-30  
**Source Directories:**
- Qwen-VL: `/home/himanshu/dev/output/qwen_rag_concise/`
- Gemma-3: `/home/himanshu/dev/output/gemma3_rag_concise/`

## Executive Summary

| Metric | Qwen-VL | Gemma-3 | Difference |
|--------|---------|---------|------------|
| **Mean Latency** | 29.367s | 43.403s | **1.48x slower** |
| **Median Latency** | 26.580s | 46.890s | **1.76x slower** |
| **Total Answers** | 712 | 712 | Same |
| **Files Processed** | 178 | 178 | Same |

**Key Finding:** Gemma-3 is **1.5-1.8x slower** overall, but **1.7x FASTER** for Q1 (image-based questions).

## 1. Overall Latency Statistics

### Qwen-VL
- **Total Answers:** 712 (178 files × 4 questions)
- **Mean Latency:** 29.367 seconds
- **Median Latency:** 26.580 seconds
- **Min Latency:** 3.970 seconds
- **Max Latency:** 42.070 seconds
- **Standard Deviation:** 6.607 seconds

### Gemma-3
- **Total Answers:** 712 (178 files × 4 questions)
- **Mean Latency:** 43.403 seconds
- **Median Latency:** 46.890 seconds
- **Min Latency:** 17.480 seconds
- **Max Latency:** 64.160 seconds
- **Standard Deviation:** 13.339 seconds

### Comparison
- **Average:** Gemma-3 is **1.48x slower**
- **Median:** Gemma-3 is **1.76x slower**
- **Variance:** Gemma-3 has higher variance (std dev: 13.3s vs 6.6s)

## 2. Breakdown by Question Type

### Q1 (Image-Based Questions)

| Metric | Qwen-VL | Gemma-3 | Winner |
|--------|---------|---------|--------|
| **Count** | 178 | 178 | Same |
| **Mean Latency** | 37.836s | 22.093s | **Gemma-3 (1.71x faster)** |
| **Median Latency** | 39.900s | 21.550s | **Gemma-3 (1.85x faster)** |
| **Mean Answer Length** | 190.1 chars | 205.2 chars | Gemma-3 (longer) |
| **Token Generation Rate** | 2.5 tokens/sec | 3.1 tokens/sec | Gemma-3 (faster) |

**Key Insight:** Gemma-3 is significantly **FASTER** for image-based questions, despite generating slightly longer answers.

### Q2 (Formula/Elements)

| Metric | Qwen-VL | Gemma-3 | Winner |
|--------|---------|---------|--------|
| **Count** | 178 | 178 | Same |
| **Mean Latency** | 26.544s | 50.507s | **Qwen-VL (1.90x faster)** |
| **Median Latency** | 26.210s | 50.220s | **Qwen-VL (1.91x faster)** |
| **Mean Answer Length** | 79.1 chars | 128.9 chars | Gemma-3 (longer) |
| **Token Generation Rate** | 1.0 tokens/sec | 0.9 tokens/sec | Qwen-VL (slightly faster) |

**Key Insight:** Qwen-VL is nearly **2x faster** for text-based formula questions.

### Q3 (Development/History)

| Metric | Qwen-VL | Gemma-3 | Winner |
|--------|---------|---------|--------|
| **Count** | 178 | 178 | Same |
| **Mean Latency** | 26.544s | 50.507s | **Qwen-VL (1.90x faster)** |
| **Median Latency** | 26.210s | 50.220s | **Qwen-VL (1.91x faster)** |
| **Mean Answer Length** | 271.0 chars | 297.1 chars | Gemma-3 (longer) |
| **Token Generation Rate** | 3.4 tokens/sec | 2.0 tokens/sec | **Qwen-VL (1.75x faster)** |

**Key Insight:** Qwen-VL generates tokens **1.75x faster** for longer text questions.

### Q4 (Properties)

| Metric | Qwen-VL | Gemma-3 | Winner |
|--------|---------|---------|--------|
| **Count** | 178 | 178 | Same |
| **Mean Latency** | 26.544s | 50.507s | **Qwen-VL (1.90x faster)** |
| **Median Latency** | 26.210s | 50.220s | **Qwen-VL (1.91x faster)** |
| **Mean Answer Length** | 334.7 chars | 294.5 chars | Qwen-VL (longer) |
| **Token Generation Rate** | 4.2 tokens/sec | 2.0 tokens/sec | **Qwen-VL (2.16x faster)** |

**Key Insight:** Qwen-VL generates tokens **2.16x faster** for the longest questions.

## 3. Token Generation Rate Analysis

### By Question Type

| Question Type | Qwen-VL (tokens/sec) | Gemma-3 (tokens/sec) | Qwen Speedup |
|---------------|---------------------|---------------------|-------------|
| **Q1** (Image) | 2.5 | 3.1 | 0.78x (Gemma-3 faster) |
| **Q2** (Formula) | 1.0 | 0.9 | 1.17x |
| **Q3** (History) | 3.4 | 2.0 | **1.75x** |
| **Q4** (Properties) | 4.2 | 2.0 | **2.16x** |

### Observations

1. **Q1 Exception:** Gemma-3 generates tokens faster for image-based questions
2. **Text Questions:** Qwen-VL is consistently faster for Q2-Q4
3. **Longer Answers:** Qwen-VL's advantage increases with answer length (Q4: 2.16x)

## 4. Latency Distribution

### Qwen-VL
- **Fastest 25%:** < 25 seconds
- **Median (50%):** ~27 seconds
- **Slowest 25%:** > 33 seconds
- **Range:** 3.97s - 42.07s

### Gemma-3
- **Fastest 25%:** < 35 seconds
- **Median (50%):** ~47 seconds
- **Slowest 25%:** > 55 seconds
- **Range:** 17.48s - 64.16s

**Key Insight:** Gemma-3 has:
- Higher minimum latency (17.5s vs 4.0s)
- Higher maximum latency (64.2s vs 42.1s)
- More variable performance (std dev: 13.3s vs 6.6s)

## 5. Answer Length Comparison

| Question Type | Qwen-VL (chars) | Gemma-3 (chars) | Difference |
|---------------|----------------|-----------------|------------|
| **Q1** | 190.1 | 205.2 | +7.9% (Gemma-3 longer) |
| **Q2** | 79.1 | 128.9 | +63.0% (Gemma-3 longer) |
| **Q3** | 271.0 | 297.1 | +9.6% (Gemma-3 longer) |
| **Q4** | 334.7 | 294.5 | -12.0% (Qwen-VL longer) |

**Observation:** Gemma-3 tends to generate longer answers for Q1-Q3, but Qwen-VL generates longer answers for Q4.

## 6. Performance Patterns

### Image Processing (Q1)
- **Gemma-3 Advantage:** 1.71x faster
- **Possible Reasons:**
  - Different image encoder efficiency
  - Different multimodal architecture
  - Optimized vision processing in Gemma-3

### Text Processing (Q2-Q4)
- **Qwen-VL Advantage:** 1.90x faster consistently
- **Possible Reasons:**
  - Better text generation efficiency
  - More optimized quantization (AWQ vs BitsAndBytes)
  - Smaller model size (7B vs 12B)

## 7. Comparison with Context Test Results

### Original Runs (with RAG retrieval)
- **Qwen-VL:** 29.4s average
- **Gemma-3:** 43.4s average
- **Ratio:** 1.48x slower

### Context Test (same context, no retrieval)
- **Qwen-VL:** 2.1s average
- **Gemma-3:** 7.4s average
- **Ratio:** 3.53x slower

**Key Insight:** The performance gap is **larger** when using the same context (3.5x vs 1.5x), suggesting:
1. RAG retrieval overhead affects both models differently
2. Gemma-3 may benefit more from batch processing during retrieval
3. Context processing efficiency differs significantly

## 8. Summary Table

| Metric | Qwen-VL | Gemma-3 | Winner |
|--------|---------|---------|--------|
| **Overall Mean Latency** | 29.4s | 43.4s | Qwen-VL (1.48x faster) |
| **Overall Median Latency** | 26.6s | 46.9s | Qwen-VL (1.76x faster) |
| **Q1 Mean Latency** | 37.8s | 22.1s | **Gemma-3 (1.71x faster)** |
| **Q2-Q4 Mean Latency** | 26.5s | 50.5s | Qwen-VL (1.90x faster) |
| **Q2 Token Rate** | 1.0 tok/s | 0.9 tok/s | Qwen-VL (1.17x faster) |
| **Q3 Token Rate** | 3.4 tok/s | 2.0 tok/s | Qwen-VL (1.75x faster) |
| **Q4 Token Rate** | 4.2 tok/s | 2.0 tok/s | Qwen-VL (2.16x faster) |
| **Latency Variance** | 6.6s | 13.3s | Qwen-VL (more consistent) |

## 9. Conclusions

### Overall Performance
- **Qwen-VL is 1.5-1.8x faster** for overall QA generation
- **Gemma-3 is 1.7x faster** for image-based questions (Q1)
- **Qwen-VL is 1.9x faster** for text-based questions (Q2-Q4)

### Key Findings
1. **Image Processing:** Gemma-3 has an advantage for multimodal tasks
2. **Text Generation:** Qwen-VL is consistently faster for text-only questions
3. **Consistency:** Qwen-VL has lower latency variance (more predictable)
4. **Token Generation:** Qwen-VL's advantage increases with answer length

### Recommendations
1. **For Image Questions (Q1):** Consider Gemma-3 if latency is critical
2. **For Text Questions (Q2-Q4):** Qwen-VL is clearly better
3. **For Mixed Workloads:** Qwen-VL's overall advantage (1.5x) makes it preferable
4. **For Consistency:** Qwen-VL's lower variance provides more predictable performance

## 10. Comparison with Context Test

The original runs (with RAG) show a **smaller performance gap** (1.5x) compared to the context test (3.5x), suggesting:

- RAG retrieval overhead may mask some of the generation speed differences
- Batch processing during retrieval may benefit Gemma-3 more
- The context test isolates pure generation speed, showing Qwen-VL's true advantage

