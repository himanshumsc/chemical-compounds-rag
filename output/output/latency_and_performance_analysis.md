# Latency and Performance Analysis: Qwen-VL vs Gemma-3

**Generated:** 2025-11-30  
**Test Cases:** 273 cases (Q2-Q4, Q1 skipped)  
**Context:** Same `rag_context_formatted` used for both models

## Executive Summary

| Metric | Qwen-VL | Gemma-3 | Difference |
|--------|---------|---------|------------|
| **Mean Latency** | 2.095s | 7.389s | **3.53x slower** |
| **Median Latency** | 2.052s | 6.986s | **3.40x slower** |
| **Output Tokens/sec** | 33.5 | 9.6 | **3.49x slower** |
| **Input Tokens/sec** | 1241.1 | 360.4 | **3.44x slower** |

**Key Finding:** Gemma-3 is approximately **3.5x slower** than Qwen-VL across all metrics.

## 1. Latency Analysis

### Overall Latency Statistics

#### Qwen-VL
- **Total Cases:** 273
- **Mean Latency:** 2.095 seconds
- **Median Latency:** 2.052 seconds
- **Min Latency:** 0.928 seconds
- **Max Latency:** 4.974 seconds
- **Standard Deviation:** 0.747 seconds

#### Gemma-3
- **Total Cases:** 273
- **Mean Latency:** 7.389 seconds
- **Median Latency:** 6.986 seconds
- **Min Latency:** 2.263 seconds
- **Max Latency:** 20.033 seconds
- **Standard Deviation:** 2.835 seconds

### Latency Comparison

- **Average Speedup:** Gemma-3 is **3.53x slower** than Qwen-VL
- **Median Speedup:** Gemma-3 is **3.40x slower** than Qwen-VL
- **Latency Range:** Gemma-3 has wider variance (std dev: 2.835s vs 0.747s)

### Latency by Question Type

| Question Type | Qwen-VL Mean | Qwen-VL Median | Gemma-3 Mean | Gemma-3 Median | Speedup Factor |
|---------------|-------------|----------------|--------------|----------------|----------------|
| **Q2** (Formula/Elements) | 1.372s | 1.336s | 5.208s | 5.148s | **3.80x slower** |
| **Q3** (Development/History) | 2.285s | 2.319s | 8.190s | 7.675s | **3.58x slower** |
| **Q4** (Properties) | 2.656s | 2.675s | 8.878s | 8.602s | **3.34x slower** |

**Observations:**
- Q2 (shortest answers) shows the largest speedup difference (3.80x)
- Q4 (longest answers) shows the smallest speedup difference (3.34x)
- Gemma-3's latency increases more with answer length than Qwen-VL

## 2. Token Rate Analysis

### Input Token Processing

| Model | Mean Input Tokens | Mean Input Tokens/sec | Speedup |
|-------|-------------------|----------------------|---------|
| **Qwen-VL** | 2,354.2 | 1,241.1 | Baseline |
| **Gemma-3** | 2,354.2 | 360.4 | **3.44x slower** |

**Note:** Both models processed the same input context (same `rag_context_formatted`), so input token counts are identical.

### Output Token Generation

| Model | Mean Output Tokens | Mean Output Tokens/sec | Median Output Tokens/sec | Speedup |
|-------|-------------------|----------------------|-------------------------|---------|
| **Qwen-VL** | 77.7 | 33.5 | 35.8 | Baseline |
| **Gemma-3** | 72.2 | 9.6 | 9.7 | **3.49x slower** |

**Observations:**
- Qwen-VL generates slightly longer answers (77.7 vs 72.2 tokens)
- Qwen-VL's output generation is **3.49x faster** than Gemma-3
- Both models have consistent output rates (median ≈ mean)

### Token Rate Comparison

```
Input Processing Speed:
  Qwen-VL:  1,241.1 tokens/sec  ████████████████████████████████████████
  Gemma-3:    360.4 tokens/sec  ████████████
  
Output Generation Speed:
  Qwen-VL:     33.5 tokens/sec  ████████████████████████████████████████
  Gemma-3:      9.6 tokens/sec  ████████████
```

## 3. Performance Efficiency

### Throughput Analysis

**Qwen-VL:**
- Processes ~1,241 input tokens/second
- Generates ~33.5 output tokens/second
- Total time per request: ~2.1 seconds average

**Gemma-3:**
- Processes ~360 input tokens/second
- Generates ~9.6 output tokens/second
- Total time per request: ~7.4 seconds average

### Efficiency Metrics

| Metric | Qwen-VL | Gemma-3 | Ratio |
|-------|---------|---------|-------|
| **Requests per minute** | ~28.6 | ~8.1 | 3.53x |
| **Output tokens per minute** | ~958 | ~115 | 8.33x |
| **Input tokens per minute** | ~74,466 | ~21,624 | 3.44x |

## 4. Latency Distribution

### Qwen-VL Latency Distribution
- **Fastest 25%:** < 1.5 seconds
- **Median (50%):** ~2.0 seconds
- **Slowest 25%:** > 2.5 seconds
- **95th percentile:** ~3.5 seconds

### Gemma-3 Latency Distribution
- **Fastest 25%:** < 5.5 seconds
- **Median (50%):** ~7.0 seconds
- **Slowest 25%:** > 9.0 seconds
- **95th percentile:** ~13.0 seconds

**Key Insight:** Gemma-3 has higher latency variance, with some requests taking up to 20 seconds.

## 5. Factors Affecting Performance

### Model Architecture
- **Qwen-VL:** 7B parameters, AWQ quantization (4-bit)
- **Gemma-3:** 12B parameters, BitsAndBytes quantization (8-bit)
- **Impact:** Larger model size contributes to Gemma-3's slower performance

### Quantization
- **Qwen-VL:** AWQ (Activation-aware Weight Quantization) - optimized for inference
- **Gemma-3:** BitsAndBytes - more general-purpose quantization
- **Impact:** AWQ may provide better inference speed

### Context Processing
- Both models process identical context (~2,354 tokens)
- Qwen-VL processes context 3.44x faster
- Suggests Qwen-VL has more efficient attention mechanisms

### Output Generation
- Qwen-VL generates answers 3.49x faster
- Both models generate similar length answers
- Qwen-VL's generation speed is more consistent

## 6. Cost-Performance Trade-off

### Performance per Parameter
- **Qwen-VL:** 7B params → 33.5 tokens/sec = **4.8 tokens/sec per billion params**
- **Gemma-3:** 12B params → 9.6 tokens/sec = **0.8 tokens/sec per billion params**

**Qwen-VL is 6x more efficient per parameter.**

### Latency per Success
- **Qwen-VL:** 2.095s / 98.5% success = **2.13s per successful answer**
- **Gemma-3:** 7.389s / 9.5% success = **77.78s per successful answer**

**Qwen-VL is 36.5x more efficient when considering success rate.**

## 7. Recommendations

### For Production Use
1. **Qwen-VL is recommended** for:
   - Lower latency requirements
   - Higher throughput needs
   - Better cost efficiency
   - Superior RAG performance (98.5% vs 9.5% success)

2. **Gemma-3 considerations:**
   - Requires optimization for RAG tasks
   - May benefit from different quantization
   - Needs prompt engineering improvements
   - Currently 3.5x slower with much lower success rate

### For Gemma-3 Optimization
1. **Quantization:** Consider AWQ or other optimized quantization methods
2. **Batch Processing:** May benefit more from batching than Qwen-VL
3. **Model Optimization:** Fine-tuning for RAG tasks could improve both speed and accuracy
4. **Hardware:** May need more GPU memory or different hardware configuration

## 8. Summary Statistics Table

| Metric | Qwen-VL | Gemma-3 | Winner |
|--------|---------|---------|--------|
| **Mean Latency** | 2.095s | 7.389s | Qwen-VL (3.53x faster) |
| **Median Latency** | 2.052s | 6.986s | Qwen-VL (3.40x faster) |
| **Input Tokens/sec** | 1,241.1 | 360.4 | Qwen-VL (3.44x faster) |
| **Output Tokens/sec** | 33.5 | 9.6 | Qwen-VL (3.49x faster) |
| **Success Rate** | 98.5% | 9.5% | Qwen-VL (10.4x better) |
| **Latency per Success** | 2.13s | 77.78s | Qwen-VL (36.5x better) |

## Conclusion

**Qwen-VL significantly outperforms Gemma-3 in all performance metrics:**

1. **3.5x faster** latency
2. **3.5x faster** token generation
3. **10.4x better** success rate
4. **36.5x more efficient** when considering success rate

The performance gap, combined with the accuracy gap (98.5% vs 9.5% success), makes Qwen-VL the clear choice for RAG applications in this use case.

