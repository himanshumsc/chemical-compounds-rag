# QWEN Answer Regeneration Comparison Report

**Generated:** 1763858213.1550307

**Total Files Compared:** 178
**Total Answers Compared:** 712

## Summary Statistics

| Question | Avg Chars (Orig) | Avg Chars (Regen) | Increase | Avg Words (Orig) | Avg Words (Regen) | Increase | Latency (Orig) | Latency (Regen) | Improvement |
|----------|------------------|-------------------|----------|-----------------|-------------------|----------|----------------|-----------------|-------------|
| Q1 | 549 | 1260 | +711 (129.6%) | 82 | 191 | +108 (131.2%) | 16.68s | 9.49s | 7.19s (43.1%) |
| Q2 | 391 | 712 | +321 (82.1%) | 63 | 114 | +51 (81.7%) | 10.53s | 9.98s | 0.55s (5.2%) |
| Q3 | 573 | 1969 | +1396 (243.6%) | 87 | 296 | +209 (240.2%) | 10.53s | 9.98s | 0.55s (5.2%) |
| Q4 | 583 | 2197 | +1614 (276.8%) | 87 | 329 | +241 (276.0%) | 10.53s | 9.98s | 0.55s (5.2%) |

## Key Findings

1. **Answer Length**: Regenerated answers are significantly longer, indicating more complete responses.
2. **Latency**: vLLM provides faster generation times compared to Transformers.
3. **Truncation**: Regenerated answers with max_tokens=500 are less likely to be truncated.
4. **Completeness**: The regenerated answers appear more complete and comprehensive.
