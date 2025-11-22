# BLEU Score Evaluation Results: QWEN vs OpenAI

**Date**: 2025-01-07  
**Total Question-Answer Pairs**: 712 (178 compounds × 4 questions)

## Overall Corpus BLEU Scores

| Metric | Mean | Median | Std Dev | Min | Max |
|--------|------|--------|---------|-----|-----|
| **BLEU-1** | 0.3126 | 0.3033 | 0.1345 | 0.0577 | 0.8966 |
| **BLEU-2** | 0.1967 | 0.1692 | 0.1141 | 0.0075 | 0.8517 |
| **BLEU-3** | 0.1344 | 0.1052 | 0.1038 | 0.0040 | 0.8181 |
| **BLEU-4** | 0.0907 | 0.0654 | 0.0931 | 0.0027 | 0.7822 |

### Interpretation
- **BLEU-1 (0.31)**: Moderate unigram overlap - about 31% of words match
- **BLEU-4 (0.09)**: Low 4-gram overlap - answers have different phrasing and structure
- **Overall**: QWEN and OpenAI answers show moderate similarity in word choice but differ significantly in phrasing and structure

## Per-Question-Type Analysis

| Question Type | Mean BLEU-4 | Median BLEU-4 | Count | Notes |
|---------------|-------------|---------------|-------|-------|
| **Q1** (Image-based) | 0.1755 | 0.1658 | 178 | Highest similarity |
| **Q2** (Text-only) | 0.0697 | 0.0245 | 178 | Lower similarity |
| **Q3** (Text-only) | 0.0620 | 0.0560 | 178 | Lower similarity |
| **Q4** (Text-only) | 0.0558 | 0.0491 | 178 | Lowest similarity |

### Key Findings
1. **Q1 (Image-based questions) perform best** with BLEU-4 of 0.18
   - Both models see the same image
   - More consistent identification and description
   
2. **Q2-Q4 (Text-only questions) show lower similarity** (BLEU-4 ~0.06)
   - Different explanation styles
   - QWEN may provide more detailed technical explanations
   - OpenAI may use different phrasing

## Answer Length Comparison

| Model | Mean Length | Median Length | Range |
|-------|-------------|---------------|-------|
| **QWEN** (candidate) | 524.1 chars | 549.0 chars | 97-739 chars |
| **OpenAI** (reference) | 355.9 chars | 392.0 chars | 73-865 chars |

### Observations
- **QWEN answers are longer** on average (524 vs 356 chars)
- QWEN was limited to 128 max_tokens, but answers are still longer
- OpenAI was constrained to 300-600 characters
- Length difference contributes to lower BLEU scores (different content density)

## Statistical Distribution

### BLEU-4 Score Distribution
- **Mean**: 0.0907
- **Median**: 0.0654 (lower than mean = right-skewed distribution)
- **Standard Deviation**: 0.0931
- **Range**: 0.0027 to 0.7822

### Interpretation
- Most answers have low BLEU scores (median 0.065)
- Some pairs show high similarity (max 0.78)
- Wide variation suggests:
  - Some questions have more standardized answers
  - Different compounds may have different answer patterns
  - Models use different explanation approaches

## Key Insights

1. **Moderate Word Overlap**: BLEU-1 of 0.31 indicates about 31% word-level similarity
2. **Different Phrasing**: Low BLEU-4 (0.09) shows answers are phrased very differently
3. **Image Questions Align Better**: Q1 scores are 2-3x higher than text-only questions
4. **Length Mismatch**: QWEN's longer answers contribute to lower n-gram overlap
5. **Both Models Valid**: Low BLEU doesn't mean wrong answers - just different explanations

## Recommendations

1. **Consider Semantic Metrics**: BLEU measures n-gram overlap, not semantic correctness
   - Consider adding ROUGE, METEOR, or BERTScore for semantic evaluation
   
2. **Analyze High/Low Scorers**: 
   - Identify compounds with highest/lowest BLEU scores
   - Understand what makes some pairs more similar

3. **Question-Type Analysis**:
   - Q1's higher scores suggest image-based questions are more standardized
   - Q2-Q4's lower scores suggest more diverse explanation styles

4. **Length Normalization**:
   - Consider comparing content density rather than exact matches
   - Account for QWEN's longer answers in evaluation

## Files Generated

- **`results/per_question_bleu.csv`**: Detailed scores for all 712 pairs
- **`results/summary_metrics.json`**: Aggregated statistics in JSON format

## Next Steps

1. Analyze per-compound BLEU scores to identify patterns
2. Compare high-scoring vs low-scoring pairs qualitatively
3. Consider additional metrics (ROUGE, semantic similarity)
4. Generate visualizations (histograms, box plots)

---

**Conclusion**: QWEN and OpenAI answers show moderate word-level similarity (BLEU-1: 0.31) but differ significantly in phrasing and structure (BLEU-4: 0.09). Image-based questions (Q1) show higher alignment than text-only questions (Q2-Q4).

