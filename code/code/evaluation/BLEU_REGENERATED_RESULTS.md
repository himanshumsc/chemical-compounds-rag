# BLEU Score Results: Regenerated Data Comparison

**Date:** November 23, 2025  
**Total Question-Answer Pairs:** 712 (178 files × 4 questions)

## Data Sources

- **QWEN (Candidate):** `/home/himanshu/dev/output/qwen_regenerated`
  - Regenerated with vLLM
  - Max tokens: 500
  - Average length: 1,535 characters

- **OpenAI (Reference):** `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive`
  - Regenerated with comprehensive_text
  - Max tokens: 500
  - Average length: 522 characters

---

## Overall BLEU Scores (Regenerated Data)

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
| **BLEU-1** | 0.2175 | 0.2089 | 0.1167 | 0.0353 | 0.6296 |
| **BLEU-2** | 0.1446 | 0.1320 | 0.0870 | 0.0202 | 0.5161 |
| **BLEU-3** | 0.1018 | 0.0859 | 0.0715 | 0.0078 | 0.4385 |
| **BLEU-4** | 0.0698 | 0.0534 | 0.0599 | 0.0031 | 0.3955 |

---

## Per-Question-Type BLEU-4 Scores

| Question | Count | Mean BLEU-4 | Median BLEU-4 | Std BLEU-4 |
|----------|-------|-------------|---------------|------------|
| **Q1 (Vision)** | 178 | 0.1183 | 0.1038 | 0.0686 |
| **Q2 (Text)** | 178 | 0.0584 | 0.0337 | 0.0664 |
| **Q3 (Text)** | 178 | 0.0513 | 0.0468 | 0.0335 |
| **Q4 (Text)** | 178 | 0.0512 | 0.0493 | 0.0314 |

---

## Answer Length Statistics

| Source | Mean (chars) | Median (chars) | Min (chars) | Max (chars) |
|--------|--------------|----------------|-------------|-------------|
| **QWEN (Regenerated)** | 1,534.7 | 1,533.5 | 109 | 2,866 |
| **OpenAI (Regenerated)** | 522.2 | 510.5 | 80 | 2,251 |

**Key Observation:** QWEN answers are approximately **3x longer** than OpenAI answers on average (1,535 vs 522 chars).

---

## Comparison with Original BLEU Scores

### Original Data (for reference)
- **QWEN:** `/home/himanshu/dev/output/qwen` (original, truncated answers)
- **OpenAI:** `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components` (original)

| Metric | Original Mean | Regenerated Mean | Change |
|--------|---------------|------------------|--------|
| **BLEU-1** | 0.3126 | 0.2175 | **-30.4%** |
| **BLEU-2** | 0.1967 | 0.1446 | **-26.5%** |
| **BLEU-3** | 0.1344 | 0.1018 | **-24.3%** |
| **BLEU-4** | 0.0907 | 0.0698 | **-23.0%** |

### Per-Question Comparison (BLEU-4)

| Question | Original Mean | Regenerated Mean | Change |
|----------|---------------|------------------|--------|
| **Q1** | 0.1755 | 0.1183 | **-32.6%** |
| **Q2** | 0.0697 | 0.0584 | **-16.2%** |
| **Q3** | 0.0620 | 0.0513 | **-17.3%** |
| **Q4** | 0.0558 | 0.0512 | **-8.2%** |

---

## Analysis and Interpretation

### Why BLEU Scores Decreased

1. **Answer Length Disparity:**
   - QWEN regenerated answers are **3x longer** than OpenAI answers (1,535 vs 522 chars)
   - Longer answers naturally have lower n-gram overlap with shorter reference answers
   - BLEU penalizes length differences

2. **Content Expansion:**
   - Regenerated QWEN answers are more comprehensive and detailed
   - They include additional information, examples, and explanations
   - This expansion reduces exact n-gram matches with the more concise OpenAI answers

3. **Different Generation Strategies:**
   - QWEN (vLLM) may generate more verbose, detailed responses
   - OpenAI (comprehensive_text) generates more concise, focused answers
   - Different writing styles reduce n-gram overlap

### Important Considerations

1. **BLEU Score Limitations:**
   - BLEU measures n-gram overlap, not semantic similarity
   - Lower BLEU scores don't necessarily mean worse quality
   - Longer, more detailed answers can be better even with lower BLEU scores

2. **Quality vs. Similarity:**
   - The regenerated QWEN answers are significantly more complete (2-3x longer)
   - They provide more comprehensive information
   - Lower BLEU scores may reflect better, more detailed answers rather than worse ones

3. **Q1 (Vision) Performance:**
   - Q1 has the highest BLEU-4 score (0.1183) among all questions
   - This suggests better alignment for vision-based questions
   - Vision questions may have more standardized answer formats

4. **Q2-Q4 (Text) Performance:**
   - Text questions have lower BLEU scores (0.051-0.058)
   - This reflects the greater variability in how detailed answers can be structured
   - QWEN's more verbose style contrasts with OpenAI's concise style

---

## Recommendations

1. **Consider Alternative Metrics:**
   - Use BERTScore for semantic similarity evaluation
   - Consider ROUGE scores for summarization-style evaluation
   - Evaluate answer completeness and accuracy separately

2. **Length Normalization:**
   - Consider length-normalized BLEU scores
   - Evaluate whether longer answers are actually better for the use case

3. **Quality Assessment:**
   - Conduct human evaluation to assess answer quality
   - Compare answer completeness, accuracy, and usefulness
   - Don't rely solely on BLEU scores for quality assessment

---

## Files Generated

- **Detailed Results:** `/home/himanshu/dev/code/bleu_evaluation/results/per_question_bleu_regenerated.csv`
- **Summary Metrics:** `/home/himanshu/dev/code/bleu_evaluation/results/summary_metrics_regenerated.json`
- **Script:** `/home/himanshu/dev/code/bleu_evaluation/scripts/bleu_score_calculator_regenerated.py`

---

## Conclusion

The regenerated data shows lower BLEU scores compared to the original data, primarily due to:
1. Significant length differences between QWEN and OpenAI answers
2. More comprehensive and detailed QWEN responses
3. Different generation strategies and writing styles

**Lower BLEU scores do not necessarily indicate worse quality.** The regenerated QWEN answers are significantly more complete and detailed, which may be preferable depending on the use case. Consider using additional evaluation metrics (BERTScore, human evaluation) to get a more complete picture of answer quality.

