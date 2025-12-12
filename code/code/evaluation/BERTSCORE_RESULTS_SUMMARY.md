# BERTScore Results Summary: QWEN vs OpenAI Answers

**Date**: 2025-01-07  
**Total Question-Answer Pairs**: 712  
**Model Used**: `microsoft/deberta-xlarge-mnli`  
**Device**: CUDA (GPU)

---

## Executive Summary

BERTScore analysis reveals **low semantic similarity** between QWEN and OpenAI answers, with an overall F1 score of **0.2148** (mean). This indicates that while both models answer the same questions, they use significantly different wording and semantic content.

### Key Findings

- **Overall BERTScore F1**: 0.2148 (mean), 0.1869 (median)
- **Best performing question type**: Q1 (F1: 0.3171)
- **Strong correlation with BLEU**: 0.80 correlation between BERTScore F1 and BLEU-4
- **Answer length difference**: QWEN answers are ~47% longer on average (524 vs 356 chars)

---

## Overall Corpus Statistics

### BERTScore Metrics

| Metric | Mean | Median | Std Dev | Min | Max |
|--------|------|--------|---------|-----|-----|
| **Precision** | 0.1302 | 0.1108 | 0.1706 | -0.3255 | 0.9557 |
| **Recall** | 0.3155 | 0.3066 | 0.1380 | -0.0873 | 0.9315 |
| **F1** | **0.2148** | **0.1869** | 0.1398 | -0.1227 | 0.9357 |

### Interpretation

- **Precision (0.1302)**: Only ~13% of QWEN answer content is semantically similar to OpenAI answers
- **Recall (0.3155)**: QWEN captures ~32% of the semantic content present in OpenAI answers
- **F1 (0.2148)**: The harmonic mean shows overall low semantic similarity

**Note**: BERTScore uses rescaled baseline, so scores can be negative. Negative scores indicate worse-than-random performance.

---

## Per-Question-Type Analysis

| Question Type | Mean F1 | Median F1 | Mean Precision | Mean Recall | Count |
|---------------|---------|-----------|----------------|-------------|-------|
| **Q1** | **0.3171** | 0.3247 | 0.2753 | 0.3589 | 178 |
| **Q2** | 0.1914 | 0.1371 | 0.0307 | 0.4079 | 178 |
| **Q3** | 0.1695 | 0.1745 | 0.0857 | 0.2619 | 178 |
| **Q4** | 0.1813 | 0.1728 | 0.1289 | 0.2334 | 178 |

### Observations

1. **Q1 performs best** (F1: 0.3171) - likely text-based questions with more structured answers
2. **Q2 has very low precision** (0.0307) but higher recall (0.4079) - QWEN generates different content but covers some similar topics
3. **Q3 and Q4 show similar performance** - both have low semantic similarity
4. **All question types show low F1** - none exceed 0.32, indicating substantial semantic differences

---

## Answer Length Comparison

| Model | Mean Length | Median Length | Min | Max |
|-------|-------------|---------------|-----|-----|
| **QWEN** (candidate) | 524.1 chars | 549.0 chars | 97 | 739 |
| **OpenAI** (reference) | 355.9 chars | 392.0 chars | 73 | 865 |

### Analysis

- QWEN answers are **~47% longer** on average than OpenAI answers
- This length difference may contribute to lower precision scores (more content to match)
- QWEN's `max_new_tokens=128` limit may be causing truncation, but answers are still longer than OpenAI's

---

## Comparison with BLEU Scores

### Correlation Analysis

| Metric Pair | Correlation |
|-------------|-------------|
| BERTScore F1 vs BLEU-4 | **0.8000** |
| BERTScore F1 vs BLEU-1 | **0.7929** |

### Mean Scores Comparison

| Metric | Mean Score |
|--------|------------|
| BERTScore F1 | 0.2148 |
| BLEU-4 | 0.0907 |
| BLEU-1 | 0.3126 |

### Interpretation

1. **Strong correlation (0.80)**: BERTScore and BLEU-4 agree on answer quality ranking
2. **BERTScore F1 > BLEU-4**: BERTScore is more forgiving of paraphrasing (0.21 vs 0.09)
3. **BLEU-1 closer to BERTScore**: Unigram overlap (0.31) is closer to semantic similarity (0.21) than 4-gram overlap
4. **Both metrics indicate low similarity**: Despite different methodologies, both show QWEN and OpenAI answers are quite different

---

## Detailed Interpretation

### What BERTScore Tells Us

1. **Semantic Divergence**: The low F1 score (0.21) indicates that QWEN and OpenAI answers, while addressing the same questions, use fundamentally different semantic content and phrasing.

2. **Recall > Precision**: Higher recall (0.32) than precision (0.13) suggests:
   - QWEN answers contain some relevant information (captured by recall)
   - But QWEN includes much content not present in OpenAI answers (low precision)
   - This aligns with QWEN generating longer, more verbose answers

3. **Question-Type Variation**: Q1's better performance (0.32 F1) suggests:
   - Text-based questions may have more standardized answer formats
   - Image-based questions (Q2-Q4) show more variation in how models interpret and answer

4. **Truncation Impact**: QWEN's `max_new_tokens=128` limit may cause:
   - Incomplete answers (ending mid-sentence)
   - Different stopping points than OpenAI
   - This could artificially lower semantic similarity scores

### Comparison with BLEU

- **BLEU focuses on exact word/phrase matches** → Very low scores (0.09 BLEU-4)
- **BERTScore focuses on semantic meaning** → Still low but higher (0.21 F1)
- **Both agree**: QWEN and OpenAI answers are substantially different
- **Correlation (0.80)**: When one metric says an answer is good/bad, the other generally agrees

---

## Quality Assessment

### Overall Quality Rating: **Low Semantic Similarity**

Based on BERTScore F1 ranges:
- **0.0 - 0.5**: Poor semantic similarity ← **QWEN is here (0.21)**
- **0.5 - 0.7**: Moderate semantic similarity
- **0.7 - 0.85**: Good semantic similarity
- **0.85 - 1.0**: High semantic similarity

### Key Insights

1. **QWEN answers are semantically different** from OpenAI answers, not just differently worded
2. **QWEN may be providing more detailed/verbose answers** (longer length, lower precision)
3. **Both models answer correctly** but use different approaches and content
4. **Q1 (text questions) shows better alignment** than image-based questions

### Potential Reasons for Low Scores

1. **Model differences**: QWEN (local, smaller) vs GPT-4o (cloud, larger)
2. **Generation settings**: Different temperature, max_tokens, prompts
3. **Answer style**: QWEN may be more verbose, OpenAI more concise
4. **Truncation**: QWEN's 128 token limit may cut off answers mid-thought
5. **Training data**: Different training data leads to different answer styles

---

## Recommendations

1. **Increase QWEN's max_new_tokens**: Current limit (128) may be too restrictive
2. **Adjust generation temperature**: Current 0.7 may be too high for consistency
3. **Fine-tune prompts**: Ensure QWEN receives similar instructions to OpenAI
4. **Consider answer length**: Match OpenAI's target length (300-600 chars) for fairer comparison
5. **Evaluate correctness separately**: Low BERTScore doesn't mean wrong answers, just different

---

## Files Generated

1. **`results/per_question_bertscore.csv`**: Detailed BERTScore for all 712 QA pairs
   - Columns: file_base, question_index, question, qwen_answer, openai_answer, bertscore_precision, bertscore_recall, bertscore_f1

2. **`results/summary_bertscore.json`**: Aggregated statistics
   - Overall corpus metrics
   - Per-question-type statistics
   - Per-compound statistics
   - BLEU correlation data

---

## Next Steps

1. Review individual high/low scoring examples to understand patterns
2. Compare with human evaluation if available
3. Experiment with different QWEN generation settings
4. Consider using BERTScore for model fine-tuning feedback

---

**Generated by**: `bertscore_calculator.py`  
**Model**: `microsoft/deberta-xlarge-mnli`  
**Library**: `bert-score` v0.3.12

