# Evaluation Summary: QWEN RAG vs OpenAI Baseline

**Date:** November 23, 2025  
**Total Question-Answer Pairs:** 712 (178 compounds × 4 questions)  
**QWEN RAG Source:** `/home/himanshu/dev/output/qwen_rag`  
**OpenAI Baseline Source:** `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive`

---

## 1. BLEU Scores

BLEU scores measure n-gram overlap between candidate (QWEN RAG) and reference (OpenAI) answers.

### Overall Corpus BLEU Scores

| Metric | Mean | Median | Std | Range |
|--------|------|--------|-----|-------|
| **BLEU-1** | 0.1827 | 0.1877 | 0.0999 | [0.0222, 0.5294] |
| **BLEU-2** | 0.1173 | 0.1139 | 0.0665 | [0.0070, 0.3920] |
| **BLEU-3** | 0.0798 | 0.0709 | 0.0503 | [0.0024, 0.3197] |
| **BLEU-4** | 0.0532 | 0.0437 | 0.0398 | [0.0013, 0.2544] |

### Per-Question-Type BLEU-4 Scores

| Question Type | Mean BLEU-4 | Median BLEU-4 | Count |
|---------------|-------------|---------------|-------|
| **Q1** (Image-based identification) | 0.0872 | 0.0787 | 178 |
| **Q2** (Formula/Type) | 0.0267 | 0.0218 | 178 |
| **Q3** (Production process) | 0.0465 | 0.0428 | 178 |
| **Q4** (Uses/Hazards) | 0.0525 | 0.0475 | 178 |

**Key Observations:**
- Q1 (image-based) shows the highest BLEU-4 score (0.0872), indicating better n-gram overlap for visual identification tasks.
- Q2 (formula/type) has the lowest BLEU-4 score (0.0267), suggesting more divergent answers for factual questions.
- Overall BLEU-4 is relatively low (0.0532), which is expected when comparing RAG-augmented comprehensive answers to concise baseline answers.

---

## 2. BERTScore (Semantic Similarity)

BERTScore measures semantic similarity using contextual embeddings, providing a more nuanced evaluation than n-gram overlap.

### Overall Corpus BERTScore

| Metric | Mean | Median | Std | Range |
|--------|------|--------|-----|-------|
| **Precision** | -0.0004 | 0.0173 | 0.1557 | [-0.4287, 0.4551] |
| **Recall** | 0.3110 | 0.2991 | 0.1331 | [-0.1556, 0.7894] |
| **F1** | 0.1319 | 0.1397 | 0.1218 | [-0.2182, 0.4663] |

**Note:** BERTScore values are rescaled with baseline. Negative values indicate lower similarity than the baseline, while positive values indicate higher similarity.

### Per-Question-Type BERTScore F1

| Question Type | Mean F1 | Median F1 | Mean Precision | Mean Recall | Count |
|---------------|---------|------------|----------------|-------------|-------|
| **Q1** (Image-based) | 0.2305 | 0.2260 | 0.1196 | 0.3611 | 178 |
| **Q2** (Formula/Type) | 0.0203 | 0.0087 | -0.1815 | 0.3442 | 178 |
| **Q3** (Production) | 0.1244 | 0.1454 | 0.0037 | 0.2754 | 178 |
| **Q4** (Uses/Hazards) | 0.1525 | 0.1548 | 0.0565 | 0.2631 | 178 |

**Key Observations:**
- Q1 again shows the highest semantic similarity (F1: 0.2305), with strong recall (0.3611), indicating QWEN RAG captures most of the relevant information from OpenAI's answers.
- Q2 has the lowest F1 (0.0203) and negative precision (-0.1815), suggesting QWEN RAG answers are semantically different from OpenAI's concise answers.
- High recall across all questions (0.26-0.36) indicates QWEN RAG answers contain most of the information present in OpenAI answers, but with significant additional content.

---

## 3. Answer Length Comparison

### Overall Statistics

#### Character Length Statistics

| Model | Mean (chars) | Median (chars) | Min (chars) | Max (chars) | Std Dev |
|-------|-------------|----------------|-------------|-------------|---------|
| **QWEN RAG** | 1,738.5 | 1,808.0 | 409 | 2,942 | 526.7 |
| **OpenAI Baseline** | 522.2 | 510.5 | 80 | 2,251 | 359.2 |

#### Token Length Statistics

| Model | Mean (tokens) | Median (tokens) | Min (tokens) | Max (tokens) | Std Dev |
|-------|--------------|-----------------|--------------|--------------|---------|
| **QWEN RAG** | 260.4 | 270.0 | 63 | 419 | 79.0 |
| **OpenAI Baseline** | 79.2 | 79.0 | 9 | 326 | 53.6 |

#### Overall Length Ratios

- **Mean characters ratio:** 6.32x (QWEN RAG is ~6.3x longer)
- **Mean tokens ratio:** 6.18x (QWEN RAG is ~6.2x longer)
- **Median characters ratio:** 3.06x
- **Median tokens ratio:** 3.02x

---

### Per-Question Length Comparison

#### Q1: Image-based Identification (178 pairs)

| Metric | QWEN RAG | OpenAI Baseline | Ratio |
|--------|----------|------------------|-------|
| **Mean (chars)** | 1,195.3 | 524.3 | **2.30x** |
| **Median (chars)** | 1,232.0 | 526.0 | **2.34x** |
| **Min (chars)** | 480 | 409 | 0.80x |
| **Max (chars)** | 1,447 | 602 | 3.17x |
| **Std Dev (chars)** | 161.1 | 50.2 | - |
| **Mean (tokens)** | 180.1 | 80.5 | **2.26x** |
| **Median (tokens)** | 185.0 | 81.0 | **2.30x** |
| **Min (tokens)** | 72 | 62 | 0.80x |
| **Max (tokens)** | 225 | 100 | 3.31x |

**Key Observations:**
- Q1 shows the **lowest length ratio** (2.3x), indicating answers are more aligned in length.
- QWEN RAG provides moderately longer answers for image-based questions.
- Both models have similar ranges, suggesting consistent answer structure for visual identification tasks.

---

#### Q2: Formula/Type (178 pairs)

| Metric | QWEN RAG | OpenAI Baseline | Ratio |
|--------|----------|------------------|-------|
| **Mean (chars)** | 1,516.5 | 118.7 | **14.67x** |
| **Median (chars)** | 1,568.0 | 94.0 | **15.28x** |
| **Min (chars)** | 440 | 80 | 2.02x |
| **Max (chars)** | 2,458 | 1,175 | 26.44x |
| **Std Dev (chars)** | 449.4 | 106.4 | - |
| **Mean (tokens)** | 232.8 | 18.2 | **14.47x** |
| **Median (tokens)** | 245.5 | 15.0 | **15.07x** |
| **Min (tokens)** | 63 | 9 | 2.24x |
| **Max (tokens)** | 385 | 155 | 35.00x |

**Key Observations:**
- Q2 shows the **highest length ratio** (14.67x), indicating massive over-elaboration for simple factual queries.
- OpenAI provides very concise answers (mean: 118.7 chars, 18.2 tokens) for formula/type questions.
- QWEN RAG provides comprehensive answers (mean: 1,516.5 chars, 232.8 tokens) that are much longer than needed.
- This explains the low BLEU and BERTScore for Q2: the answers are semantically different due to length and structure.

---

#### Q3: Production Process (178 pairs)

| Metric | QWEN RAG | OpenAI Baseline | Ratio |
|--------|----------|------------------|-------|
| **Mean (chars)** | 2,090.1 | 609.3 | **5.22x** |
| **Median (chars)** | 2,155.5 | 589.0 | **3.60x** |
| **Min (chars)** | 409 | 83 | 1.08x |
| **Max (chars)** | 2,803 | 1,868 | 27.65x |
| **Std Dev (chars)** | 350.3 | 332.6 | - |
| **Mean (tokens)** | 311.8 | 92.8 | **5.00x** |
| **Median (tokens)** | 322.5 | 90.0 | **3.46x** |
| **Min (tokens)** | 63 | 14 | 1.08x |
| **Max (tokens)** | 400 | 297 | 24.71x |

**Key Observations:**
- Q3 shows a **moderate-high length ratio** (5.22x), indicating comprehensive answers for production questions.
- QWEN RAG provides detailed explanations (mean: 2,090 chars, 311.8 tokens) compared to OpenAI's moderate length (609.3 chars, 92.8 tokens).
- The high ratio suggests QWEN RAG is providing extensive context from RAG chunks about production processes.

---

#### Q4: Uses/Hazards (178 pairs)

| Metric | QWEN RAG | OpenAI Baseline | Ratio |
|--------|----------|------------------|-------|
| **Mean (chars)** | 2,152.3 | 836.6 | **3.08x** |
| **Median (chars)** | 2,229.0 | 801.5 | **2.69x** |
| **Min (chars)** | 791 | 191 | 0.96x |
| **Max (chars)** | 2,942 | 2,251 | 10.16x |
| **Std Dev (chars)** | 347.5 | 350.6 | - |
| **Mean (tokens)** | 316.7 | 125.3 | **3.00x** |
| **Median (tokens)** | 331.5 | 117.5 | **2.62x** |
| **Min (tokens)** | 112 | 27 | 0.95x |
| **Max (tokens)** | 419 | 326 | 11.33x |

**Key Observations:**
- Q4 shows a **moderate length ratio** (3.08x), indicating detailed but reasonable answers.
- QWEN RAG provides comprehensive coverage (mean: 2,152 chars, 316.7 tokens) compared to OpenAI's detailed answers (836.6 chars, 125.3 tokens).
- Both models provide longer answers for Q4, suggesting this question type benefits from comprehensive responses.

---

### Summary of Length Differences

| Question Type | Length Ratio (Mean) | Interpretation |
|---------------|-------------------|----------------|
| **Q1** (Image-based) | 2.30x | ✓ Moderate - answers are reasonably aligned |
| **Q2** (Formula/Type) | **14.67x** | ⚠️ Very high - significant over-elaboration |
| **Q3** (Production) | 5.22x | ⚠️ High - comprehensive but may be excessive |
| **Q4** (Uses/Hazards) | 3.08x | ✓ Moderate - detailed but reasonable |
| **Overall** | 6.32x | ⚠️ High - driven primarily by Q2 |

**Key Findings:**
- Q2 (formula/type) is the primary driver of the high overall length ratio.
- Q1 shows the best length alignment, which correlates with better BLEU and BERTScore.
- Q3 and Q4 show moderate ratios, suggesting comprehensive but potentially excessive detail.
- The length difference explains the lower BLEU scores (longer answers have more n-grams that don't match) but higher recall in BERTScore (more comprehensive coverage).

---

## 4. Correlation Analysis

### BERTScore vs BLEU Correlation

- **BERTScore F1 vs BLEU-4:** 0.7221 (strong positive correlation)
- **BERTScore F1 vs BLEU-1:** 0.7682 (strong positive correlation)

**Interpretation:**
- Strong correlation indicates that both metrics are measuring related aspects of answer quality.
- BLEU-1 correlation is higher than BLEU-4, suggesting that unigram overlap is more aligned with semantic similarity than 4-gram overlap.
- This makes sense given the length difference: longer answers may have different phrasing but similar semantic content.

---

## 5. Key Findings

### Strengths of QWEN RAG

1. **Comprehensive Coverage:** High recall scores (0.26-0.36) indicate QWEN RAG answers contain most information from OpenAI baseline, plus additional context.
2. **Image-Based Questions:** Q1 shows the best performance (BLEU-4: 0.0872, BERTScore F1: 0.2305), suggesting RAG-augmented image understanding is effective.
3. **Detailed Responses:** 6.3x longer answers provide more comprehensive information, which may be valuable for educational or research purposes.

### Areas for Improvement

1. **Precision:** Low/negative precision scores indicate QWEN RAG includes information not present in OpenAI answers, which may be:
   - Additional relevant context (positive)
   - Irrelevant or redundant information (negative)
   - Hallucinated content (negative)

2. **Factual Questions (Q2):** Lowest scores for formula/type questions suggest:
   - Possible over-elaboration on simple factual queries
   - Different answer structure/formatting
   - Need for better precision in concise factual responses

3. **N-gram Overlap:** Low BLEU scores are expected given length differences, but could be improved by:
   - Better alignment with reference answer structure
   - More focused responses for specific question types
   - Balancing comprehensiveness with conciseness

---

## 6. Recommendations

1. **Question-Specific Optimization:**
   - For Q1 (image-based): Current performance is good; maintain current approach.
   - For Q2 (factual): Consider shorter, more focused responses for simple factual queries.
   - For Q3-Q4: Balance comprehensiveness with precision.

2. **RAG Chunk Selection:**
   - Review RAG chunk selection to ensure highest relevance.
   - Consider reducing `n_chunks` for simpler questions (Q2) to improve precision.

3. **Answer Length Control:**
   - Consider implementing question-specific `max_tokens`:
     - Q1: 300 tokens (current) ✓
     - Q2: 100-150 tokens (reduce from 500)
     - Q3-Q4: 500 tokens (current) ✓

4. **Evaluation Metrics:**
   - BERTScore is more appropriate for evaluating RAG outputs due to semantic similarity focus.
   - BLEU is useful for measuring exact overlap but may penalize comprehensive answers.

---

## 7. Files Generated

### Detailed Results
- **BLEU per-question:** `/home/himanshu/dev/code/evaluation/results/per_question_bleu_rag.csv`
- **BERTScore per-question:** `/home/himanshu/dev/code/evaluation/results/per_question_bertscore_rag.csv`

### Summary Statistics
- **BLEU summary:** `/home/himanshu/dev/code/evaluation/results/summary_metrics_rag.json`
- **BERTScore summary:** `/home/himanshu/dev/code/evaluation/results/summary_bertscore_rag.json`

---

## 8. Conclusion

QWEN RAG produces significantly longer, more comprehensive answers compared to OpenAI baseline. While BLEU scores are low due to length differences, BERTScore shows moderate semantic similarity (F1: 0.1319) with strong recall (0.3110), indicating good coverage of reference content. Q1 (image-based) questions show the best performance, while Q2 (factual) questions may benefit from more concise responses.

The evaluation suggests QWEN RAG is effective for comprehensive, educational-style answers but may need optimization for concise factual queries.

