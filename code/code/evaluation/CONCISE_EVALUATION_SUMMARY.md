# Evaluation Summary: QWEN RAG Concise vs OpenAI Baseline

**Date:** November 23, 2025  
**Total Question-Answer Pairs:** 712 (178 compounds × 4 questions)  
**QWEN RAG Concise Source:** `/home/himanshu/dev/output/qwen_rag_concise`  
**OpenAI Baseline Source:** `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive`

**Configuration:**
- Character Limits: Q1=600, Q2=1000, Q3=1800, Q4=2000
- Max Tokens: Q1=200, Q2=333, Q3=600, Q4=666 (division by 3.0)
- Prompt Instruction: "brief, concise, and to the point"

---

## 1. BLEU Scores

BLEU scores measure n-gram overlap between candidate (QWEN RAG Concise) and reference (OpenAI) answers.

### Overall Corpus BLEU Scores

| Metric | Mean | Median | Std | Range |
|--------|------|--------|-----|-------|
| **BLEU-1** | 0.2469 | 0.1867 | 0.2210 | [0.0000, 1.0000] |
| **BLEU-2** | 0.1741 | 0.1104 | 0.1893 | [0.0000, 1.0000] |
| **BLEU-3** | 0.1235 | 0.0622 | 0.1677 | [0.0000, 1.0000] |
| **BLEU-4** | 0.0914 | 0.0353 | 0.1525 | [0.0000, 1.0000] |

### Per-Question-Type BLEU-4 Scores

| Question Type | Mean BLEU-4 | Median BLEU-4 | Count |
|---------------|-------------|---------------|-------|
| **Q1** (Image-based identification) | 0.0317 | 0.0194 | 178 |
| **Q2** (Formula/Type) | **0.2168** | 0.0974 | 178 |
| **Q3** (Production process) | 0.0759 | 0.0494 | 178 |
| **Q4** (Uses/Hazards) | 0.0413 | 0.0252 | 178 |

**Key Observations:**
- **Q2 shows the highest BLEU-4 score (0.2168)**, indicating excellent n-gram overlap for factual formula/type questions. This is a **significant improvement** over the previous RAG version (0.0267).
- Q1 has the lowest BLEU-4 (0.0317), but this is expected as concise answers may omit some details present in longer reference answers.
- Overall BLEU-4 (0.0914) is **higher than the previous RAG version (0.0532)**, indicating better alignment with reference answers.

---

## 2. BERTScore (Semantic Similarity)

BERTScore measures semantic similarity using contextual embeddings.

### Overall Corpus BERTScore

| Metric | Mean | Median | Std | Range |
|--------|------|--------|-----|-------|
| **Precision** | 0.5039 | 0.4637 | 0.1984 | [-0.3919, 1.0000] |
| **Recall** | 0.3150 | 0.2412 | 0.2499 | [-0.4533, 1.0000] |
| **F1** | **0.4022** | 0.3398 | 0.2191 | [-0.2882, 1.0000] |

**Key Improvement:** Overall F1 (0.4022) is **significantly higher** than the previous RAG version (0.1319), indicating much better semantic similarity.

### Per-Question-Type BERTScore F1

| Question Type | Mean F1 | Median F1 | Mean Precision | Mean Recall | Count |
|---------------|---------|------------|----------------|-------------|-------|
| **Q1** (Image-based) | 0.3201 | 0.3172 | 0.4733 | 0.1937 | 178 |
| **Q2** (Formula/Type) | **0.6777** | 0.7199 | **0.7364** | 0.6257 | 178 |
| **Q3** (Production) | 0.3253 | 0.3043 | 0.4172 | 0.2454 | 178 |
| **Q4** (Uses/Hazards) | 0.2856 | 0.2858 | 0.3885 | 0.1952 | 178 |

**Key Observations:**
- **Q2 shows exceptional performance (F1: 0.6777)**, with high precision (0.7364) and recall (0.6257). This is a **massive improvement** over the previous RAG version (F1: 0.0203).
- Q1, Q3, and Q4 show moderate F1 scores (0.28-0.33), which are **significantly better** than the previous RAG version.
- High precision across all questions indicates concise answers contain relevant information without excessive irrelevant content.

---

## 3. Answer Length Comparison

### Overall Statistics

| Model | Mean (chars) | Median (chars) | Min (chars) | Max (chars) | Mean (tokens) |
|-------|-------------|----------------|-------------|-------------|---------------|
| **QWEN Concise** | 218.7 | 200.0 | 11 | 697 | 32.6 |
| **OpenAI Baseline** | 522.2 | 510.5 | 80 | 2,251 | 79.2 |

### Length Ratios

- **Mean characters ratio:** 0.54x (QWEN Concise is ~54% of OpenAI length)
- **Mean tokens ratio:** 0.52x (QWEN Concise is ~52% of OpenAI length)

**Key Observations:**
- QWEN Concise answers are **shorter** than OpenAI baseline (0.54x), which is expected for concise mode.
- This is a **significant reduction** from the previous RAG version (6.32x longer).
- Answers are well within character limits, with mean length (218.7 chars) well below the highest limit (2000 for Q4).

---

## 4. Comparison: Concise vs Previous RAG

### Overall BLEU Score Comparison

| Metric | Previous RAG | Concise | Improvement |
|--------|--------------|---------|-------------|
| **BLEU-1** | 0.1827 | 0.2469 | +35.1% |
| **BLEU-2** | 0.1173 | 0.1741 | +48.4% |
| **BLEU-3** | 0.0798 | 0.1235 | +54.8% |
| **BLEU-4** | 0.0532 | 0.0914 | +71.8% |

### Overall BERTScore Comparison

| Metric | Previous RAG | Concise | Improvement |
|--------|--------------|---------|-------------|
| **Precision** | -0.0004 | 0.5039 | **Massive** |
| **Recall** | 0.3110 | 0.3150 | +1.3% |
| **F1** | 0.1319 | 0.4022 | **+204.9%** |

### Overall Length Comparison

| Metric | Previous RAG | Concise | Change |
|--------|--------------|---------|--------|
| **Mean chars** | 1,738.5 | 218.7 | -87.4% |
| **Length ratio** | 6.32x | 0.54x | -91.5% |

---

### Per-Question-Type Comparison

#### Q1: Image-based Identification

| Metric | Previous RAG | Concise | Change |
|--------|--------------|---------|--------|
| **BLEU-4** | 0.0872 | 0.0317 | -63.6% |
| **BERTScore F1** | 0.2305 | 0.3201 | +38.9% |
| **BERTScore Precision** | 0.1196 | 0.4733 | +295.4% |
| **BERTScore Recall** | 0.3611 | 0.1937 | -46.4% |
| **Mean chars** | 1,195.3 | 190.1 | -84.1% |
| **Length ratio** | 2.30x | 0.36x | -84.3% |

**Analysis:**
- **BLEU-4 decreased** (0.0872 → 0.0317) because concise answers omit some visual details present in longer reference answers. The shorter format reduces n-gram overlap.
- **BERTScore F1 improved** (0.2305 → 0.3201), indicating better semantic alignment despite lower n-gram overlap. This suggests concise answers capture key concepts more effectively.
- **Precision significantly improved** (0.1196 → 0.4733), showing concise answers are more focused and relevant without excessive elaboration.
- **Recall decreased** (0.3611 → 0.1937), expected as concise answers cover less information, but the precision gain compensates for this.
- **Length reduced dramatically** (1,195.3 → 190.1 chars, 2.30x → 0.36x ratio), making answers much more concise while maintaining semantic quality.

---

#### Q2: Formula/Type

| Metric | Previous RAG | Concise | Change |
|--------|--------------|---------|--------|
| **BLEU-4** | 0.0267 | **0.2168** | **+711.6%** |
| **BERTScore F1** | 0.0203 | **0.6777** | **+3,238.4%** |
| **BERTScore Precision** | -0.1815 | **0.7364** | **Massive** |
| **BERTScore Recall** | 0.3442 | 0.6257 | +81.8% |
| **Mean chars** | 1,516.5 | 79.1 | -94.8% |
| **Length ratio** | 14.67x | 0.67x | -95.4% |

**Analysis:**
- **Exceptional improvement** across all metrics for Q2 - this is the biggest success story.
- **BLEU-4 increased dramatically** (0.0267 → 0.2168), indicating much better n-gram alignment. The concise format matches OpenAI's concise factual answers perfectly.
- **BERTScore F1 increased massively** (0.0203 → 0.6777), showing excellent semantic similarity. This is the highest F1 score across all question types.
- **Precision improved from negative** (-0.1815) to very high (0.7364), indicating concise answers are highly relevant and focused.
- **Recall improved** (0.3442 → 0.6257), showing concise answers capture most relevant information despite being shorter.
- **Length reduced dramatically** (1,516.5 → 79.1 chars, 14.67x → 0.67x ratio), eliminating the massive over-elaboration issue from the previous RAG version.

---

#### Q3: Production Process

| Metric | Previous RAG | Concise | Change |
|--------|--------------|---------|--------|
| **BLEU-4** | 0.0465 | 0.0759 | +63.2% |
| **BERTScore F1** | 0.1244 | 0.3253 | +161.5% |
| **BERTScore Precision** | 0.0037 | 0.4172 | **+11,170.3%** |
| **BERTScore Recall** | 0.2754 | 0.2454 | -10.9% |
| **Mean chars** | 2,090.1 | 271.0 | -87.0% |
| **Length ratio** | 5.22x | 0.44x | -91.6% |

**Analysis:**
- **BLEU-4 improved** (0.0465 → 0.0759), indicating better n-gram alignment. The concise format better matches the reference answer structure.
- **BERTScore F1 improved significantly** (0.1244 → 0.3253), showing better semantic similarity. This is a substantial improvement while maintaining conciseness.
- **Precision improved dramatically** (0.0037 → 0.4172), indicating more focused and relevant answers. The previous RAG version had near-zero precision, showing it included too much irrelevant content.
- **Recall slightly decreased** (0.2754 → 0.2454), expected for concise answers, but the precision gain more than compensates.
- **Length reduced significantly** (2,090.1 → 271.0 chars, 5.22x → 0.44x ratio), making answers much more concise while improving quality metrics.

---

#### Q4: Uses/Hazards

| Metric | Previous RAG | Concise | Change |
|--------|--------------|---------|--------|
| **BLEU-4** | 0.0525 | 0.0413 | -21.3% |
| **BERTScore F1** | 0.1525 | 0.2856 | +87.3% |
| **BERTScore Precision** | 0.0565 | 0.3885 | +587.6% |
| **BERTScore Recall** | 0.2631 | 0.1952 | -25.8% |
| **Mean chars** | 2,152.3 | 334.7 | -84.4% |
| **Length ratio** | 3.08x | 0.40x | -87.0% |

**Analysis:**
- **BLEU-4 slightly decreased** (0.0525 → 0.0413), as concise answers may omit some details present in longer reference answers. However, this is a small decrease.
- **BERTScore F1 improved** (0.1525 → 0.2856), indicating better semantic alignment. The concise format captures key concepts more effectively.
- **Precision improved significantly** (0.0565 → 0.3885), showing more focused answers. The previous RAG version had very low precision, indicating excessive irrelevant content.
- **Recall decreased** (0.2631 → 0.1952), expected as concise answers cover less information, but the precision gain provides better overall quality.
- **Length reduced significantly** (2,152.3 → 334.7 chars, 3.08x → 0.40x ratio), making answers much more concise while improving semantic similarity.

---

### Summary: Per-Question Improvements

| Question | BLEU-4 Change | BERTScore F1 Change | Precision Change | Length Reduction |
|----------|---------------|---------------------|------------------|------------------|
| **Q1** (Image) | -63.6% | +38.9% | +295.4% | -84.1% |
| **Q2** (Formula) | **+711.6%** | **+3,238.4%** | **Massive** | -94.8% |
| **Q3** (Production) | +63.2% | +161.5% | **+11,170.3%** | -87.0% |
| **Q4** (Uses/Hazards) | -21.3% | +87.3% | +587.6% | -84.4% |

**Key Insights:**
- **Q2 shows the most dramatic improvements** across all metrics, especially BLEU-4 and BERTScore F1.
- **Q3 shows strong improvements** in BERTScore F1 and precision, with BLEU-4 also improving.
- **Q1 and Q4** show mixed results: BERTScore F1 and precision improve significantly, but BLEU-4 decreases (expected for concise answers).
- **All questions show massive length reductions** (84-95%), making answers much more practical while improving semantic quality.

---

## 5. Key Findings

### Strengths of QWEN RAG Concise

1. **Excellent Q2 Performance:** Q2 (formula/type) shows exceptional BLEU-4 (0.2168) and BERTScore F1 (0.6777), indicating concise answers are highly aligned with reference answers for factual questions.

2. **Improved Precision:** High precision (0.5039) indicates concise answers contain relevant information without excessive irrelevant content, addressing a key weakness of the previous RAG version.

3. **Better Alignment:** Overall BLEU-4 (0.0914) and BERTScore F1 (0.4022) are significantly higher than the previous RAG version, indicating better alignment with reference answers.

4. **Appropriate Length:** Answers are concise (0.54x of reference length) while maintaining semantic similarity, making them more suitable for practical applications.

### Areas for Improvement

1. **Q1 Performance:** Q1 (image-based) shows lower BLEU-4 (0.0317) and moderate BERTScore F1 (0.3201), suggesting concise answers may omit some visual details present in reference answers.

2. **Recall:** Lower recall (0.3150) compared to previous RAG (0.3110) indicates concise answers may miss some information present in reference answers, though this is expected for concise mode.

3. **Q3-Q4:** Moderate performance for Q3 (production) and Q4 (uses/hazards) suggests room for improvement in balancing conciseness with completeness.

---

## 6. Recommendations

1. **Q2 Optimization:** Current Q2 performance is excellent; maintain current approach.

2. **Q1 Enhancement:** Consider slightly increasing Q1 character limit (e.g., 700-800 chars) to capture more visual details while maintaining conciseness.

3. **Q3-Q4 Balance:** Review RAG chunk selection and prompt engineering for Q3-Q4 to improve balance between conciseness and completeness.

4. **Evaluation Metrics:** BERTScore is more appropriate for evaluating concise answers due to semantic similarity focus, while BLEU may penalize concise answers that use different phrasing.

---

## 7. Files Generated

### Detailed Results
- **BLEU per-question:** `/home/himanshu/dev/code/evaluation/results/per_question_bleu_concise.csv`
- **BERTScore per-question:** `/home/himanshu/dev/code/evaluation/results/per_question_bertscore_concise.csv`

### Summary Statistics
- **BLEU summary:** `/home/himanshu/dev/code/evaluation/results/summary_metrics_concise.json`
- **BERTScore summary:** `/home/himanshu/dev/code/evaluation/results/summary_bertscore_concise.json`

---

## 8. Conclusion

QWEN RAG Concise produces significantly shorter, more aligned answers compared to both the previous RAG version and OpenAI baseline. The concise mode shows **exceptional performance for Q2 (formula/type) questions** with BLEU-4 of 0.2168 and BERTScore F1 of 0.6777. Overall, the concise version achieves **much better alignment** (BLEU-4: 0.0914, BERTScore F1: 0.4022) compared to the previous RAG version while maintaining appropriate length (0.54x of reference).

The evaluation suggests QWEN RAG Concise is highly effective for factual questions and provides a good balance between conciseness and semantic similarity for practical applications.

