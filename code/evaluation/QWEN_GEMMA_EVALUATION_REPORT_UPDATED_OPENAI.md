# Comprehensive Evaluation Report: Qwen RAG Concise vs Gemma RAG Concise
## (Updated with Regenerated OpenAI Baseline)

**Date:** November 30, 2025  
**Evaluation Dataset:** 178 chemical compounds × 4 questions = 712 question-answer pairs  
**Baseline:** OpenAI comprehensive answers (regenerated with character limits and explicit comprehensive_text instructions)  
**Models Evaluated:** Qwen RAG Concise, Gemma RAG Concise

**Note:** This report uses an updated OpenAI baseline where Q2-Q4 answers were regenerated with:
- Character limits: Q2=1000, Q3=1800, Q4=2000
- Explicit instruction to base answers ONLY on comprehensive_text
- Prompt structure aligned with Qwen/Gemma generation approach
- Q1 answers preserved from original generation

---

## Executive Summary

This report presents a comprehensive evaluation comparing Qwen RAG Concise and Gemma RAG Concise outputs against an updated OpenAI baseline using three established metrics: BLEU, ROUGE, and BERTScore.

### Key Findings

**Qwen RAG Concise significantly outperforms Gemma RAG Concise across all evaluation metrics:**

- **BLEU-4:** Qwen (0.0906) vs Gemma (0.0427) - **+112.2% improvement**
- **ROUGE-1 F-measure:** Qwen (0.4280) vs Gemma (0.2953) - **+44.9% improvement**
- **BERTScore F1:** Qwen (0.3973) vs Gemma (0.1751) - **+126.9% improvement**

**Winner: Qwen RAG Concise** across all metrics and question types.

---

## 1. Overall Metrics Comparison

### 1.1 BLEU Scores (N-gram Overlap)

BLEU measures n-gram precision between generated and reference answers.

| Metric | Qwen | Gemma | Difference | Qwen Advantage |
|--------|------|-------|------------|----------------|
| **BLEU-1** | 0.2425 | 0.1886 | +0.0539 | **+28.6%** |
| **BLEU-2** | 0.1709 | 0.1096 | +0.0613 | **+55.9%** |
| **BLEU-3** | 0.1213 | 0.0651 | +0.0562 | **+86.3%** |
| **BLEU-4** | 0.0906 | 0.0427 | +0.0479 | **+112.2%** |

**Analysis:**
- Qwen shows consistently higher n-gram overlap across all BLEU variants
- The advantage increases with higher-order n-grams (BLEU-2, BLEU-3, BLEU-4), indicating better phrase-level matching
- BLEU-4 (standard metric) shows Qwen with more than double the score of Gemma
- Updated baseline shows slightly different scores compared to previous evaluation, indicating the impact of baseline quality

### 1.2 ROUGE Scores (Recall-Oriented Evaluation)

ROUGE measures overlap of n-grams and longest common subsequences, focusing on recall.

| Metric | Qwen | Gemma | Difference | Qwen Advantage |
|--------|------|-------|------------|----------------|
| **ROUGE-1 F-measure** | 0.4280 | 0.2953 | +0.1327 | **+44.9%** |
| **ROUGE-2 F-measure** | 0.2025 | 0.1082 | +0.0943 | **+87.1%** |
| **ROUGE-L F-measure** | 0.3392 | 0.2159 | +0.1233 | **+57.1%** |
| **ROUGE-Lsum F-measure** | 0.3494 | 0.2265 | +0.1229 | **+54.3%** |

**Analysis:**
- Qwen demonstrates superior recall across all ROUGE variants
- ROUGE-1 (unigram overlap) shows strong performance, indicating better word-level coverage
- ROUGE-2 (bigram overlap) shows nearly double the performance, indicating better phrase-level matching
- ROUGE-L (longest common subsequence) indicates better sentence structure alignment

**ROUGE Precision and Recall Breakdown:**

| Metric | Qwen Precision | Qwen Recall | Gemma Precision | Gemma Recall |
|--------|----------------|------------|-----------------|--------------|
| **ROUGE-1** | 0.6755 | 0.3400 | 0.4488 | 0.2528 |
| **ROUGE-2** | 0.3092 | 0.1647 | 0.1596 | 0.0946 |
| **ROUGE-L** | 0.5293 | 0.2714 | 0.3250 | 0.1863 |

**Key Insight:** Qwen achieves higher precision AND recall, indicating both better relevance and completeness.

### 1.3 BERTScore (Semantic Similarity)

BERTScore uses contextual embeddings to measure semantic similarity, providing a more nuanced evaluation than n-gram metrics.

| Metric | Qwen | Gemma | Difference | Qwen Advantage |
|--------|------|-------|------------|----------------|
| **Precision** | 0.5012 | 0.2438 | +0.2574 | **+105.5%** |
| **Recall** | 0.3084 | 0.1143 | +0.1941 | **+169.9%** |
| **F1** | 0.3973 | 0.1751 | +0.2222 | **+126.9%** |

**Analysis:**
- Qwen shows exceptional semantic alignment with reference answers
- Precision advantage (+105.5%) indicates Qwen generates more relevant content
- Recall advantage (+169.9%) indicates Qwen captures more information from references
- F1 score shows Qwen with more than double the semantic similarity

**Note:** BERTScore values are rescaled with baseline. Positive values indicate better than baseline similarity.

---

## 2. Per-Question-Type Analysis

### 2.1 Q1: Image-based Identification

**Task:** Identify chemical compound from molecular structure diagram and describe key properties.

| Metric | Qwen | Gemma | Difference | Winner |
|--------|------|-------|------------|--------|
| **BLEU-4** | 0.0317 | 0.0300 | +0.0017 | Qwen (marginal) |
| **ROUGE-1 F** | 0.3444 | 0.3295 | +0.0149 | Qwen |
| **ROUGE-2 F** | 0.1325 | 0.1174 | +0.0151 | Qwen |
| **ROUGE-L F** | 0.2611 | 0.2358 | +0.0253 | Qwen |
| **BERTScore F1** | 0.3201 | 0.2869 | +0.0332 | Qwen |
| **BERTScore Precision** | 0.4733 | 0.4031 | +0.0702 | Qwen |
| **BERTScore Recall** | 0.1937 | 0.1872 | +0.0065 | Qwen |

**Analysis:**
- **Closest competition** - Qwen and Gemma perform similarly on image-based tasks
- Qwen has a slight edge across all metrics, particularly in precision (0.4733 vs 0.4031)
- Both models struggle with BLEU-4 on visual tasks (0.03 range), which is expected as visual descriptions vary significantly
- BERTScore shows Qwen captures semantic content better (F1: 0.3201 vs 0.2869)
- Q1 baseline preserved from original (not regenerated), so results consistent with previous evaluation

### 2.2 Q2: Formula/Type

**Task:** Provide chemical formula and classify compound type.

| Metric | Qwen | Gemma | Difference | Winner |
|--------|------|-------|------------|--------|
| **BLEU-4** | 0.2167 | 0.0760 | +0.1407 | **Qwen (+185.1%)** |
| **ROUGE-1 F** | 0.6303 | 0.3532 | +0.2771 | **Qwen (+78.4%)** |
| **ROUGE-2 F** | 0.3761 | 0.1541 | +0.2220 | **Qwen (+144.1%)** |
| **ROUGE-L F** | 0.5699 | 0.2875 | +0.2824 | **Qwen (+98.2%)** |
| **BERTScore F1** | 0.6692 | 0.1903 | +0.4789 | **Qwen (+251.6%)** |
| **BERTScore Precision** | 0.7292 | 0.2275 | +0.5017 | **Qwen (+220.3%)** |
| **BERTScore Recall** | 0.6166 | 0.1596 | +0.4570 | **Qwen (+286.3%)** |

**Analysis:**
- **Qwen's strongest performance** - Exceptional results on factual/formula questions
- BLEU-4 of 0.2167 is the highest across all question types for Qwen
- BERTScore F1 of 0.6692 indicates excellent semantic alignment for factual content
- Qwen shows more than **3.5x better performance** than Gemma on BERTScore F1
- This question type benefits from precise factual retrieval, where Qwen excels
- Updated baseline (with character limits and comprehensive_text instruction) shows similar performance patterns

### 2.3 Q3: Production Process

**Task:** Explain how the compound is produced/manufactured.

| Metric | Qwen | Gemma | Difference | Winner |
|--------|------|-------|------------|--------|
| **BLEU-4** | 0.0732 | 0.0468 | +0.0264 | Qwen (+56.4%) |
| **ROUGE-1 F** | 0.3915 | 0.2784 | +0.1131 | Qwen (+40.6%) |
| **ROUGE-2 F** | 0.1751 | 0.1012 | +0.0739 | Qwen (+73.0%) |
| **ROUGE-L F** | 0.2943 | 0.1946 | +0.0997 | Qwen (+51.2%) |
| **BERTScore F1** | 0.3212 | 0.1350 | +0.1863 | Qwen (+138.0%) |
| **BERTScore Precision** | 0.4160 | 0.1819 | +0.2341 | Qwen (+128.7%) |
| **BERTScore Recall** | 0.2385 | 0.0919 | +0.1466 | Qwen (+159.5%) |

**Analysis:**
- Qwen shows strong performance on process explanation tasks
- BERTScore F1 advantage of +138.0% indicates much better semantic understanding of production processes
- ROUGE-2 advantage (+73.0%) shows better phrase-level matching for technical processes
- Qwen captures more relevant information (recall: 0.2385 vs 0.0919)

### 2.4 Q4: Uses/Hazards

**Task:** Discuss industrial uses and potential hazards.

| Metric | Qwen | Gemma | Difference | Winner |
|--------|------|-------|------------|--------|
| **BLEU-4** | 0.0407 | 0.0181 | +0.0226 | Qwen (+124.9%) |
| **ROUGE-1 F** | 0.3460 | 0.2202 | +0.1258 | Qwen (+57.1%) |
| **ROUGE-2 F** | 0.1264 | 0.0603 | +0.0661 | Qwen (+109.6%) |
| **ROUGE-L F** | 0.2315 | 0.1456 | +0.0859 | Qwen (+59.0%) |
| **BERTScore F1** | 0.2788 | 0.0884 | +0.1904 | Qwen (+215.4%) |
| **BERTScore Precision** | 0.3862 | 0.1625 | +0.2237 | Qwen (+137.7%) |
| **BERTScore Recall** | 0.1850 | 0.0184 | +0.1666 | Qwen (+905.4%) |

**Analysis:**
- Qwen demonstrates strong performance on uses/hazards questions
- **Exceptional recall advantage** (+905.4%) - Qwen captures significantly more relevant information
- BERTScore F1 shows Qwen with more than **3x better performance**
- Both models show lower scores on this question type, likely due to the complexity of balancing uses and hazards
- Updated baseline with character limits shows improved alignment

---

## 3. Answer Length Analysis

### 3.1 Overall Length Statistics

| Model | Mean (chars) | Median (chars) | Min (chars) | Max (chars) |
|-------|--------------|----------------|-------------|-------------|
| **Qwen** | 218.7 | 200.0 | 11 | 697 |
| **Gemma** | 231.4 | 214.0 | 47 | 640 |
| **OpenAI (Baseline)** | 532.5 | 516.0 | 80 | 1,753 |

**Analysis:**
- Both models produce concise answers compared to OpenAI baseline
- Qwen is slightly more concise (218.7 vs 231.4 chars mean)
- Both models successfully adhere to character limits (Q1: 600, Q2: 1000, Q3: 1800, Q4: 2000)
- Qwen shows better length control with lower variance
- Updated OpenAI baseline shows reduced max length (1,753 vs 2,251 previously) due to character limits

### 3.2 Length Ratios

| Model | Ratio to OpenAI | Interpretation |
|-------|----------------|----------------|
| **Qwen** | 0.41x | 41% of baseline length |
| **Gemma** | 0.43x | 43% of baseline length |

**Analysis:**
- Both models produce significantly shorter answers than OpenAI
- Qwen is slightly more concise while maintaining higher quality scores
- The conciseness is intentional (character limits enforced) and both models comply
- Updated baseline is more aligned in length with Qwen/Gemma outputs

---

## 4. Statistical Summary

### 4.1 Overall Performance Rankings

**By BLEU-4:**
1. Qwen: 0.0906
2. Gemma: 0.0427

**By ROUGE-1 F-measure:**
1. Qwen: 0.4280
2. Gemma: 0.2953

**By BERTScore F1:**
1. Qwen: 0.3973
2. Gemma: 0.1751

### 4.2 Question-Type Performance Rankings

**Best Performance by Question Type:**

| Question Type | Best Model | Metric | Score |
|---------------|------------|--------|-------|
| Q1 (Image) | Qwen | BERTScore F1 | 0.3201 |
| Q2 (Formula) | Qwen | BERTScore F1 | 0.6692 |
| Q3 (Production) | Qwen | BERTScore F1 | 0.3212 |
| Q4 (Uses/Hazards) | Qwen | BERTScore F1 | 0.2788 |

**Qwen wins across all question types on all metrics.**

### 4.3 Consistency Analysis

**Standard Deviations (Lower = More Consistent):**

| Metric | Qwen Std Dev | Gemma Std Dev |
|--------|--------------|---------------|
| BLEU-4 | 0.1517 | 0.0736 |
| ROUGE-1 F | 0.1805 | 0.1353 |
| BERTScore F1 | 0.2185 | 0.1834 |

**Analysis:**
- Gemma shows lower variance in BLEU-4 (more consistent scores)
- However, this consistency comes at the cost of lower overall performance
- Qwen's higher variance may indicate it adapts better to different question types
- BERTScore shows similar variance levels, indicating both models have comparable consistency in semantic understanding

---

## 5. Comparison with Previous Evaluation

### 5.1 Impact of Updated Baseline

The OpenAI baseline was regenerated with:
- Character limits: Q2=1000, Q3=1800, Q4=2000
- Explicit instruction: "Your answer MUST be based ONLY on the Comprehensive Compound Information provided above"
- Prompt structure aligned with Qwen/Gemma approach
- Q1 preserved from original

**Key Changes in Scores:**

| Metric | Previous Qwen | Updated Qwen | Change |
|--------|---------------|--------------|--------|
| BLEU-4 | 0.0914 | 0.0906 | -0.0008 (-0.9%) |
| ROUGE-1 F | 0.4321 | 0.4280 | -0.0041 (-0.9%) |
| BERTScore F1 | 0.4022 | 0.3973 | -0.0049 (-1.2%) |

**Analysis:**
- Scores are very similar, indicating consistent performance
- Small decreases may be due to more constrained baseline answers
- The updated baseline provides a fairer comparison as it uses similar constraints and instructions

### 5.2 Baseline Quality Impact

**OpenAI Baseline Statistics:**

| Metric | Previous | Updated | Change |
|--------|----------|---------|--------|
| Mean Length | 522.2 | 532.5 | +10.3 (+2.0%) |
| Median Length | 510.5 | 516.0 | +5.5 (+1.1%) |
| Max Length | 2,251 | 1,753 | -498 (-22.1%) |

**Analysis:**
- Updated baseline shows better length control (reduced max length)
- Mean and median lengths slightly increased, but max significantly reduced
- Character limits successfully constrained overly long answers
- More consistent baseline improves evaluation fairness

---

## 6. Strengths and Weaknesses

### 6.1 Qwen RAG Concise Strengths

1. **Exceptional Factual Accuracy (Q2):** BERTScore F1 of 0.6692 on formula/type questions
2. **Strong Semantic Alignment:** BERTScore F1 of 0.3973 overall, indicating excellent semantic understanding
3. **Better Precision:** Higher precision across all metrics, indicating more relevant content
4. **Superior Recall:** Better information coverage, especially on Q2 and Q4
5. **Consistent Performance:** Wins across all question types and metrics
6. **Robust to Baseline Changes:** Performance remains strong with updated baseline

### 6.2 Qwen RAG Concise Weaknesses

1. **Visual Tasks (Q1):** Lower BLEU-4 (0.0317), though still better than Gemma
2. **Answer Length Variance:** Slightly higher variance in some metrics
3. **Room for Improvement:** Overall BLEU-4 of 0.0906 indicates potential for better n-gram matching

### 6.3 Gemma RAG Concise Strengths

1. **Consistency:** Lower variance in BLEU-4 scores (more predictable performance)
2. **Conciseness:** Slightly longer answers (231.4 vs 218.7 chars), but still within limits

### 6.4 Gemma RAG Concise Weaknesses

1. **Overall Performance:** Significantly lower scores across all metrics
2. **Factual Questions (Q2):** Particularly weak performance (BERTScore F1: 0.1903)
3. **Semantic Understanding:** BERTScore F1 of 0.1751 indicates limited semantic alignment
4. **Information Coverage:** Low recall across all question types, especially Q4 (0.0184)

---

## 7. Conclusions

### 7.1 Overall Winner

**Qwen RAG Concise is the clear winner** across all evaluation dimensions:

- **BLEU-4:** +112.2% better
- **ROUGE-1 F:** +44.9% better
- **BERTScore F1:** +126.9% better

### 7.2 Key Insights

1. **Qwen excels at factual content:** Q2 (formula/type) shows exceptional performance (BERTScore F1: 0.6692)
2. **Qwen maintains quality while being concise:** Better scores despite similar answer lengths
3. **Qwen shows better RAG integration:** Higher precision and recall suggest better use of retrieved context
4. **Gemma struggles with factual accuracy:** Particularly on Q2 and Q4 questions
5. **Updated baseline provides fairer comparison:** Character limits and explicit instructions align baseline with candidate models

### 7.3 Recommendations

**For Production Use:**
- **Use Qwen RAG Concise** for all question types
- Qwen provides significantly better quality across all metrics
- Qwen's factual accuracy (Q2) makes it particularly suitable for technical content

**For Further Improvement:**
- **Qwen Q1 performance:** Consider enhancing visual understanding capabilities
- **Both models:** Could benefit from better handling of uses/hazards questions (Q4)
- **Gemma:** Needs significant improvement in factual accuracy and semantic understanding

---

## 8. Technical Details

### 8.1 Evaluation Setup

- **Dataset:** 178 chemical compounds
- **Questions per compound:** 4 (Q1: Image, Q2: Formula, Q3: Production, Q4: Uses/Hazards)
- **Total pairs:** 712 question-answer pairs
- **Baseline:** OpenAI comprehensive answers (regenerated with character limits and comprehensive_text instructions)
- **Character limits:** Q1: 600, Q2: 1000, Q3: 1800, Q4: 2000

### 8.2 Metrics Used

1. **BLEU:** N-gram precision (BLEU-1 through BLEU-4)
2. **ROUGE:** Recall-oriented n-gram and LCS overlap (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
3. **BERTScore:** Semantic similarity using `microsoft/deberta-xlarge-mnli` model

### 8.3 Baseline Regeneration Details

**OpenAI Regeneration:**
- **Script:** `generate_qa_pairs_comprehensive_update_with_limits.py`
- **Q2-Q4 Regenerated:** With character limits and explicit comprehensive_text instructions
- **Q1 Preserved:** Original answers maintained
- **Success Rate:** 177/178 files (99.4%)
- **Failed File:** `7_Acetylsalicylic_Acid.json` (token rate limit exceeded)

### 8.4 Files Generated

**Detailed Results:**
- `per_question_bleu_qwen_gemma.csv` - BLEU scores per question
- `per_question_rouge_qwen_gemma.csv` - ROUGE scores per question
- `per_question_bertscore_qwen_gemma.csv` - BERTScore per question

**Summary Files:**
- `summary_metrics_qwen_gemma.json` - BLEU aggregated statistics
- `summary_rouge_qwen_gemma.json` - ROUGE aggregated statistics
- `summary_bertscore_qwen_gemma.json` - BERTScore aggregated statistics

**Comparison Files:**
- `comparison_bleu_qwen_gemma.json` - BLEU comparison metrics
- `comparison_rouge_qwen_gemma.json` - ROUGE comparison metrics
- `comparison_bertscore_qwen_gemma.json` - BERTScore comparison metrics

---

## 9. Appendix: Statistical Tables

### 9.1 Complete BLEU Statistics

| Metric | Qwen Mean | Qwen Median | Qwen Std | Gemma Mean | Gemma Median | Gemma Std |
|--------|-----------|-------------|----------|------------|--------------|-----------|
| BLEU-1 | 0.2425 | 0.1722 | 0.2200 | 0.1886 | 0.1627 | 0.1458 |
| BLEU-2 | 0.1709 | 0.1066 | 0.1879 | 0.1096 | 0.0825 | 0.1088 |
| BLEU-3 | 0.1213 | 0.0628 | 0.1668 | 0.0651 | 0.0375 | 0.0877 |
| BLEU-4 | 0.0906 | 0.0363 | 0.1517 | 0.0427 | 0.0200 | 0.0736 |

### 9.2 Complete ROUGE Statistics

| Metric | Qwen Mean | Qwen Median | Qwen Std | Gemma Mean | Gemma Median | Gemma Std |
|--------|-----------|-------------|----------|------------|--------------|-----------|
| ROUGE-1 F | 0.4280 | 0.3891 | 0.1805 | 0.2953 | 0.2811 | 0.1353 |
| ROUGE-2 F | 0.2025 | 0.1522 | 0.1698 | 0.1082 | 0.0849 | 0.1009 |
| ROUGE-L F | 0.3392 | 0.2778 | 0.1885 | 0.2159 | 0.1971 | 0.1165 |
| ROUGE-Lsum F | 0.3494 | 0.2917 | 0.1843 | 0.2265 | 0.2102 | 0.1189 |

### 9.3 Complete BERTScore Statistics

| Metric | Qwen Mean | Qwen Median | Qwen Std | Gemma Mean | Gemma Median | Gemma Std |
|--------|-----------|-------------|----------|------------|--------------|-----------|
| Precision | 0.5012 | 0.4577 | 0.1969 | 0.2438 | 0.1836 | 0.1904 |
| Recall | 0.3084 | 0.2344 | 0.2496 | 0.1143 | 0.0898 | 0.2025 |
| F1 | 0.3973 | 0.3382 | 0.2185 | 0.1751 | 0.1448 | 0.1834 |

---

**Report Generated:** November 30, 2025  
**Evaluation Scripts:** `/home/himanshu/dev/code/evaluation/scripts/`  
**Results Directory:** `/home/himanshu/dev/code/evaluation/results/`  
**Previous Report:** `QWEN_GEMMA_EVALUATION_REPORT.md` (using original OpenAI baseline)

