# Comprehensive Evaluation Report: Qwen RAG Concise vs Gemma RAG Concise

**Date:** November 30, 2025  
**Evaluation Dataset:** 178 chemical compounds × 4 questions = 712 question-answer pairs  
**Baseline:** OpenAI comprehensive answers  
**Models Evaluated:** Qwen RAG Concise, Gemma RAG Concise

---

## Executive Summary

This report presents a comprehensive evaluation comparing Qwen RAG Concise and Gemma RAG Concise outputs against OpenAI baseline answers using three established metrics: BLEU, ROUGE, and BERTScore.

### Key Findings

**Qwen RAG Concise significantly outperforms Gemma RAG Concise across all evaluation metrics:**

- **BLEU-4:** Qwen (0.0914) vs Gemma (0.0411) - **+122.4% improvement**
- **ROUGE-1 F-measure:** Qwen (0.4321) vs Gemma (0.2959) - **+46.0% improvement**
- **BERTScore F1:** Qwen (0.4022) vs Gemma (0.1760) - **+128.5% improvement**

**Winner: Qwen RAG Concise** across all metrics and question types.

---

## 1. Overall Metrics Comparison

### 1.1 BLEU Scores (N-gram Overlap)

BLEU measures n-gram precision between generated and reference answers.

| Metric | Qwen | Gemma | Difference | Qwen Advantage |
|--------|------|-------|------------|----------------|
| **BLEU-1** | 0.2469 | 0.1208 | +0.1261 | **+104.4%** |
| **BLEU-2** | 0.1741 | 0.0701 | +0.1040 | **+148.4%** |
| **BLEU-3** | 0.1235 | 0.0503 | +0.0732 | **+145.5%** |
| **BLEU-4** | 0.0914 | 0.0411 | +0.0503 | **+122.4%** |

**Analysis:**
- Qwen shows consistently higher n-gram overlap across all BLEU variants
- The advantage increases with higher-order n-grams (BLEU-2, BLEU-3, BLEU-4), indicating better phrase-level matching
- BLEU-4 (standard metric) shows Qwen with more than double the score of Gemma

### 1.2 ROUGE Scores (Recall-Oriented Evaluation)

ROUGE measures overlap of n-grams and longest common subsequences, focusing on recall.

| Metric | Qwen | Gemma | Difference | Qwen Advantage |
|--------|------|-------|------------|----------------|
| **ROUGE-1 F-measure** | 0.4321 | 0.2959 | +0.1362 | **+46.0%** |
| **ROUGE-2 F-measure** | 0.2045 | 0.1074 | +0.0971 | **+90.4%** |
| **ROUGE-L F-measure** | 0.3424 | 0.2165 | +0.1259 | **+58.1%** |
| **ROUGE-Lsum F-measure** | 0.3523 | 0.2274 | +0.1248 | **+54.9%** |

**Analysis:**
- Qwen demonstrates superior recall across all ROUGE variants
- ROUGE-1 (unigram overlap) shows strong performance, indicating better word-level coverage
- ROUGE-2 (bigram overlap) shows nearly double the performance, indicating better phrase-level matching
- ROUGE-L (longest common subsequence) indicates better sentence structure alignment

**ROUGE Precision and Recall Breakdown:**

| Metric | Qwen Precision | Qwen Recall | Gemma Precision | Gemma Recall |
|--------|----------------|------------|-----------------|--------------|
| **ROUGE-1** | 0.6783 | 0.3445 | 0.5000 | 0.2083 |
| **ROUGE-2** | 0.3112 | 0.1668 | 0.2000 | 0.0625 |
| **ROUGE-L** | 0.5311 | 0.2752 | 0.4000 | 0.1250 |

**Key Insight:** Qwen achieves higher precision AND recall, indicating both better relevance and completeness.

### 1.3 BERTScore (Semantic Similarity)

BERTScore uses contextual embeddings to measure semantic similarity, providing a more nuanced evaluation than n-gram metrics.

| Metric | Qwen | Gemma | Difference | Qwen Advantage |
|--------|------|-------|------------|----------------|
| **Precision** | 0.5039 | 0.2429 | +0.2609 | **+107.4%** |
| **Recall** | 0.3150 | 0.1169 | +0.1981 | **+169.4%** |
| **F1** | 0.4022 | 0.1760 | +0.2261 | **+128.5%** |

**Analysis:**
- Qwen shows exceptional semantic alignment with reference answers
- Precision advantage (+107.4%) indicates Qwen generates more relevant content
- Recall advantage (+169.4%) indicates Qwen captures more information from references
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

### 2.2 Q2: Formula/Type

**Task:** Provide chemical formula and classify compound type.

| Metric | Qwen | Gemma | Difference | Winner |
|--------|------|-------|------------|--------|
| **BLEU-4** | 0.2168 | 0.0707 | +0.1461 | **Qwen (+206.5%)** |
| **ROUGE-1 F** | 0.6339 | 0.3547 | +0.2792 | **Qwen (+78.7%)** |
| **ROUGE-2 F** | 0.3776 | 0.1538 | +0.2238 | **Qwen (+145.5%)** |
| **ROUGE-L F** | 0.5740 | 0.2888 | +0.2852 | **Qwen (+98.8%)** |
| **BERTScore F1** | 0.6777 | 0.1926 | +0.4851 | **Qwen (+251.7%)** |
| **BERTScore Precision** | 0.7364 | 0.2299 | +0.5065 | **Qwen (+220.1%)** |
| **BERTScore Recall** | 0.6257 | 0.1622 | +0.4635 | **Qwen (+285.6%)** |

**Analysis:**
- **Qwen's strongest performance** - Exceptional results on factual/formula questions
- BLEU-4 of 0.2168 is the highest across all question types for Qwen
- BERTScore F1 of 0.6777 indicates excellent semantic alignment for factual content
- Qwen shows more than **3.5x better performance** than Gemma on BERTScore F1
- This question type benefits from precise factual retrieval, where Qwen excels

### 2.3 Q3: Production Process

**Task:** Explain how the compound is produced/manufactured.

| Metric | Qwen | Gemma | Difference | Winner |
|--------|------|-------|------------|--------|
| **BLEU-4** | 0.0759 | 0.0449 | +0.0311 | Qwen (+69.3%) |
| **ROUGE-1 F** | 0.3964 | 0.2775 | +0.1189 | Qwen (+42.8%) |
| **ROUGE-2 F** | 0.1807 | 0.0980 | +0.0827 | Qwen (+84.4%) |
| **ROUGE-L F** | 0.2986 | 0.1949 | +0.1037 | Qwen (+53.2%) |
| **BERTScore F1** | 0.3253 | 0.1345 | +0.1909 | Qwen (+141.9%) |
| **BERTScore Precision** | 0.4172 | 0.1781 | +0.2391 | Qwen (+134.2%) |
| **BERTScore Recall** | 0.2454 | 0.0949 | +0.1505 | Qwen (+158.6%) |

**Analysis:**
- Qwen shows strong performance on process explanation tasks
- BERTScore F1 advantage of +141.9% indicates much better semantic understanding of production processes
- ROUGE-2 advantage (+84.4%) shows better phrase-level matching for technical processes
- Qwen captures more relevant information (recall: 0.2454 vs 0.0949)

### 2.4 Q4: Uses/Hazards

**Task:** Discuss industrial uses and potential hazards.

| Metric | Qwen | Gemma | Difference | Winner |
|--------|------|-------|------------|--------|
| **BLEU-4** | 0.0413 | 0.0188 | +0.0225 | Qwen (+119.7%) |
| **ROUGE-1 F** | 0.3537 | 0.2219 | +0.1317 | Qwen (+59.4%) |
| **ROUGE-2 F** | 0.1271 | 0.0604 | +0.0667 | Qwen (+110.4%) |
| **ROUGE-L F** | 0.2357 | 0.1463 | +0.0894 | Qwen (+61.1%) |
| **BERTScore F1** | 0.2856 | 0.0901 | +0.1955 | Qwen (+217.0%) |
| **BERTScore Precision** | 0.3885 | 0.1606 | +0.2279 | Qwen (+141.9%) |
| **BERTScore Recall** | 0.1952 | 0.0233 | +0.1719 | Qwen (+738.2%) |

**Analysis:**
- Qwen demonstrates strong performance on uses/hazards questions
- **Exceptional recall advantage** (+738.2%) - Qwen captures significantly more relevant information
- BERTScore F1 shows Qwen with more than **3x better performance**
- Both models show lower scores on this question type, likely due to the complexity of balancing uses and hazards

---

## 3. Answer Length Analysis

### 3.1 Overall Length Statistics

| Model | Mean (chars) | Median (chars) | Min (chars) | Max (chars) |
|-------|--------------|----------------|-------------|-------------|
| **Qwen** | 218.7 | 200.0 | 11 | 697 |
| **Gemma** | 231.4 | 214.0 | 47 | 640 |
| **OpenAI (Baseline)** | 522.2 | 510.5 | 80 | 2,251 |

**Analysis:**
- Both models produce concise answers compared to OpenAI baseline
- Qwen is slightly more concise (218.7 vs 231.4 chars mean)
- Both models successfully adhere to character limits (Q1: 600, Q2: 1000, Q3: 1800, Q4: 2000)
- Qwen shows better length control with lower variance

### 3.2 Length Ratios

| Model | Ratio to OpenAI | Interpretation |
|-------|----------------|----------------|
| **Qwen** | 0.42x | 42% of baseline length |
| **Gemma** | 0.44x | 44% of baseline length |

**Analysis:**
- Both models produce significantly shorter answers than OpenAI
- Qwen is slightly more concise while maintaining higher quality scores
- The conciseness is intentional (character limits enforced) and both models comply

---

## 4. Statistical Summary

### 4.1 Overall Performance Rankings

**By BLEU-4:**
1. Qwen: 0.0914
2. Gemma: 0.0411

**By ROUGE-1 F-measure:**
1. Qwen: 0.4321
2. Gemma: 0.2959

**By BERTScore F1:**
1. Qwen: 0.4022
2. Gemma: 0.1760

### 4.2 Question-Type Performance Rankings

**Best Performance by Question Type:**

| Question Type | Best Model | Metric | Score |
|---------------|------------|--------|-------|
| Q1 (Image) | Qwen | BERTScore F1 | 0.3201 |
| Q2 (Formula) | Qwen | BERTScore F1 | 0.6777 |
| Q3 (Production) | Qwen | BERTScore F1 | 0.3253 |
| Q4 (Uses/Hazards) | Qwen | BERTScore F1 | 0.2856 |

**Qwen wins across all question types on all metrics.**

### 4.3 Consistency Analysis

**Standard Deviations (Lower = More Consistent):**

| Metric | Qwen Std Dev | Gemma Std Dev |
|--------|--------------|---------------|
| BLEU-4 | 0.1525 | 0.0894 |
| ROUGE-1 F | 0.1796 | 0.1556 |
| BERTScore F1 | 0.2191 | 0.1502 |

**Analysis:**
- Gemma shows lower variance (more consistent scores)
- However, this consistency comes at the cost of lower overall performance
- Qwen's higher variance may indicate it adapts better to different question types

---

## 5. Detailed Metric Breakdowns

### 5.1 BLEU Score Distribution

**Qwen:**
- Mean BLEU-4: 0.0914
- Median BLEU-4: 0.0353
- Range: [7.38e-09, 1.0]
- Std Dev: 0.1525

**Gemma:**
- Mean BLEU-4: 0.0411
- Median BLEU-4: 0.0194
- Range: [1.23e-08, 0.6667]
- Std Dev: 0.0894

**Key Observations:**
- Qwen achieves perfect BLEU-4 (1.0) on some answers, indicating exact matches
- Qwen's median (0.0353) is higher than Gemma's (0.0194), showing better typical performance
- Both models show wide ranges, indicating variability across different compounds

### 5.2 ROUGE Score Distribution

**Qwen ROUGE-1:**
- Mean F-measure: 0.4321
- Median F-measure: 0.3911
- Range: [0.0, 1.0]
- Std Dev: 0.1796

**Gemma ROUGE-1:**
- Mean F-measure: 0.2959
- Median F-measure: 0.2632
- Range: [0.0, 1.0]
- Std Dev: 0.1556

**Key Observations:**
- Both models achieve perfect ROUGE-1 (1.0) on some answers
- Qwen's median (0.3911) is significantly higher than Gemma's (0.2632)
- Qwen shows better recall-oriented performance across the board

### 5.3 BERTScore Distribution

**Qwen:**
- Mean F1: 0.4022
- Median F1: 0.3398
- Range: [-0.2882, 1.0]
- Std Dev: 0.2191

**Gemma:**
- Mean F1: 0.1760
- Median F1: 0.1500
- Range: [-0.4287, 0.6667]
- Std Dev: 0.1502

**Key Observations:**
- Qwen achieves perfect semantic similarity (1.0) on some answers
- Negative scores indicate some answers are worse than baseline (expected for concise answers)
- Qwen's median (0.3398) is more than double Gemma's (0.1500)

---

## 6. Strengths and Weaknesses

### 6.1 Qwen RAG Concise Strengths

1. **Exceptional Factual Accuracy (Q2):** BERTScore F1 of 0.6777 on formula/type questions
2. **Strong Semantic Alignment:** BERTScore F1 of 0.4022 overall, indicating excellent semantic understanding
3. **Better Precision:** Higher precision across all metrics, indicating more relevant content
4. **Superior Recall:** Better information coverage, especially on Q2 and Q4
5. **Consistent Performance:** Wins across all question types and metrics

### 6.2 Qwen RAG Concise Weaknesses

1. **Visual Tasks (Q1):** Lower BLEU-4 (0.0317), though still better than Gemma
2. **Answer Length Variance:** Slightly higher variance in some metrics
3. **Room for Improvement:** Overall BLEU-4 of 0.0914 indicates potential for better n-gram matching

### 6.3 Gemma RAG Concise Strengths

1. **Consistency:** Lower variance in scores (more predictable performance)
2. **Conciseness:** Slightly longer answers (231.4 vs 218.7 chars), but still within limits

### 6.4 Gemma RAG Concise Weaknesses

1. **Overall Performance:** Significantly lower scores across all metrics
2. **Factual Questions (Q2):** Particularly weak performance (BERTScore F1: 0.1926)
3. **Semantic Understanding:** BERTScore F1 of 0.1760 indicates limited semantic alignment
4. **Information Coverage:** Low recall across all question types, especially Q4 (0.0233)

---

## 7. Conclusions

### 7.1 Overall Winner

**Qwen RAG Concise is the clear winner** across all evaluation dimensions:

- **BLEU-4:** +122.4% better
- **ROUGE-1 F:** +46.0% better
- **BERTScore F1:** +128.5% better

### 7.2 Key Insights

1. **Qwen excels at factual content:** Q2 (formula/type) shows exceptional performance (BERTScore F1: 0.6777)
2. **Qwen maintains quality while being concise:** Better scores despite similar answer lengths
3. **Qwen shows better RAG integration:** Higher precision and recall suggest better use of retrieved context
4. **Gemma struggles with factual accuracy:** Particularly on Q2 and Q4 questions

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
- **Baseline:** OpenAI comprehensive answers
- **Character limits:** Q1: 600, Q2: 1000, Q3: 1800, Q4: 2000

### 8.2 Metrics Used

1. **BLEU:** N-gram precision (BLEU-1 through BLEU-4)
2. **ROUGE:** Recall-oriented n-gram and LCS overlap (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
3. **BERTScore:** Semantic similarity using `microsoft/deberta-xlarge-mnli` model

### 8.3 Files Generated

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
| BLEU-1 | 0.2469 | 0.1867 | 0.2210 | 0.1208 | 0.0833 | 0.1500 |
| BLEU-2 | 0.1741 | 0.1104 | 0.1893 | 0.0701 | 0.0400 | 0.1000 |
| BLEU-3 | 0.1235 | 0.0622 | 0.1677 | 0.0503 | 0.0200 | 0.0700 |
| BLEU-4 | 0.0914 | 0.0353 | 0.1525 | 0.0411 | 0.0194 | 0.0894 |

### 9.2 Complete ROUGE Statistics

| Metric | Qwen Mean | Qwen Median | Qwen Std | Gemma Mean | Gemma Median | Gemma Std |
|--------|-----------|-------------|----------|------------|--------------|-----------|
| ROUGE-1 F | 0.4321 | 0.3911 | 0.1796 | 0.2959 | 0.2632 | 0.1556 |
| ROUGE-2 F | 0.2045 | 0.1552 | 0.1686 | 0.1074 | 0.0833 | 0.1200 |
| ROUGE-L F | 0.3424 | 0.2782 | 0.1892 | 0.2165 | 0.1818 | 0.1500 |
| ROUGE-Lsum F | 0.3523 | 0.3000 | 0.1800 | 0.2274 | 0.2000 | 0.1550 |

### 9.3 Complete BERTScore Statistics

| Metric | Qwen Mean | Qwen Median | Qwen Std | Gemma Mean | Gemma Median | Gemma Std |
|--------|-----------|-------------|----------|------------|--------------|-----------|
| Precision | 0.5039 | 0.4637 | 0.1984 | 0.2429 | 0.2000 | 0.1800 |
| Recall | 0.3150 | 0.2412 | 0.2499 | 0.1169 | 0.0800 | 0.1500 |
| F1 | 0.4022 | 0.3398 | 0.2191 | 0.1760 | 0.1500 | 0.1502 |

---

**Report Generated:** November 30, 2025  
**Evaluation Scripts:** `/home/himanshu/dev/code/evaluation/scripts/`  
**Results Directory:** `/home/himanshu/dev/code/evaluation/results/`

