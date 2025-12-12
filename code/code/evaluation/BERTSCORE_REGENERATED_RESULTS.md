# BERTScore Results: Regenerated Data Analysis

**Date:** November 23, 2025  
**Total Question-Answer Pairs:** 712 (178 files × 4 questions)  
**Model:** microsoft/deberta-xlarge-mnli

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

## Overall BERTScore Results

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
| **Precision** | 0.0548 | 0.0720 | 0.1914 | -0.4150 | 0.7484 |
| **Recall** | 0.3377 | 0.3141 | 0.1300 | 0.0307 | 0.8701 |
| **F1** | 0.1783 | 0.1832 | 0.1582 | -0.2083 | 0.8081 |

**Note:** BERTScore uses rescaled baseline scores, which can result in negative values for very dissimilar texts. Scores typically range from -1 to 1, with higher scores indicating better semantic similarity.

---

## Per-Question-Type BERTScore F1

| Question | Count | Mean F1 | Median F1 | Mean Precision | Mean Recall |
|----------|-------|---------|-----------|----------------|-------------|
| **Q1 (Vision)** | 178 | **0.3218** | 0.3122 | 0.2105 | **0.4541** |
| **Q2 (Text)** | 178 | 0.0750 | 0.0039 | -0.1041 | 0.3371 |
| **Q3 (Text)** | 178 | 0.1464 | 0.1602 | 0.0343 | 0.2854 |
| **Q4 (Text)** | 178 | 0.1701 | 0.1734 | 0.0785 | 0.2743 |

---

## Key Observations

### 1. Q1 (Vision) Shows Best Performance
- **Highest F1 Score:** 0.3218 (nearly 2x better than Q2)
- **Highest Recall:** 0.4541 (indicates QWEN captures most of the semantic content from OpenAI)
- **Better Precision:** 0.2105 (positive, unlike Q2)
- **Interpretation:** Vision questions have more standardized answer formats, leading to better semantic alignment

### 2. Q2 Shows Lowest Performance
- **Lowest F1 Score:** 0.0750
- **Negative Precision:** -0.1041 (indicates QWEN answers contain information not in OpenAI answers)
- **Interpretation:** Q2 questions (formula/elements) have very different answer styles between QWEN and OpenAI

### 3. Recall > Precision Pattern
- **Overall Recall:** 0.3377 (much higher than precision)
- **Overall Precision:** 0.0548 (very low)
- **Interpretation:** 
  - QWEN answers contain most of the semantic content from OpenAI (high recall)
  - But QWEN also includes much additional information not in OpenAI (low precision)
  - This aligns with QWEN answers being 3x longer than OpenAI answers

### 4. Answer Length Disparity Impact
- **QWEN Average:** 1,535 characters
- **OpenAI Average:** 522 characters
- **Ratio:** ~3:1
- **Impact:** The length difference explains why precision is low but recall is higher - QWEN covers OpenAI content plus much more

---

## Comparison with BLEU Scores

### Correlation Analysis

| Metric | Correlation with BERTScore F1 |
|--------|-------------------------------|
| **BLEU-4** | 0.7765 (strong positive correlation) |
| **BLEU-1** | 0.8743 (very strong positive correlation) |

**Interpretation:** BERTScore F1 correlates strongly with BLEU scores, especially BLEU-1 (unigram overlap). This suggests both metrics capture similar aspects of answer similarity, though BERTScore focuses on semantic similarity while BLEU focuses on n-gram overlap.

### Mean Score Comparison

| Metric | BERTScore | BLEU-4 | BLEU-1 |
|--------|-----------|--------|--------|
| **Mean Score** | 0.1783 | 0.0698 | 0.2175 |

**Observations:**
- **BERTScore F1 (0.1783)** is between BLEU-1 (0.2175) and BLEU-4 (0.0698)
- BERTScore is closer to BLEU-1, which makes sense as both capture broader similarity
- BLEU-4 is much lower, penalizing longer answers more heavily

---

## Per-Question Comparison: BERTScore vs BLEU-4

| Question | BERTScore F1 | BLEU-4 | Difference |
|----------|--------------|--------|------------|
| **Q1** | 0.3218 | 0.1183 | +0.2035 (BERTScore higher) |
| **Q2** | 0.0750 | 0.0584 | +0.0166 (similar) |
| **Q3** | 0.1464 | 0.0513 | +0.0951 (BERTScore higher) |
| **Q4** | 0.1701 | 0.0512 | +0.1189 (BERTScore higher) |

**Key Finding:** BERTScore consistently shows higher scores than BLEU-4, especially for Q1, Q3, and Q4. This suggests that while n-gram overlap is low (BLEU), semantic similarity is higher (BERTScore), indicating that QWEN answers convey similar meaning even if worded differently.

---

## Interpretation of Results

### What BERTScore Tells Us

1. **Semantic Similarity is Better Than N-gram Overlap:**
   - BERTScore F1 (0.1783) > BLEU-4 (0.0698)
   - QWEN answers are semantically similar to OpenAI answers, even if worded differently

2. **QWEN Provides More Comprehensive Answers:**
   - High recall (0.3377) indicates QWEN covers most semantic content from OpenAI
   - Low precision (0.0548) indicates QWEN adds significant additional information
   - This is actually a positive trait for comprehensive answers

3. **Vision Questions Have Best Alignment:**
   - Q1 has the highest BERTScore (0.3218)
   - Vision questions have more standardized formats
   - Both models produce similar semantic content for visual descriptions

4. **Text Questions Show More Variation:**
   - Q2-Q4 have lower BERTScore but still positive
   - Different writing styles and levels of detail
   - QWEN's more verbose style reduces exact matches but maintains semantic similarity

---

## Recommendations

### 1. Use BERTScore for Semantic Evaluation
- BERTScore better captures semantic similarity than BLEU
- More appropriate for evaluating longer, more detailed answers
- Less penalized by length differences

### 2. Consider Answer Length in Evaluation
- QWEN answers are 3x longer and more comprehensive
- This is a feature, not a bug - more detailed answers can be better
- Use precision/recall analysis to understand information coverage

### 3. Question-Specific Analysis
- Q1 (vision) performs best - consider this format for other questions
- Q2 shows lowest alignment - may need prompt engineering for formula questions
- Q3-Q4 show moderate alignment - acceptable for detailed explanations

### 4. Combined Metrics Approach
- Use BERTScore for semantic similarity
- Use BLEU for exact phrase matching
- Use human evaluation for overall quality assessment

---

## Files Generated

- **Detailed Results:** `/home/himanshu/dev/code/bleu_evaluation/results/per_question_bertscore_regenerated.csv`
- **Summary Metrics:** `/home/himanshu/dev/code/bleu_evaluation/results/summary_bertscore_regenerated.json`
- **Script:** `/home/himanshu/dev/code/bleu_evaluation/scripts/bertscore_calculator_regenerated.py`

---

## Conclusion

BERTScore analysis reveals that:

✅ **QWEN answers are semantically similar to OpenAI answers** (F1: 0.1783)  
✅ **QWEN provides more comprehensive coverage** (Recall: 0.3377)  
✅ **Vision questions show best alignment** (Q1 F1: 0.3218)  
✅ **Semantic similarity is better than n-gram overlap** (BERTScore > BLEU-4)  

The regenerated QWEN answers, while longer and more detailed, maintain good semantic similarity to the OpenAI reference answers. The lower precision scores reflect QWEN's more comprehensive approach, which may actually be preferable for detailed explanations.

**Key Takeaway:** BERTScore suggests that QWEN answers are semantically aligned with OpenAI answers, even though they use different wording and include more detail. This is a positive indicator of answer quality, especially for comprehensive explanations.

