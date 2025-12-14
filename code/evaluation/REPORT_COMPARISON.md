# Key Differences Between the Two Evaluation Reports

## 1. BASELINE DIFFERENCES

### Original Report (QWEN_GEMMA_EVALUATION_REPORT.md)
- **Baseline:** Original OpenAI comprehensive answers
- **No character limits** on OpenAI answers
- **No explicit comprehensive_text instruction** in prompt
- **Q1-Q4:** All original answers

### Updated Report (QWEN_GEMMA_EVALUATION_REPORT_UPDATED_OPENAI.md)
- **Baseline:** Regenerated OpenAI answers (Q2-Q4 only)
- **Character limits applied:** Q2=1000, Q3=1800, Q4=2000
- **Explicit instruction:** "Your answer MUST be based ONLY on comprehensive_text"
- **Q1:** Preserved from original (not regenerated)
- **Q2-Q4:** Regenerated with aligned prompt structure

---

## 2. SCORE DIFFERENCES

### Overall Metrics Comparison

| Metric | Original Report | Updated Report | Change |
|--------|----------------|----------------|--------|
| **BLEU-4 (Qwen)** | 0.0914 | 0.0906 | -0.0008 (-0.9%) |
| **BLEU-4 (Gemma)** | 0.0411 | 0.0427 | +0.0016 (+3.9%) |
| **ROUGE-1 F (Qwen)** | 0.4321 | 0.4280 | -0.0041 (-0.9%) |
| **ROUGE-1 F (Gemma)** | 0.2959 | 0.2953 | -0.0006 (-0.2%) |
| **BERTScore F1 (Qwen)** | 0.4022 | 0.3973 | -0.0049 (-1.2%) |
| **BERTScore F1 (Gemma)** | 0.1760 | 0.1751 | -0.0009 (-0.5%) |

### Q2 (Formula/Type) - Most Impacted

| Metric | Original | Updated | Change |
|--------|----------|---------|--------|
| **BLEU-4 (Qwen)** | 0.2168 | 0.2167 | -0.0001 |
| **BLEU-4 (Gemma)** | 0.0707 | 0.0760 | +0.0053 (+7.5%) |
| **BERTScore F1 (Qwen)** | 0.6777 | 0.6692 | -0.0085 (-1.3%) |
| **BERTScore F1 (Gemma)** | 0.1926 | 0.1903 | -0.0023 (-1.2%) |

### Q3 (Production Process)

| Metric | Original | Updated | Change |
|--------|----------|---------|--------|
| **BLEU-4 (Qwen)** | 0.0759 | 0.0732 | -0.0027 (-3.6%) |
| **BERTScore F1 (Qwen)** | 0.3253 | 0.3212 | -0.0041 (-1.3%) |

### Q4 (Uses/Hazards)

| Metric | Original | Updated | Change |
|--------|----------|---------|--------|
| **BLEU-4 (Qwen)** | 0.0413 | 0.0407 | -0.0006 (-1.5%) |
| **BERTScore Recall (Qwen)** | 0.1952 | 0.1850 | -0.0102 (-5.2%) |
| **BERTScore Recall (Gemma)** | 0.0233 | 0.0184 | -0.0049 (-21.0%) |

---

## 3. PERCENTAGE IMPROVEMENT DIFFERENCES

### Overall Advantage (Qwen vs Gemma)

| Metric | Original Report | Updated Report | Difference |
|--------|----------------|----------------|------------|
| **BLEU-4 Advantage** | +122.4% | +112.2% | -10.2% |
| **ROUGE-1 F Advantage** | +46.0% | +44.9% | -1.1% |
| **BERTScore F1 Advantage** | +128.5% | +126.9% | -1.6% |

**Note:** Qwen still wins by a large margin in both reports, but the advantage is slightly smaller in the updated report (Gemma improved slightly relative to Qwen).

---

## 4. ANSWER LENGTH DIFFERENCES

### OpenAI Baseline Lengths

| Metric | Original Report | Updated Report | Change |
|--------|----------------|----------------|--------|
| **Mean Length** | 522.2 chars | 532.5 chars | +10.3 (+2.0%) |
| **Median Length** | 510.5 chars | 516.0 chars | +5.5 (+1.1%) |
| **Max Length** | 2,251 chars | 1,753 chars | -498 (-22.1%) |

**Key Insight:** Updated baseline has better length control (reduced max length significantly).

---

## 5. NEW SECTIONS IN UPDATED REPORT

The updated report includes:

1. **Section 5: Comparison with Previous Evaluation**
   - Impact of updated baseline
   - Baseline quality impact analysis
   - Score change analysis

2. **Section 8.3: Baseline Regeneration Details**
   - Script used for regeneration
   - Success rate (177/178 files)
   - Failed file information

3. **Additional Notes:**
   - Explanation of baseline regeneration process
   - Note about Q1 preservation
   - Reference to previous report

---

## 6. KEY INSIGHTS

### Why Scores Changed:

1. **Qwen scores slightly decreased** (-0.9% to -1.2%):
   - Updated baseline has character limits, making it more constrained
   - More aligned baseline may be slightly harder to match exactly
   - Still maintains strong performance

2. **Gemma BLEU-4 improved** (+3.9%):
   - Updated baseline may be more aligned with Gemma's output style
   - Character limits may help Gemma match better

3. **Overall conclusion unchanged:**
   - Qwen still significantly outperforms Gemma
   - All metrics show Qwen as clear winner
   - Performance patterns remain consistent

### Why Updated Baseline is Better:

1. **Fairer comparison:** Character limits align baseline with candidate models
2. **Explicit instructions:** "ONLY from comprehensive_text" matches Qwen/Gemma approach
3. **Better length control:** Reduced max length from 2,251 to 1,753 chars
4. **Consistent prompt structure:** Aligned with Qwen/Gemma generation approach

---

## 7. RECOMMENDATION

**Use the Updated Report** (`QWEN_GEMMA_EVALUATION_REPORT_UPDATED_OPENAI.md`) because:
- More fair comparison (aligned constraints and instructions)
- Better baseline quality control
- Includes comparison with previous evaluation
- Documents the baseline regeneration process

The original report remains valuable for:
- Historical reference
- Understanding impact of baseline quality
- Comparison purposes
