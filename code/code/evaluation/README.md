# Evaluation Metrics: QWEN vs OpenAI Answers

This directory contains scripts and documentation for calculating evaluation metrics (BLEU and BERTScore) to compare QWEN-generated answers against OpenAI-generated answers (baseline).

## Directory Structure

```
evaluation/
├── README.md                          # This file
├── BLEU_SCORE_PLAN.md                # Detailed plan and methodology
├── BERTSCORE_PLAN.md                 # BERTScore plan and methodology
├── scripts/
│   ├── bleu_score_calculator.py     # Main BLEU calculation script
│   ├── bleu_score_calculator_regenerated.py  # BLEU calculator for regenerated data
│   ├── bertscore_calculator.py      # Main BERTScore calculation script
│   └── bertscore_calculator_regenerated.py  # BERTScore calculator for regenerated data
├── results/                           # Output directory (generated)
│   ├── per_question_bleu.csv        # Detailed per-question BLEU scores (original)
│   ├── per_question_bleu_regenerated.csv  # BLEU scores for regenerated data
│   ├── per_question_bertscore.csv   # Detailed per-question BERTScore (original)
│   ├── per_question_bertscore_regenerated.csv  # BERTScore for regenerated data
│   ├── summary_metrics.json         # Aggregated BLEU statistics (original)
│   ├── summary_metrics_regenerated.json  # BLEU stats for regenerated data
│   ├── summary_bertscore.json       # Aggregated BERTScore statistics (original)
│   └── summary_bertscore_regenerated.json  # BERTScore stats for regenerated data
└── visualizations/                   # Charts and plots (generated)
```

## Quick Start

### 1. Install Dependencies

```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate
pip install -r evaluation/requirements.txt
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"
```

### 2. Run Evaluation Scripts

**BLEU Score Calculation (Original Data):**
```bash
cd /home/himanshu/dev/code/evaluation/scripts
python bleu_score_calculator.py
```

**BLEU Score Calculation (Regenerated Data):**
```bash
cd /home/himanshu/dev/code/evaluation/scripts
python bleu_score_calculator_regenerated.py
```

**BERTScore Calculation (Original Data):**
```bash
cd /home/himanshu/dev/code/evaluation/scripts
python bertscore_calculator.py
```

**BERTScore Calculation (Regenerated Data):**
```bash
cd /home/himanshu/dev/code/evaluation/scripts
python bertscore_calculator_regenerated.py
```

**Note**: BERTScore requires a GPU for efficient processing. The script will automatically use CPU if GPU is not available, but it will be slower.

## Data Sources

### Original Data
- **QWEN Answers (Candidate)**: `/home/himanshu/dev/output/qwen/*__answers.json`
- **OpenAI Answers (Reference)**: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components/*.json`

### Regenerated Data
- **QWEN Answers (Candidate)**: `/home/himanshu/dev/output/qwen_regenerated/*__answers.json`
  - Regenerated with vLLM, max_tokens=500
- **OpenAI Answers (Reference)**: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive/*.json`
  - Regenerated with comprehensive_text, max_tokens=500

## Output

### Files Generated

**BLEU Results:**
1. **`results/per_question_bleu.csv`**: Detailed BLEU scores for each question-answer pair
   - Columns: file_base, question_index, question, qwen_answer, openai_answer, bleu_1, bleu_2, bleu_3, bleu_4

2. **`results/summary_metrics.json`**: Aggregated BLEU statistics
   - Overall corpus BLEU scores
   - Per-question-type statistics (Q1, Q2, Q3, Q4)
   - Per-compound statistics
   - Answer length comparisons

**BERTScore Results:**
3. **`results/per_question_bertscore.csv`**: Detailed BERTScore for each question-answer pair
   - Columns: file_base, question_index, question, qwen_answer, openai_answer, bertscore_precision, bertscore_recall, bertscore_f1

4. **`results/summary_bertscore.json`**: Aggregated BERTScore statistics
   - Overall corpus BERTScore (Precision, Recall, F1)
   - Per-question-type statistics
   - Per-compound statistics
   - Answer length comparisons
   - Correlation with BLEU scores

### Metrics Calculated

**BLEU Scores:**
- **BLEU-1**: Unigram precision
- **BLEU-2**: Bigram precision  
- **BLEU-3**: Trigram precision
- **BLEU-4**: 4-gram precision (standard BLEU)

**BERTScore:**
- **Precision**: How much of the candidate answer is semantically similar to the reference
- **Recall**: How much of the reference answer is captured in the candidate
- **F1**: Harmonic mean of Precision and Recall (most commonly used)

## Interpretation

### BLEU Score Ranges

- **0.0 - 0.3**: Poor similarity (very different answers)
- **0.3 - 0.5**: Moderate similarity (some overlap)
- **0.5 - 0.7**: Good similarity (substantial overlap)
- **0.7 - 0.9**: High similarity (very similar content)
- **0.9 - 1.0**: Excellent similarity (nearly identical)

### BERTScore Ranges

- **0.0 - 0.5**: Poor semantic similarity (very different meaning)
- **0.5 - 0.7**: Moderate semantic similarity (some shared concepts)
- **0.7 - 0.85**: Good semantic similarity (substantial meaning overlap)
- **0.85 - 0.95**: High semantic similarity (very similar meaning)
- **0.95 - 1.0**: Excellent semantic similarity (nearly identical meaning)

### Important Notes

**BLEU:**
- Measures n-gram overlap, not semantic correctness
- Low BLEU doesn't necessarily mean wrong answer (could be different phrasing)
- High BLEU suggests similar wording and content
- For QA evaluation, BLEU-1 and BLEU-2 are often most informative

**BERTScore:**
- Measures semantic similarity using contextual embeddings
- More robust to paraphrasing than BLEU
- Better captures meaning equivalence even with different word choices
- F1 score is the most balanced metric (considers both precision and recall)
- Typically correlates with human judgment better than BLEU

## Methodology

- **BLEU**: See `BLEU_SCORE_PLAN.md` for detailed methodology and implementation details.
- **BERTScore**: See `BERTSCORE_PLAN.md` for detailed methodology and implementation details.

## Troubleshooting

### Missing NLTK Data
If you see errors about missing NLTK data:
```python
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"
```

### BERTScore Model Download
On first run, BERTScore will download the model (~1.5GB for deberta-xlarge). This is automatic but may take a few minutes depending on your internet connection.

### GPU vs CPU
BERTScore runs much faster on GPU. If you have CUDA available, it will be used automatically. To force CPU usage:
```bash
python bertscore_calculator.py --device cpu
```

### File Matching Issues
The script matches files by extracting the base filename:
- QWEN: `1_13-Butadiene__answers.json` → `1_13-Butadiene`
- OpenAI: `1_13-Butadiene.json` → `1_13-Butadiene`

If files don't match, check the console output for warnings.

---

**Last Updated**: November 23, 2025

## Regenerated Data Analysis

See the following documents for detailed analysis of regenerated data:
- `BLEU_REGENERATED_RESULTS.md` - BLEU score analysis for regenerated QWEN vs OpenAI answers
- `BERTSCORE_REGENERATED_RESULTS.md` - BERTScore analysis for regenerated QWEN vs OpenAI answers

