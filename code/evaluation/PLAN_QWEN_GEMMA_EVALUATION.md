# Plan: BLEU and BERTScore Evaluation for Qwen and Gemma RAG Concise

## Overview
Calculate BLEU and BERTScore metrics for both Qwen RAG Concise and Gemma RAG Concise outputs, comparing them against the OpenAI baseline.

## Data Sources

### Input Directories
1. **Qwen RAG Concise**: `/home/himanshu/dev/output/qwen_rag_concise`
   - Format: `*__answers.json` files with `answers` array
   - Each answer has `question` and `answer` fields

2. **Gemma RAG Concise**: `/home/himanshu/dev/output/gemma3_rag_concise`
   - Format: `*__answers.json` files with `answers` array
   - Each answer has `question` and `answer` fields

3. **OpenAI Baseline**: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive`
   - Format: `*.json` files with `qa_pairs` array
   - Each qa_pair has `question` and `answer` fields

## Scripts to Create

### 1. `bleu_score_calculator_qwen_gemma.py`
**Purpose**: Calculate BLEU scores for both Qwen and Gemma against OpenAI baseline.

**Features**:
- Load Qwen answers from `qwen_rag_concise`
- Load Gemma answers from `gemma3_rag_concise`
- Load OpenAI baseline from `qa_pairs_individual_components_comprehensive`
- Match answers by file base and question index (Q1=0, Q2=1, Q3=2, Q4=3)
- Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 for each model
- Generate per-question statistics (Q1, Q2, Q3, Q4)
- Generate per-compound statistics
- Include answer length analysis
- Output comparison summary

**Output Files**:
- `results/per_question_bleu_qwen_gemma.csv` - Detailed per-question BLEU scores for both models
- `results/summary_metrics_qwen_gemma.json` - Aggregated statistics for both models
- `results/comparison_bleu_qwen_gemma.json` - Direct comparison metrics

### 2. `bertscore_calculator_qwen_gemma.py`
**Purpose**: Calculate BERTScore for both Qwen and Gemma against OpenAI baseline.

**Features**:
- Load Qwen answers from `qwen_rag_concise`
- Load Gemma answers from `gemma3_rag_concise`
- Load OpenAI baseline from `qa_pairs_individual_components_comprehensive`
- Match answers by file base and question index
- Calculate Precision, Recall, F1 for each model
- Use `microsoft/deberta-xlarge-mnli` model (same as previous scripts)
- Generate per-question statistics (Q1, Q2, Q3, Q4)
- Generate per-compound statistics
- Include answer length analysis
- Output comparison summary

**Output Files**:
- `results/per_question_bertscore_qwen_gemma.csv` - Detailed per-question BERTScore for both models
- `results/summary_bertscore_qwen_gemma.json` - Aggregated statistics for both models
- `results/comparison_bertscore_qwen_gemma.json` - Direct comparison metrics

## Implementation Details

### Matching Logic
- Extract file base from filename (e.g., `1_13-Butadiene__answers.json` → `1_13-Butadiene`)
- Match by file base across all three sources
- Match by question index (0-3 for Q1-Q4)
- Log warnings for mismatched questions or missing files

### Statistics to Calculate
For each model (Qwen and Gemma):
1. **Overall Statistics**:
   - Mean, median, std, min, max for all metrics
   - Total number of matched pairs

2. **Per-Question-Type Statistics**:
   - Separate stats for Q1, Q2, Q3, Q4
   - Count, mean, median, std for each question type

3. **Per-Compound Statistics**:
   - Mean BLEU-4 / BERTScore F1 per compound
   - Count of questions per compound

4. **Answer Length Statistics**:
   - Mean, median, min, max character and token lengths
   - Length ratios (model / OpenAI)

### Comparison Metrics
- Direct side-by-side comparison of Qwen vs Gemma
- Percentage differences
- Best model per question type
- Overall winner

## Execution Order

1. **Run BLEU calculation**:
   ```bash
   cd /home/himanshu/dev/code/evaluation
   python3 scripts/bleu_score_calculator_qwen_gemma.py
   ```

2. **Run BERTScore calculation**:
   ```bash
   cd /home/himanshu/dev/code/evaluation
   python3 scripts/bertscore_calculator_qwen_gemma.py
   ```

3. **Review results** in `results/` directory

## Expected Output Structure

### CSV Files
- One row per question-answer pair
- Columns: file_base, question_index, question, qwen_answer, gemma_answer, openai_answer, qwen_bleu_4, gemma_bleu_4, qwen_length, gemma_length, openai_length, etc.

### JSON Summary Files
```json
{
  "qwen": {
    "overall": {...},
    "per_question_type": {...},
    "per_compound": {...},
    "answer_lengths": {...}
  },
  "gemma": {
    "overall": {...},
    "per_question_type": {...},
    "per_compound": {...},
    "answer_lengths": {...}
  },
  "comparison": {
    "overall": {
      "qwen_vs_gemma": {...},
      "qwen_vs_openai": {...},
      "gemma_vs_openai": {...}
    },
    "per_question_type": {...}
  }
}
```

## Notes
- Both scripts should follow the same structure as `bleu_score_calculator_regenerated.py` and `bertscore_calculator_regenerated.py`
- Handle missing files gracefully
- Include progress indicators for long-running calculations
- Use the same tokenization and BLEU calculation methods as existing scripts
- Use the same BERTScore model and configuration as existing scripts

