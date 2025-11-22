# BLEU Score Calculation Plan: QWEN vs OpenAI Answers

## Overview
Compare QWEN-generated answers (candidate) against OpenAI-generated answers (reference/baseline) using BLEU scores.

## Data Sources

### 1. QWEN Answers (Candidate)
- **Location**: `/home/himanshu/dev/output/qwen/*__answers.json`
- **Structure**: 
  ```json
  {
    "source_file": "1_13-Butadiene.json",
    "model": "qwen",
    "answers": [
      {"question": "...", "answer": "...", "latency_s": ...},
      ...
    ]
  }
  ```
- **Count**: 178 files
- **Answers per file**: 4 (Q1-Q4)

### 2. OpenAI Answers (Reference/Baseline)
- **Location**: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components/*.json`
- **Structure**:
  ```json
  {
    "qa_pairs": [
      {"question": "...", "answer": "...", "updated_by": "openai_api", ...},
      ...
    ]
  }
  ```
- **Count**: 178 files
- **Answers per file**: 4 (Q1-Q4)
- **Note**: First answer (Q1) was updated by OpenAI API

## Matching Strategy

### File Matching
- QWEN files: `{N}_{Compound}__answers.json`
- OpenAI files: `{N}_{Compound}.json`
- Match by: Extract numeric prefix and compound name from filenames

### Question Matching
- Both sets have 4 questions per file
- Questions are in the same order (Q1, Q2, Q3, Q4)
- Match by: Index position (0, 1, 2, 3) within each file

### Matching Algorithm
1. For each QWEN file:
   - Extract base filename (remove `__answers.json`)
   - Find corresponding OpenAI file: `{base}.json`
   - Match answers by array index (Q1=0, Q2=1, Q3=2, Q4=3)
   - Verify questions match (optional sanity check)

## BLEU Score Calculation

### What is BLEU?
- BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between candidate and reference
- Range: 0.0 (no overlap) to 1.0 (perfect match)
- Commonly used for machine translation, adaptable for text generation evaluation

### BLEU Components
1. **Precision**: Fraction of candidate n-grams that appear in reference
2. **N-gram orders**: Typically 1-gram, 2-gram, 3-gram, 4-gram
3. **Brevity Penalty**: Penalizes candidates shorter than reference
4. **Smoothing**: Handle cases with zero n-gram matches

### Implementation Approach

#### Option 1: NLTK BLEU (Recommended)
- Library: `nltk.translate.bleu_score`
- Pros: Standard implementation, well-tested
- Cons: Requires tokenization

#### Option 2: SacreBLEU
- Library: `sacrebleu`
- Pros: More robust, handles edge cases better
- Cons: Additional dependency

#### Option 3: Custom Implementation
- Pros: Full control
- Cons: More work, potential bugs

### Tokenization Strategy
- **Method**: Word-level tokenization (split on whitespace)
- **Normalization**: 
  - Lowercase (optional, but recommended for fair comparison)
  - Remove punctuation (optional)
  - Handle chemical formulas (preserve subscripts/superscripts)

### BLEU Variants to Calculate

1. **Sentence-level BLEU**: One score per question-answer pair
2. **Corpus-level BLEU**: Aggregate score across all questions
3. **Question-type BLEU**: Separate scores for Q1, Q2, Q3, Q4
4. **Compound-level BLEU**: Average BLEU per compound

## Implementation Plan

### Step 1: Data Loading
```python
def load_qwen_answers(qwen_dir):
    """Load all QWEN answer files"""
    # Return: dict[filename_base] -> list of answers

def load_openai_answers(openai_dir):
    """Load all OpenAI answer files"""
    # Return: dict[filename_base] -> list of qa_pairs
```

### Step 2: Matching
```python
def match_qa_pairs(qwen_data, openai_data):
    """Match QWEN and OpenAI answers by file and question index"""
    # Return: list of (qwen_answer, openai_answer, question, file_id) tuples
```

### Step 3: BLEU Calculation
```python
def calculate_bleu_scores(matched_pairs):
    """Calculate BLEU scores for all matched pairs"""
    # Return: dict with various BLEU metrics
```

### Step 4: Aggregation & Reporting
```python
def aggregate_and_report(bleu_scores):
    """Calculate statistics and generate report"""
    # Metrics:
    # - Overall corpus BLEU
    # - Per-question-type BLEU (Q1, Q2, Q3, Q4)
    # - Per-compound BLEU
    # - Distribution statistics (mean, median, std, min, max)
```

## Expected Output

### Metrics to Report

1. **Overall Corpus BLEU**
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4
   - Average across all 178 × 4 = 712 question pairs

2. **Per-Question-Type BLEU**
   - Q1 (image-based): Separate score
   - Q2, Q3, Q4 (text-only): Separate scores
   - Compare performance across question types

3. **Per-Compound Statistics**
   - Mean BLEU per compound
   - Identify compounds with highest/lowest BLEU scores

4. **Distribution Statistics**
   - Mean, median, standard deviation
   - Min, max, quartiles
   - Histogram of BLEU scores

5. **Detailed Report**
   - CSV file with per-question BLEU scores
   - JSON summary with aggregated metrics
   - Visualizations (histograms, box plots)

## File Structure

```
dev/code/
├── bleu_score_calculator.py      # Main script
├── bleu_results/
│   ├── per_question_bleu.csv     # Detailed scores
│   ├── summary_metrics.json      # Aggregated metrics
│   └── visualizations/           # Charts and plots
└── BLEU_SCORE_PLAN.md           # This file
```

## Dependencies

```python
# Required packages
nltk>=3.8
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0  # Optional, for better visualizations
```

## Implementation Details

### Tokenization Function
```python
def tokenize_answer(text):
    """Tokenize answer text for BLEU calculation"""
    # Lowercase
    # Split on whitespace
    # Optional: Remove punctuation
    # Return: list of tokens
```

### BLEU Calculation Function
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_sentence_bleu(candidate, reference):
    """Calculate BLEU score for a single answer pair"""
    candidate_tokens = tokenize_answer(candidate)
    reference_tokens = tokenize_answer(reference)
    
    # Use smoothing to handle zero matches
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU-1 through BLEU-4
    bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, 
                          weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, 
                          weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, 
                          weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, 
                          smoothing_function=smoothing)
    
    return {
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_3': bleu_3,
        'bleu_4': bleu_4
    }
```

### Corpus-Level BLEU
```python
from nltk.translate.bleu_score import corpus_bleu

def calculate_corpus_bleu(all_candidates, all_references):
    """Calculate corpus-level BLEU score"""
    # Format: list of token lists
    candidates = [tokenize_answer(c) for c in all_candidates]
    references = [[tokenize_answer(r)] for r in all_references]
    
    bleu_4 = corpus_bleu(references, candidates)
    return bleu_4
```

## Validation Steps

1. **File Count Check**: Verify both sets have 178 files
2. **Question Matching**: Verify questions match between sets
3. **Answer Length Check**: Compare answer lengths (QWEN vs OpenAI)
4. **Sample Inspection**: Manually review a few pairs to ensure matching is correct

## Edge Cases to Handle

1. **Missing Files**: If a QWEN file has no corresponding OpenAI file
2. **Mismatched Questions**: If questions don't match (log warning)
3. **Empty Answers**: Handle empty or very short answers
4. **Special Characters**: Handle chemical formulas, subscripts, etc.
5. **Truncated Answers**: QWEN answers might be truncated (max_tokens=128)

## Interpretation Guidelines

### BLEU Score Ranges
- **0.0 - 0.3**: Poor similarity (very different answers)
- **0.3 - 0.5**: Moderate similarity (some overlap)
- **0.5 - 0.7**: Good similarity (substantial overlap)
- **0.7 - 0.9**: High similarity (very similar content)
- **0.9 - 1.0**: Excellent similarity (nearly identical)

### Important Notes
- BLEU measures n-gram overlap, not semantic similarity
- Low BLEU doesn't necessarily mean wrong answer (could be different phrasing)
- High BLEU suggests similar wording and content
- For QA evaluation, BLEU-1 and BLEU-2 are often most informative

## Next Steps

1. ✅ Create plan (this document)
2. ⬜ Implement data loading functions
3. ⬜ Implement matching logic
4. ⬜ Implement BLEU calculation
5. ⬜ Add aggregation and statistics
6. ⬜ Generate reports and visualizations
7. ⬜ Validate results
8. ⬜ Document findings

---

**Created**: 2025-01-07  
**Status**: Planning Complete - Ready for Implementation

