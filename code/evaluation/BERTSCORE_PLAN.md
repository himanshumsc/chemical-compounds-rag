# BERTScore Calculation Plan: QWEN vs OpenAI Answers

## Overview
Calculate BERTScore to measure semantic similarity between QWEN-generated answers (candidate) and OpenAI-generated answers (reference/baseline). BERTScore uses contextual embeddings from BERT models to capture semantic meaning, making it more suitable for evaluating answer quality than n-gram overlap metrics like BLEU.

## Why BERTScore?

### Advantages over BLEU
1. **Semantic Understanding**: Captures meaning, not just word overlap
2. **Synonym Recognition**: Recognizes that "compound" and "chemical compound" are similar
3. **Paraphrasing**: Understands different phrasings of same content
4. **Context-Aware**: Uses contextual embeddings (BERT) vs. surface-level n-grams
5. **Better for QA**: More appropriate for evaluating answer quality

### BERTScore Components
- **Precision**: How much of candidate is semantically similar to reference
- **Recall**: How much of reference is covered by candidate
- **F1**: Harmonic mean of precision and recall

## Data Sources (Same as BLEU)

### 1. QWEN Answers (Candidate)
- **Location**: `/home/himanshu/dev/output/qwen/*__answers.json`
- **Structure**: Same as BLEU analysis
- **Count**: 178 files, 712 question-answer pairs

### 2. OpenAI Answers (Reference/Baseline)
- **Location**: `/home/himanshu/dev/test/data/processed/qa_pairs_individual_components/*.json`
- **Structure**: Same as BLEU analysis
- **Count**: 178 files, 712 question-answer pairs

## Matching Strategy (Same as BLEU)

### File Matching
- QWEN files: `{N}_{Compound}__answers.json`
- OpenAI files: `{N}_{Compound}.json`
- Match by: Extract numeric prefix and compound name

### Question Matching
- Both sets have 4 questions per file
- Match by: Index position (0, 1, 2, 3) within each file
- Same matching logic as BLEU calculation

## BERTScore Implementation

### Library Options

#### Option 1: bert-score (Recommended)
- **Package**: `bert-score`
- **Pros**: 
  - Easy to use, well-maintained
  - Supports multiple BERT models
  - Fast computation with GPU support
  - Returns precision, recall, F1
- **Cons**: Requires model download (~400MB)

#### Option 2: Custom with transformers
- **Package**: `transformers`, `torch`
- **Pros**: Full control
- **Cons**: More implementation work

### Recommended Model

**`microsoft/deberta-xlarge-mnli`** or **`roberta-large`**
- Good balance of accuracy and speed
- Works well for scientific/technical text
- Alternative: `bert-base-uncased` (faster, less accurate)

### BERTScore Calculation Process

1. **Load BERT Model**: Load pre-trained BERT model and tokenizer
2. **Tokenize**: Tokenize both candidate and reference texts
3. **Generate Embeddings**: Get contextual embeddings for each token
4. **Calculate Similarity**: 
   - Cosine similarity between candidate and reference embeddings
   - Greedy matching (each candidate token matched to best reference token)
5. **Compute Metrics**:
   - **Precision**: Average similarity of candidate tokens
   - **Recall**: Average similarity of reference tokens
   - **F1**: Harmonic mean of precision and recall

## Implementation Plan

### Step 1: Data Loading
```python
def load_qwen_answers(qwen_dir):
    """Load all QWEN answer files"""
    # Same as BLEU implementation

def load_openai_answers(openai_dir):
    """Load all OpenAI answer files"""
    # Same as BLEU implementation
```

### Step 2: Matching
```python
def match_qa_pairs(qwen_data, openai_data):
    """Match QWEN and OpenAI answers"""
    # Same matching logic as BLEU
    # Return: list of (qwen_answer, openai_answer, metadata) tuples
```

### Step 3: BERTScore Calculation
```python
from bert_score import score

def calculate_bertscore(candidates, references, model_type='microsoft/deberta-xlarge-mnli'):
    """
    Calculate BERTScore for candidate-reference pairs.
    
    Args:
        candidates: List of candidate texts (QWEN answers)
        references: List of reference texts (OpenAI answers)
        model_type: BERT model to use
    
    Returns:
        Dictionary with precision, recall, F1 scores
    """
    P, R, F1 = score(
        candidates, 
        references, 
        lang='en',
        model_type=model_type,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    return {
        'precision': P.tolist(),  # Per-sentence precision
        'recall': R.tolist(),      # Per-sentence recall
        'f1': F1.tolist()          # Per-sentence F1
    }
```

### Step 4: Aggregation & Reporting
```python
def aggregate_bertscore_stats(bertscore_results):
    """Calculate statistics from BERTScore results"""
    # Metrics:
    # - Overall corpus F1 (mean across all pairs)
    # - Per-question-type F1 (Q1, Q2, Q3, Q4)
    # - Per-compound F1
    # - Distribution statistics
    # - Precision vs Recall analysis
```

## Expected Output

### Metrics to Report

1. **Overall Corpus BERTScore**
   - Mean Precision, Recall, F1 across all 712 pairs
   - Median, std, min, max for each metric

2. **Per-Question-Type BERTScore**
   - Q1 (image-based): Separate F1, Precision, Recall
   - Q2, Q3, Q4 (text-only): Separate scores
   - Compare semantic similarity across question types

3. **Per-Compound BERTScore**
   - Mean F1 per compound
   - Identify compounds with highest/lowest semantic similarity

4. **Precision vs Recall Analysis**
   - Which model covers more content (recall)
   - Which model is more precise (precision)
   - Trade-off analysis

5. **Distribution Statistics**
   - Mean, median, std, min, max for F1
   - Histogram of F1 scores
   - Comparison with BLEU distribution

6. **Detailed Report**
   - CSV file with per-question BERTScore (F1, P, R)
   - JSON summary with aggregated metrics
   - Visualizations (histograms, scatter plots, precision-recall curves)

## File Structure

```
dev/code/bleu_evaluation/
├── scripts/
│   ├── bleu_score_calculator.py      # Existing BLEU calculator
│   └── bertscore_calculator.py       # New BERTScore calculator
├── results/
│   ├── per_question_bleu.csv         # Existing BLEU results
│   ├── per_question_bertscore.csv    # New BERTScore results
│   ├── summary_metrics.json          # Existing (BLEU)
│   └── summary_bertscore.json        # New (BERTScore)
└── visualizations/
    ├── bleu_distributions.png        # Existing
    └── bertscore_distributions.png   # New
```

## Dependencies

```python
# Required packages
bert-score>=0.3.13
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## Implementation Details

### BERTScore Function
```python
from bert_score import score
import torch

def calculate_bertscore_batch(candidates, references, model_type='microsoft/deberta-xlarge-mnli'):
    """
    Calculate BERTScore for a batch of candidate-reference pairs.
    
    Args:
        candidates: List of candidate texts
        references: List of reference texts (same length as candidates)
        model_type: BERT model identifier
    
    Returns:
        Dictionary with precision, recall, F1 arrays
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Calculate BERTScore
    P, R, F1 = score(
        candidates,
        references,
        lang='en',
        model_type=model_type,
        device=device,
        batch_size=32,  # Process in batches for efficiency
        verbose=True
    )
    
    return {
        'precision': P.cpu().numpy(),
        'recall': R.cpu().numpy(),
        'f1': F1.cpu().numpy()
    }
```

### Model Selection

**Recommended Models** (in order of preference):

1. **`microsoft/deberta-xlarge-mnli`** ⭐ Recommended
   - Best accuracy for semantic similarity
   - Good for technical/scientific text
   - Larger model (~1.5GB download)

2. **`roberta-large`**
   - Good balance of accuracy and speed
   - Widely used for BERTScore
   - Medium size (~1.3GB)

3. **`bert-base-uncased`**
   - Fastest, smallest (~440MB)
   - Lower accuracy
   - Good for quick testing

### Batch Processing

BERTScore can process multiple pairs efficiently:
- Use `batch_size=32` or higher for GPU
- Process all 712 pairs in batches
- Much faster than sentence-by-sentence

## Comparison with BLEU

### What BERTScore Adds

1. **Semantic Similarity**: Understands meaning, not just words
2. **Synonym Handling**: "compound" ≈ "chemical compound"
3. **Paraphrase Recognition**: Different phrasings of same content
4. **Context Awareness**: Understands word meaning in context

### Expected Differences

- **BERTScore F1** should be **higher** than BLEU-4 for semantically similar but differently phrased answers
- **Better correlation** with human judgment of answer quality
- **More fair comparison** when answers are correct but phrased differently

## Validation Steps

1. **File Count Check**: Verify both sets have 178 files
2. **Question Matching**: Verify questions match between sets
3. **Model Download**: Ensure BERT model is downloaded
4. **Sample Inspection**: Manually review a few pairs to ensure matching
5. **Score Validation**: Check that scores are in [0, 1] range

## Edge Cases to Handle

1. **Empty Answers**: Handle empty or very short answers
2. **Very Long Answers**: May need truncation for BERT context window
3. **Special Characters**: Chemical formulas, subscripts, etc.
4. **GPU Memory**: Large batches may need smaller batch_size
5. **Model Download**: First run will download model (~1.5GB)

## Interpretation Guidelines

### BERTScore Ranges

- **0.0 - 0.5**: Low semantic similarity
- **0.5 - 0.7**: Moderate semantic similarity
- **0.7 - 0.9**: High semantic similarity
- **0.9 - 1.0**: Very high semantic similarity (nearly identical meaning)

### Precision vs Recall

- **High Precision**: QWEN answers are semantically precise (few irrelevant parts)
- **High Recall**: QWEN answers cover most of OpenAI's content
- **High F1**: Good balance of precision and recall

### Comparison with BLEU

- **BERTScore > BLEU**: Answers are semantically similar but phrased differently
- **BERTScore ≈ BLEU**: Answers have both semantic and surface-level similarity
- **BERTScore < BLEU**: Unlikely, but possible if answers are semantically different

## Expected Results

Based on BLEU analysis:
- **BLEU-4**: 0.0907 (low phrase overlap)
- **Expected BERTScore F1**: 0.4-0.6 (moderate semantic similarity)
- **Rationale**: Answers are semantically similar but phrased differently

## Performance Considerations

### Computation Time
- **BERTScore**: ~2-5 seconds per batch of 32 pairs
- **Total time**: ~45-90 seconds for 712 pairs (with GPU)
- **Without GPU**: ~5-10 minutes

### Memory Requirements
- **Model size**: ~1.5GB (deberta-xlarge)
- **GPU memory**: ~2-4GB for batch processing
- **CPU memory**: ~4-8GB

## Next Steps

1. ✅ Create plan (this document)
2. ⬜ Install bert-score package
3. ⬜ Implement data loading (reuse BLEU code)
4. ⬜ Implement matching (reuse BLEU code)
5. ⬜ Implement BERTScore calculation
6. ⬜ Add aggregation and statistics
7. ⬜ Generate reports and visualizations
8. ⬜ Compare with BLEU results
9. ⬜ Document findings

## Code Structure

```python
# bertscore_calculator.py structure

import json
from pathlib import Path
from bert_score import score
import torch
import pandas as pd
import numpy as np

# Reuse data loading from BLEU
from bleu_score_calculator import load_qwen_answers, load_openai_answers, match_qa_pairs

def calculate_bertscore_scores(matched_pairs, model_type='microsoft/deberta-xlarge-mnli'):
    """Calculate BERTScore for all matched pairs"""
    # Extract candidates and references
    candidates = [p['qwen_answer'] for p in matched_pairs]
    references = [p['openai_answer'] for p in matched_pairs]
    
    # Calculate BERTScore
    P, R, F1 = score(candidates, references, ...)
    
    # Combine with metadata
    results = []
    for i, pair in enumerate(matched_pairs):
        results.append({
            **pair,
            'bertscore_precision': float(P[i]),
            'bertscore_recall': float(R[i]),
            'bertscore_f1': float(F1[i])
        })
    
    return results

def aggregate_bertscore_statistics(results):
    """Calculate aggregated statistics"""
    # Similar to BLEU aggregation
    # Return summary statistics
```

## Output Files

### 1. `per_question_bertscore.csv`
Columns:
- file_base, question_index, question
- qwen_answer, openai_answer
- qwen_length, openai_length
- bertscore_precision, bertscore_recall, bertscore_f1

### 2. `summary_bertscore.json`
Structure:
```json
{
  "overall": {
    "precision": {"mean": ..., "median": ..., ...},
    "recall": {"mean": ..., "median": ..., ...},
    "f1": {"mean": ..., "median": ..., ...}
  },
  "per_question_type": {
    "Q1": {"mean_f1": ..., "count": ...},
    ...
  },
  "per_compound": {...},
  "comparison_with_bleu": {...}
}
```

### 3. Visualizations
- F1 score distribution histogram
- Precision vs Recall scatter plot
- Per-question-type comparison (box plots)
- BERTScore vs BLEU correlation plot

## Integration with Existing BLEU Results

### Combined Analysis
- Load both BLEU and BERTScore results
- Compare metrics side-by-side
- Identify cases where BERTScore > BLEU (semantic similarity despite different phrasing)
- Generate combined report

### Correlation Analysis
- Calculate correlation between BLEU-4 and BERTScore F1
- Identify outliers (high BERTScore, low BLEU = semantic similarity, different phrasing)
- Visualize correlation

---

**Created**: 2025-01-07  
**Status**: Planning Complete - Ready for Implementation  
**Related**: BLEU_SCORE_PLAN.md, bleu_score_calculator.py

