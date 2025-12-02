# Missing Information Answers Analyzer

## Overview

This program analyzes filtered answers from Gemma-3 that contain "missing information" responses to determine whether:
1. **Answer EXISTS in chunks** → Gemma model failed to use context (MODEL_FAILURE)
2. **Answer NOT in chunks** → Wrong chunks retrieved (RETRIEVAL_FAILURE)

## How It Works

### 1. Question Classification
The program classifies each question into types:
- **Q2**: Chemical formula and elements questions
- **Q3**: Development/history questions (e.g., "Why was X developed?")
- **Q4**: Properties questions (e.g., melting point, boiling point)

### 2. Compound Name Extraction
Extracts compound names from questions using pattern matching:
- Extracts names after "of", "about", "for"
- Handles parentheses (common names)
- Handles complex names like "2-(4-Isobutylphenyl)propionic acid"

### 3. Context Analysis

#### For Q2 (Formula & Elements):
- Searches for "FORMULA:" labels
- Searches for chemical formula patterns (e.g., C13H18O2)
- Searches for "MOLECULAR WEIGHT:" labels
- Searches for "ELEMENTS:" labels

#### For Q3 (Development/History):
- Searches for development keywords ("developed", "discovered", "created")
- Searches for company/researcher names
- Searches for years (1900-2100)
- Searches for purpose keywords ("because", "to", "for")

#### For Q4 (Properties):
- Searches for property labels (MELTING POINT, BOILING POINT, SOLUBILITY)
- Searches for molecular weight
- Searches for state (Solid/Liquid/Gas)

### 4. Classification Logic

```
1. Extract compound name from question
2. Check if compound name exists in context
   - If NO → RETRIEVAL_FAILURE (wrong compound retrieved)
   - If YES → Continue
3. Based on question type, search for expected information
   - If found → MODEL_FAILURE (info exists, model didn't use it)
   - If not found → RETRIEVAL_FAILURE (wrong chunks retrieved)
```

## Usage

```bash
cd /home/himanshu/dev
python3 code/analyze_missing_info_answers.py
```

## Output

### 1. JSON Results (`analysis_results.json`)
Contains:
- **Summary statistics**: Total analyzed, classification counts, rates
- **Detailed analyses**: Per-answer analysis with evidence

### 2. Markdown Report (`analysis_report.md`)
Contains:
- Summary statistics
- Breakdown by question type
- Sample analyses (first 10 of each type)

## Output Structure

### Per-Answer Analysis:
```json
{
  "file": "2_Ibuprofen__answers.json",
  "question_idx": 2,
  "question": "What is the chemical formula...",
  "question_type": "Q2",
  "compound_names": ["ibuprofen", "2-(4-Isobutylphenyl)propionic acid"],
  "compound_found_in_context": true,
  "compound_mention_count": 5,
  "matched_compound_names": ["ibuprofen"],
  "classification": "MODEL_FAILURE",
  "confidence": "high",
  "evidence": {
    "formula_found": true,
    "formulas": ["C13H18O2"],
    "molecular_weight_found": true,
    "weights": ["206.28"],
    "reason": "Formula or molecular weight found in context"
  }
}
```

### Classifications:
- **MODEL_FAILURE**: Information exists in context, but model failed to use it
- **RETRIEVAL_FAILURE**: Information not found in retrieved chunks (wrong chunks)
- **NO_CONTEXT**: No context available for analysis
- **UNKNOWN_TYPE**: Question type could not be classified

### Confidence Levels:
- **high**: Strong evidence (e.g., compound not found, or formula explicitly found)
- **medium**: Moderate evidence (e.g., compound found but specific info not found)
- **low**: Weak evidence (e.g., unknown question type)

## Example Use Cases

### Case 1: Model Failure
- **Question**: "What is the chemical formula of ibuprofen?"
- **Gemma Answer**: "Information not found in context"
- **Analysis**: Formula "C13H18O2" found in context
- **Classification**: MODEL_FAILURE
- **Action**: Improve prompt or model behavior

### Case 2: Retrieval Failure
- **Question**: "What is the chemical formula of ibuprofen?"
- **Gemma Answer**: "Information not found in context"
- **Analysis**: Compound "ibuprofen" not found in any chunks
- **Classification**: RETRIEVAL_FAILURE
- **Action**: Improve ChromaDB search or query formulation

## Limitations

1. **Compound Name Extraction**: May miss some compound name variations
2. **Pattern Matching**: Relies on specific patterns that may not cover all cases
3. **Question Classification**: May misclassify ambiguous questions
4. **Semantic Understanding**: Does not use embeddings/semantic similarity (future enhancement)

## Future Enhancements

1. Use embeddings to check semantic similarity between question and context
2. Compare with Qwen's answers to verify expected answers
3. Add more sophisticated compound name extraction
4. Add confidence scoring based on multiple factors
5. Generate visualizations of failure types

