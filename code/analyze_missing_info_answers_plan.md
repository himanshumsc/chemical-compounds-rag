# Plan: Analyze Missing Information Answers

## Objective
Determine for each "missing information" answer whether:
1. **Answer EXISTS in chunks** → Gemma model failed to use context (Model Issue)
2. **Answer NOT in chunks** → Wrong chunks retrieved (Retrieval Issue)

## Input Data
- Filtered JSON files from `/home/himanshu/dev/output/gemma3_rag_concise_missing_ans/`
- Each file contains answers with `filtered_as_missing_info: true`
- Each answer has `rag_context_formatted` and `rag_chunks`

## Analysis Strategy

### 1. Question Type Classification
Classify questions into types:
- **Q2**: Chemical formula and elements
- **Q3**: Development/history/uses
- **Q4**: Properties/characteristics

### 2. Answer Extraction Strategy

#### For Q2 (Formula & Elements):
- Extract compound name from question
- Search for:
  - Chemical formula patterns (e.g., "C13H18O2", "CH3COOH")
  - "FORMULA:" label
  - "ELEMENTS:" label
  - Element names mentioned
- Expected: Formula and element list should be in context

#### For Q3 (Development/History):
- Extract compound name
- Search for:
  - Development timeline keywords ("developed", "discovered", "created")
  - Company/researcher names
  - Year mentions
  - Purpose/reason keywords ("because", "to", "for")
- Expected: Historical information about development

#### For Q4 (Properties):
- Extract compound name
- Search for:
  - Property keywords (molecular weight, melting point, boiling point, solubility)
  - Numerical values
  - Physical/chemical properties
- Expected: Specific property values or descriptions

### 3. Search Methods

#### Method 1: Entity Extraction
- Extract compound name from question
- Check if compound name appears in context (case-insensitive)
- If not found → Retrieval issue (wrong compound retrieved)

#### Method 2: Pattern Matching
- For formulas: Regex patterns for chemical formulas
- For properties: Keywords + numerical patterns
- For history: Temporal keywords + compound name

#### Method 3: Semantic Similarity (Optional)
- Use embeddings to check if question and context are semantically related
- Lower similarity → Retrieval issue

### 4. Classification Logic

```
For each filtered answer:
  1. Extract question type (Q2, Q3, Q4)
  2. Extract compound name from question
  3. Check if compound name exists in context
     - If NO → Classification: "RETRIEVAL_FAILURE" (wrong compound)
     - If YES → Continue
  4. Based on question type, search for expected information:
     - Q2: Formula + Elements
     - Q3: Development history
     - Q4: Properties
  5. If expected info found:
     - Classification: "MODEL_FAILURE" (info exists, model didn't use it)
  6. If expected info NOT found:
     - Classification: "RETRIEVAL_FAILURE" (wrong chunks retrieved)
```

### 5. Output Structure

#### Per-Answer Analysis:
```json
{
  "file": "2_Ibuprofen__answers.json",
  "question_idx": 2,
  "question": "...",
  "compound_name": "ibuprofen",
  "compound_found_in_context": true/false,
  "question_type": "Q2/Q3/Q4",
  "expected_info_type": "formula_and_elements",
  "expected_info_found": true/false,
  "classification": "MODEL_FAILURE" | "RETRIEVAL_FAILURE",
  "confidence": "high" | "medium" | "low",
  "evidence": {
    "compound_mentions": 5,
    "formula_found": true,
    "elements_found": ["Carbon", "Hydrogen", "Oxygen"],
    "relevant_chunks": [0, 1, 2]
  }
}
```

#### Summary Statistics:
- Total filtered answers analyzed
- Model failures count (info exists)
- Retrieval failures count (info missing)
- Breakdown by question type
- Confidence distribution

## Implementation Steps

1. **Load filtered JSON files**
2. **Extract compound names** from questions
3. **Classify question types** (Q2/Q3/Q4)
4. **Search context** for compound and expected info
5. **Classify** each answer
6. **Generate report** with statistics and details
7. **Save results** to JSON and markdown

## Edge Cases to Handle

1. **Compound name variations**: "ibuprofen" vs "2-(4-Isobutylphenyl)propionic acid"
2. **Partial matches**: Compound mentioned but not the specific info requested
3. **Low relevance chunks**: Chunks exist but scores are very low
4. **Q1 answers**: Skip (image-based, no text chunks)
5. **Empty context**: Handle gracefully

## Files to Create

1. `analyze_missing_info_answers.py` - Main analysis program
2. `answer_analyzer.py` - Core analysis logic (optional, can be in main file)
3. Output: `missing_info_analysis_results.json` and `missing_info_analysis_report.md`

## Validation

- Manual spot-check: Review 10-20 cases manually
- Compare with original QA files to verify expected answers
- Check if classifications make sense

