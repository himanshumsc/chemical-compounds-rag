# RAG Pipeline EDA Report - Individual Compounds Dataset

This report analyzes data completeness and suitability for the multimodal RAG pipeline.

---

## Main Entry Length Analysis

### Statistics

- **Count:** 178 compounds
- **Mean:** 2692.0 characters
- **Median:** 2472.5 characters
- **Min:** 1,002 characters
- **Max:** 5,546 characters
- **Std Dev:** 1305.0 characters

### Percentiles

- **25th:** 1485.5 characters
- **50th (Median):** 2472.5 characters
- **75th:** 3983.5 characters
- **90th:** 4544.2 characters
- **95th:** 5063.3 characters
- **99th:** 5319.2 characters

### RAG Character Limit Compatibility

| Question | Character Limit | Compounds That Fit | Percentage |
|----------|----------------|-------------------|------------|
| Q1 | 600 | 0 | 0.0% |
| Q2 | 1000 | 0 | 0.0% |
| Q3 | 1800 | 69 | 38.8% |
| Q4 | 2000 | 70 | 39.3% |
| Exceeds Q4 | >2000 | 108 | 60.7% |

## Comprehensive Text Length Analysis

### Statistics

- **Count:** 178 compounds
- **Mean:** 6529.4 characters
- **Median:** 4854.5 characters
- **Min:** 1,601 characters
- **Max:** 124,356 characters

### RAG Character Limit Compatibility

| Question | Character Limit | Compounds That Fit | Percentage |
|----------|----------------|-------------------|------------|
| Q1 | 600 | 0 | 0.0% |
| Q2 | 1000 | 0 | 0.0% |
| Q3 | 1800 | 3 | 1.7% |
| Q4 | 2000 | 4 | 2.2% |
| Exceeds Q4 | >2000 | 174 | 97.8% |

## RAG Suitability Analysis

### Main Entry vs Comprehensive Text

- **Mean Ratio (comp/main):** 3.00x
- **Median Ratio:** 2.02x
- **Compounds with >1.5x enrichment:** 147

**Insight:** Comprehensive text provides significantly more context for RAG retrieval.

### Reference Enrichment

- **Mean References per Compound:** 26.6
- **Compounds with References:** 178
- **Compounds without References:** 0
- **High Reference Compounds (>20):** 47

## Context Window Analysis (Token-based)

### Main Entry (Estimated Tokens)

- **Mean:** 673.0 tokens
- **Median:** 618.1 tokens
- **Max:** 1386.5 tokens

### Token Limit Compatibility

| Question | Token Limit | Compounds That Fit |
|----------|-------------|-------------------|
| Q1 | 200 | 0 |
| Q2 | 333 | 20 |
| Q3 | 600 | 82 |
| Q4 | 666 | 100 |

## Chunking Analysis (for RAG Retrieval)

Assuming average chunk size of 800 characters:

### Main Entry Chunking

- **Mean Chunks per Compound:** 3.4
- **Median Chunks:** 3.1
- **Max Chunks:** 6
- **Single Chunk Compounds:** 0
- **Multi-Chunk Compounds:** 178

## Content Structure Analysis

### Section Presence (Important for RAG Retrieval)

| Section | Present | Coverage % |
|---------|---------|------------|
| FORMULA | 178 | 100.0% |
| ELEMENTS | 178 | 100.0% |
| STATE | 178 | 100.0% |
| MELTING_POINT | 178 | 100.0% |
| BOILING_POINT | 178 | 100.0% |
| SOLUBILITY | 178 | 100.0% |
| OVERVIEW | 178 | 100.0% |
| COMPOUND_TYPE | 177 | 99.4% |
| MOLECULAR_WEIGHT | 177 | 99.4% |
| HOW_IT_IS_MADE | 132 | 74.2% |
| COMMON_USES | 118 | 66.3% |
| HAZARDS | 118 | 66.3% |

### Content Completeness Score

(Based on presence of key sections: FORMULA, ELEMENTS, TYPE, STATE, MW, OVERVIEW, HOW_IT_IS_MADE, COMMON_USES)

- **Mean Score:** 7.4/8
- **Max Score:** 8/8
- **High Completeness (≥7):** 132 compounds
- **Low Completeness (<4):** 0 compounds

## Recommendations for RAG Pipeline

⚠️ **108 compounds exceed Q4 character limit** - Consider truncation or chunking strategies

📦 **178 compounds require multi-chunk retrieval** - Ensure RAG system handles chunk aggregation

✅ **Comprehensive text provides significantly more context** - Consider using comprehensive text for richer RAG retrieval

---

**Generated:** RAG Pipeline EDA Script
