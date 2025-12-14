# Text Length EDA Summary

## Overview

Script `07_text_length_eda.py` provides general exploratory data analysis for `main_entry_length` and `comprehensive_text_length` **without RAG-specific character limits**. This focuses on pure data distribution analysis.

---

## Key Differences from RAG Pipeline EDA

### Removed
- ❌ Q1-Q4 character limit references (600, 1000, 1800, 2000)
- ❌ RAG compatibility analysis
- ❌ Token limit calculations
- ❌ Chunking analysis based on limits

### Focused On
- ✅ Pure statistical analysis (mean, median, percentiles)
- ✅ Distribution characteristics (skewness, kurtosis)
- ✅ Comparison between main entry and comprehensive text
- ✅ Enrichment ratios and differences
- ✅ General visualizations without limit lines

---

## Generated Visualizations

### 1. Main Entry Length Analysis
**File:** `main_entry_length_analysis.png`
- Histogram with mean/median lines
- Box plot showing quartiles and outliers
- No character limit references

### 2. Comprehensive Text Length Analysis
**File:** `comprehensive_text_length_analysis.png`
- Histogram with mean/median lines
- Box plot showing quartiles and outliers
- No character limit references

### 3. Side-by-Side Comparison
**File:** `text_length_comparison.png`
- Direct comparison of both distributions
- Mean and median indicators

### 4. Scatter Plot
**File:** `main_vs_comprehensive_scatter.png`
- Relationship between main entry and comprehensive text
- Linear regression line
- Reference line (y=x)

### 5. Cumulative Distribution
**File:** `text_length_cumulative_distribution.png`
- Cumulative percentage curves
- Shows what % of compounds are below each length

### 6. Overlaid Histograms
**File:** `text_length_overlaid_histograms.png`
- Both distributions on same plot
- Easy visual comparison

### 7. Enrichment Ratio Distribution
**File:** `enrichment_ratio_distribution.png`
- Distribution of comprehensive/main entry ratios
- Shows how much longer comprehensive text is

### 8. Box Plot Comparison
**File:** `text_length_boxplot_comparison.png`
- Side-by-side box plots
- Quartiles, medians, means, outliers

---

## Key Statistics

### Main Entry Length
- **Mean:** 2,692 characters
- **Median:** 2,473 characters
- **Range:** 1,002 - 5,546 characters
- **Std Dev:** 1,305 characters

### Comprehensive Text Length
- **Mean:** 6,529 characters
- **Median:** 4,855 characters
- **Range:** 1,601 - 124,356 characters
- **Std Dev:** 8,789 characters

### Enrichment
- **Mean Ratio:** 3.00x (comprehensive is 3x longer on average)
- **Median Ratio:** 2.02x
- **147 compounds** have >1.5x enrichment

---

## Output Files

### Data Files
- `07_text_length_analysis.json` - Complete analysis results

### Reports
- `07_text_length_eda_report.md` - Comprehensive markdown report

### Visualizations
- `plots/07_text_length_analysis/` - 8 visualization files

---

## Usage

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
source /home/himanshu/llama-env/bin/activate
python scripts/07_text_length_eda.py
```

---

## When to Use This vs RAG Pipeline EDA

### Use `07_text_length_eda.py` (This Script)
- General data exploration
- Understanding data distributions
- Statistical analysis without application-specific limits
- Research and documentation

### Use `06_rag_pipeline_eda.py` (RAG-Specific)
- When planning RAG pipeline implementation
- Need to understand compatibility with specific character limits
- Evaluating chunking strategies
- Pipeline optimization

---

## Insights

1. **Main Entry:** More consistent length (std dev: 1,305) - good for uniform processing
2. **Comprehensive Text:** Highly variable (std dev: 8,789) - includes one outlier (124K chars)
3. **Enrichment:** Comprehensive text provides 3x more context on average
4. **Distribution:** Both are right-skewed (long tail on right) - most compounds are shorter than mean

---

## Plot Locations

All plots saved to: `/home/himanshu/MSC_FINAL/dev/EDA/plots/07_text_length_analysis/`

