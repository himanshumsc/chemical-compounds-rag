# Text Length EDA Report - Individual Compounds Dataset

This report analyzes `main_entry_length` and `comprehensive_text_length` distributions.

---

## Main Entry Length Analysis

### Descriptive Statistics

- **Count:** 178 compounds
- **Mean:** 2692.0 characters
- **Median:** 2472.5 characters
- **Standard Deviation:** 1305.0 characters
- **Min:** 1,002 characters
- **Max:** 5,546 characters
- **Total:** 479,184 characters

### Percentiles

| Percentile | Length (characters) |
|------------|---------------------|
| 10th | 1308.9 |
| 25th | 1485.5 |
| 50th (Median) | 2472.5 |
| 75th | 3983.5 |
| 90th | 4544.2 |
| 95th | 5063.3 |
| 99th | 5319.2 |

### Distribution Characteristics

- **Skewness:** 0.57
  - Positive = right-skewed (long tail on right)
  - Negative = left-skewed (long tail on left)
- **Kurtosis:** -1.00
  - >0 = heavy-tailed distribution
  - <0 = light-tailed distribution

## Comprehensive Text Length Analysis

### Descriptive Statistics

- **Count:** 178 compounds
- **Mean:** 6529.4 characters
- **Median:** 4854.5 characters
- **Standard Deviation:** 10108.2 characters
- **Min:** 1,601 characters
- **Max:** 124,356 characters
- **Total:** 1,162,226 characters

### Percentiles

| Percentile | Length (characters) |
|------------|---------------------|
| 10th | 3148.0 |
| 25th | 3893.8 |
| 50th (Median) | 4854.5 |
| 75th | 6554.0 |
| 90th | 8613.3 |
| 95th | 11046.9 |
| 99th | 28613.7 |

### Distribution Characteristics

- **Skewness:** 9.86
- **Kurtosis:** 109.03

## Comparison Analysis

### Comprehensive/Main Entry Ratio

- **Mean Ratio:** 3.00x
- **Median Ratio:** 2.02x
- **Min Ratio:** 1.02x
- **Max Ratio:** 86.18x
- **Std Dev:** 7.01x

**Interpretation:**
- Ratio > 1.0 = Comprehensive text is longer
- Ratio = 1.0 = Equal lengths
- Ratio < 1.0 = Main entry is longer (unusual)

### Length Difference (Comprehensive - Main Entry)

- **Mean Difference:** 3837.3 characters
- **Median Difference:** 2145.5 characters
- **Min Difference:** 100.0 characters
- **Max Difference:** 122913.0 characters

### Enrichment Categories

- **Significant Enrichment (>1.5x):** 147 compounds
- **Minimal Enrichment (1.0-1.5x):** 31 compounds
- **Equal Lengths (1.0x):** 0 compounds

## Key Insights

### Main Entry Length
- Average length: **2692 characters**
- Most compounds fall between 1,000-5,000 characters
- Suitable for direct use in most contexts

### Comprehensive Text Length
- Average length: **6529 characters**
- Significantly longer than main entry (typically 2-3x)
- Includes additional references and cross-references
- May require chunking for processing

### Relationship
- Comprehensive text is on average **3.00x longer** than main entry
- This enrichment comes from timeline references and cross-references
- Both text types are available for different use cases

---

**Generated:** Text Length EDA Script
