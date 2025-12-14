# Exploratory Data Analysis (EDA) Plan
## Individual Compounds Dataset

**Data Source:** `/home/himanshu/MSC_FINAL/dev/test/test/data/processed/individual_compounds/`  
**Output Directory:** `/home/himanshu/MSC_FINAL/dev/EDA/`

---

## 1. Dataset Overview

### 1.1 Basic Statistics
- Total number of compounds
- File naming patterns
- Data completeness check
- Missing fields analysis

### 1.2 Data Structure Analysis
- JSON schema validation
- Field presence/absence across all files
- Data type consistency

---

## 2. Compound Metadata Analysis

### 2.1 Compound Identification
- Compound ID distribution
- Name patterns and variations
- Name length statistics
- Special characters in names

### 2.2 Page Information
- Arabic page range analysis (start/end)
- PDF page mapping
- Total pages per compound distribution
- Page span statistics

---

## 3. Content Analysis

### 3.1 Text Length Analysis
- Main entry content length distribution
- Comprehensive text length distribution
- Length ratio analysis (comprehensive/main)
- Outliers identification (very short/long entries)

### 3.2 Text Content Quality
- Word count statistics
- Sentence count (approximate)
- Content structure analysis
- Key sections presence (FORMULA, ELEMENTS, STATE, etc.)

### 3.3 Chemical Information Extraction
- Formula patterns analysis
- Element frequency analysis
- Compound type distribution
- State distribution (Solid/Liquid/Gas)
- Molecular weight statistics (if extractable)

---

## 4. Reference Analysis

### 4.1 Reference Counts
- Total references per compound distribution
- Reference count vs. compound ID correlation
- Compounds with zero references
- Compounds with high reference counts

### 4.2 Reference Types
- Reference type distribution (timeline, etc.)
- Reference type combinations
- Most common reference types

### 4.3 Reference Details
- Page number distribution in references
- Reference context length analysis
- Found variations analysis (case sensitivity, naming)

---

## 5. Comprehensive Text Analysis

### 5.1 Section Breakdown
- Main entry vs. other references ratio
- Timeline references analysis
- Cross-reference patterns

### 5.2 Content Enrichment
- Additional content beyond main entry
- Reference context quality
- Information completeness

---

## 6. Metadata Analysis

### 6.1 Source Information
- Source file consistency
- Extraction method analysis
- Timestamp patterns

### 6.2 Index Information
- Compound index distribution
- Total compounds consistency
- Index gaps or duplicates

---

## 7. Statistical Analysis

### 7.1 Descriptive Statistics
- Mean, median, mode for all numeric fields
- Standard deviation and variance
- Quartiles and percentiles
- Skewness and kurtosis

### 7.2 Correlation Analysis
- Text length vs. reference count
- Page span vs. content length
- Compound ID vs. various metrics

### 7.3 Distribution Analysis
- Histograms for all numeric fields
- Box plots for outlier detection
- Cumulative distribution functions

---

## 8. Data Quality Assessment

### 8.1 Completeness
- Missing field analysis
- Incomplete entries
- Truncated content detection

### 8.2 Consistency
- Naming convention consistency
- Format consistency
- Data type consistency

### 8.3 Anomaly Detection
- Unusual patterns
- Outliers identification
- Data integrity issues

---

## 9. Visualization Plan

### 9.1 Distribution Plots
- Text length distributions (histograms, box plots)
- Reference count distributions
- Page span distributions
- Compound type pie charts

### 9.2 Relationship Plots
- Scatter plots (length vs. references)
- Correlation heatmaps
- Time series (if applicable)

### 9.3 Summary Dashboards
- Overview statistics dashboard
- Key metrics summary
- Quality metrics visualization

---

## 10. Output Deliverables

### 10.1 Reports
- `eda_summary_report.md` - Executive summary
- `eda_detailed_report.md` - Comprehensive analysis
- `data_quality_report.md` - Quality assessment

### 10.2 Data Files
- `eda_statistics.json` - All computed statistics
- `eda_summary.csv` - Summary table
- `anomalies.json` - Detected anomalies

### 10.3 Visualizations
- `plots/` directory with all generated charts
- Interactive HTML dashboard (optional)

### 10.4 Scripts
- `01_basic_statistics.py` - Basic dataset overview
- `02_content_analysis.py` - Text and content analysis
- `03_reference_analysis.py` - Reference patterns
- `04_metadata_analysis.py` - Metadata exploration
- `05_visualizations.py` - Generate all plots
- `06_data_quality.py` - Quality assessment
- `07_generate_reports.py` - Report generation
- `run_all_eda.py` - Master script to run all analyses

---

## 11. Analysis Workflow

1. **Data Loading & Validation**
   - Load all JSON files
   - Validate structure
   - Check for errors

2. **Basic Statistics**
   - Count, completeness
   - Basic distributions

3. **Content Analysis**
   - Text metrics
   - Chemical information extraction

4. **Reference Analysis**
   - Reference patterns
   - Cross-references

5. **Metadata Analysis**
   - Source tracking
   - Extraction method analysis

6. **Statistical Analysis**
   - Descriptive stats
   - Correlations
   - Distributions

7. **Quality Assessment**
   - Completeness check
   - Consistency validation
   - Anomaly detection

8. **Visualization**
   - Generate all plots
   - Create dashboards

9. **Report Generation**
   - Compile statistics
   - Generate markdown reports
   - Export data files

---

## 12. Tools & Libraries

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Basic plotting
- **seaborn** - Statistical visualizations
- **json** - JSON file handling
- **pathlib** - File path management
- **statistics** - Statistical functions
- **re** - Pattern matching for chemical formulas
- **collections** - Counter, defaultdict for aggregations

---

## 13. Success Criteria

- ✅ All 178 compound files analyzed
- ✅ Comprehensive statistics computed
- ✅ All visualizations generated
- ✅ Quality issues identified
- ✅ Reports generated in markdown format
- ✅ Data exported in reusable formats (JSON, CSV)
- ✅ Anomalies documented

---

## 14. Timeline Estimate

- Basic statistics: 1 hour
- Content analysis: 2 hours
- Reference analysis: 1.5 hours
- Metadata analysis: 1 hour
- Statistical analysis: 2 hours
- Quality assessment: 1.5 hours
- Visualization: 2 hours
- Report generation: 1 hour

**Total Estimated Time:** ~12 hours

---

## 15. Next Steps

1. Create directory structure in `dev/EDA/`
2. Implement scripts in order (01-07)
3. Run analyses and validate results
4. Generate visualizations
5. Compile reports
6. Review and refine

