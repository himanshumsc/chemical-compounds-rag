# Exploratory Data Analysis (EDA) - Individual Compounds Dataset

This directory contains comprehensive EDA scripts and analysis for the individual compounds dataset located at:
`/home/himanshu/MSC_FINAL/dev/test/test/data/processed/individual_compounds/`

## Directory Structure

```
dev/EDA/
├── README.md                    # This file
├── EDA_PLAN.md                  # Comprehensive EDA plan
├── requirements.txt             # Python dependencies
├── scripts/                      # Analysis scripts
│   ├── 01_basic_statistics.py   # Basic dataset statistics
│   ├── 02_content_analysis.py   # Text and content analysis
│   ├── 03_reference_analysis.py # Reference patterns analysis
│   ├── 04_metadata_analysis.py   # Metadata exploration (to be implemented)
│   ├── 05_visualizations.py     # Generate visualizations (to be implemented)
│   ├── 06_data_quality.py       # Quality assessment (to be implemented)
│   ├── 07_generate_reports.py  # Report generation (to be implemented)
│   └── run_all_eda.py           # Master script to run all analyses
├── output/                       # Generated analysis outputs (JSON, CSV)
├── plots/                        # Generated visualizations
└── reports/                      # Generated markdown reports
```

## Quick Start

### 1. Activate llama-env (Recommended)

This project uses `llama-env` at `/home/himanshu/llama-env`:

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
source /home/himanshu/llama-env/bin/activate
```

Or use the wrapper script:
```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
./run_with_llama_env.sh
```

### 2. Install Dependencies (if needed)

Dependencies should already be installed in llama-env. If not:

```bash
source /home/himanshu/llama-env/bin/activate
pip install -r requirements.txt
```

### 3. Run Individual Analysis

```bash
# Run basic statistics
python scripts/01_basic_statistics.py

# Run content analysis
python scripts/02_content_analysis.py

# Run reference analysis
python scripts/03_reference_analysis.py
```

### 4. Run All Analyses

```bash
# With llama-env activated
source /home/himanshu/llama-env/bin/activate
python scripts/run_all_eda.py

# Or use wrapper script
./run_with_llama_env.sh
```

## Analysis Scripts

### 01_basic_statistics.py
- Dataset overview (total compounds, ID ranges)
- Field presence analysis
- Basic distributions (pages, references, text lengths)
- Naming pattern analysis

**Output:**
- `output/01_basic_statistics.json`
- `output/01_basic_statistics_report.md`

### 02_content_analysis.py
- Text metrics (characters, words, sentences)
- Chemical information extraction (formulas, elements)
- Compound type and state analysis
- Content structure and section presence

**Output:**
- `output/02_content_analysis.json`
- `output/02_content_analysis_report.md`

### 03_reference_analysis.py
- Reference count statistics
- Reference type distribution
- Page reference analysis
- Cross-reference patterns

**Output:**
- `output/03_reference_analysis.json`
- `output/03_reference_analysis_report.md`

## Output Files

All outputs are saved in the `output/` directory:

- **JSON files**: Machine-readable analysis results
- **Markdown reports**: Human-readable analysis reports

## Reports

Comprehensive summary reports are generated in the `reports/` directory:

- `reports/eda_comprehensive_summary.md` - Overall summary of all analyses

## Data Source

The analysis scripts read from:
```
/home/himanshu/MSC_FINAL/dev/test/test/data/processed/individual_compounds/
```

Each JSON file contains:
- `compound_id`: Unique identifier
- `name`: Compound name
- `arabic_start_page`, `arabic_end_page`, `pdf_start_page`: Page information
- `main_entry_content`: Main text content
- `main_entry_length`: Length of main entry
- `comprehensive_text`: Full text including references
- `comprehensive_text_length`: Length of comprehensive text
- `total_references`: Number of references
- `reference_types_found`: Types of references
- `references_breakdown`: Detailed reference information
- `metadata`: Source and extraction metadata

## Next Steps

1. Review the generated reports in `output/` and `reports/`
2. Implement remaining scripts (04-07) as needed
3. Generate visualizations using the analysis data
4. Use findings to inform data processing improvements

## Notes

- All scripts use absolute paths for data directories
- Scripts are designed to be run independently or via the master script
- Output directories are created automatically if they don't exist
- Scripts handle missing data gracefully

## Troubleshooting

If you encounter errors:

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **File not found**: Verify the data directory path is correct
3. **Permission errors**: Check file permissions on output directories
4. **Memory issues**: Process files in batches if dataset is very large

