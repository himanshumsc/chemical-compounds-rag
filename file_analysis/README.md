# File Analysis Tools

This directory contains tools for analyzing file modification dates across the project.

## Files

- `check_file_dates.py` - Main script to collect file modification data and generate reports
- `file_modified_data.json` - Collected file metadata (modification dates, sizes, etc.)
- `file_modified_report.md` - Generated Markdown report with analysis and groupings

## Usage

### Collect Data
```bash
python3 check_file_dates.py --collect
```

### Generate Report
```bash
python3 check_file_dates.py --json file_modified_data.json --report file_modified_report.md
```

## Features

- Collects modification dates for `.py`, `.json`, `.sh`, and `.md` files
- Excludes files in hidden folders (starting with `.`)
- Generates comprehensive Markdown reports with:
  - Summary statistics
  - Analysis by date ranges
  - Grouping by directory, extension, and date ranges
  - Most recently modified files
  - Largest files

