# Using EDA Scripts with llama-env

## Overview

This project uses the `llama-env` virtual environment located at `/home/himanshu/llama-env`. All EDA scripts should be run with this environment activated.

## Quick Start

### Option 1: Use Wrapper Script (Easiest)

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
./run_with_llama_env.sh
```

This will:
- ✅ Activate llama-env automatically
- ✅ Check for missing dependencies
- ✅ Offer to install them if needed
- ✅ Run all EDA scripts

To run a specific script:
```bash
./run_with_llama_env.sh 05_cleaning_and_eda.py
```

### Option 2: Manual Activation

```bash
# Activate llama-env
source /home/himanshu/llama-env/bin/activate

# Navigate to EDA directory
cd /home/himanshu/MSC_FINAL/dev/EDA

# Run scripts
python scripts/05_cleaning_and_eda.py
# or
python scripts/run_all_eda.py
```

## Dependencies

Dependencies are now installed in llama-env:
- ✅ matplotlib
- ✅ seaborn
- ✅ scikit-learn
- ✅ pandas
- ✅ numpy

## Verification

Check if llama-env has all dependencies:

```bash
source /home/himanshu/llama-env/bin/activate
python -c "import matplotlib; import seaborn; import sklearn; print('✅ All packages available')"
```

## Running Scripts

### Run All EDA Scripts

```bash
source /home/himanshu/llama-env/bin/activate
cd /home/himanshu/MSC_FINAL/dev/EDA
python scripts/run_all_eda.py
```

### Run Individual Scripts

```bash
source /home/himanshu/llama-env/bin/activate
cd /home/himanshu/MSC_FINAL/dev/EDA

# Basic statistics
python scripts/01_basic_statistics.py

# Content analysis
python scripts/02_content_analysis.py

# Reference analysis
python scripts/03_reference_analysis.py

# Structured data extraction
python scripts/04_extract_structured_data.py

# Cleaning and EDA (with visualizations and ML)
python scripts/05_cleaning_and_eda.py
```

## Expected Output

With llama-env activated, you should see:

✅ **Full functionality:**
- Visualizations (histograms, scatter plots, boxplots)
- Clustering analysis
- ML model training
- TF-IDF feature extraction

Instead of:
- ⚠️ Visualization not available
- ⚠️ scikit-learn not available

## Troubleshooting

### "llama-env not found"
If llama-env doesn't exist, create it:
```bash
python3 -m venv /home/himanshu/llama-env
source /home/himanshu/llama-env/bin/activate
pip install -r requirements.txt
```

### "Module not found" errors
Install missing packages:
```bash
source /home/himanshu/llama-env/bin/activate
pip install matplotlib seaborn scikit-learn pandas numpy
```

### Wrong Python environment
Make sure llama-env is activated:
```bash
which python  # Should show /home/himanshu/llama-env/bin/python
```

## Notes

- Always activate llama-env before running EDA scripts
- The wrapper script (`run_with_llama_env.sh`) handles activation automatically
- All dependencies are installed in llama-env, not system Python

