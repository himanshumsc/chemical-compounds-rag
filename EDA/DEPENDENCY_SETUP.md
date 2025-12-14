# Dependency Setup Guide

## Current Status

The script `05_cleaning_and_eda.py` ran successfully but some optional dependencies are missing:
- ⚠️ matplotlib/seaborn (for visualizations)
- ⚠️ scikit-learn (for ML and clustering)

**The script still works** - it gracefully handles missing dependencies and completes all cleaning operations. However, to get full functionality (visualizations, clustering, ML), you need to install these packages.

## Solutions

### Option 0: Use llama-env (Recommended for this project)

This project uses `llama-env` at `/home/himanshu/llama-env`. Use the wrapper script:

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
./run_with_llama_env.sh
```

Or manually activate and run:
```bash
source /home/himanshu/llama-env/bin/activate
cd /home/himanshu/MSC_FINAL/dev/EDA
pip install matplotlib seaborn scikit-learn  # If not already installed
python scripts/05_cleaning_and_eda.py
```

### Option 1: User Installation (Recommended if no admin access)

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
pip install --user matplotlib seaborn scikit-learn
```

Then verify:
```bash
python3 -c "import matplotlib; import seaborn; import sklearn; print('✅ All packages available')"
```

### Option 2: Virtual Environment (Best Practice)

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run scripts with the venv activated:
```bash
source venv/bin/activate
python scripts/05_cleaning_and_eda.py
```

### Option 3: System Packages (If available)

```bash
sudo apt update
sudo apt install python3-matplotlib python3-seaborn python3-sklearn python3-pandas python3-numpy
```

### Option 4: Quick Install Script

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
bash QUICK_INSTALL.sh
```

### Option 5: Force Install (Use with caution)

```bash
pip install --break-system-packages matplotlib seaborn scikit-learn
```

⚠️ **Warning**: This may affect system Python packages. Use only if you understand the risks.

## What Works Without Dependencies

Even without matplotlib/seaborn/scikit-learn, the script successfully:

✅ **Data Cleaning**
- Temperature parsing (MP/BP to float)
- Decomposes flagging
- One-hot encoding (Elements, Compound Types)
- RDKit feature extraction (if RDKit available)

✅ **Data Export**
- Cleaned CSV file: `05_compounds_cleaned.csv`
- All cleaned data with new features

✅ **Reports**
- Cleaning report: `05_cleaning_eda_report.md`

## What Requires Dependencies

❌ **Visualizations** (requires matplotlib/seaborn)
- Histograms (MW, MP, BP distributions)
- Correlation heatmaps
- Scatter plots
- Boxplots

❌ **Advanced Analysis** (requires scikit-learn)
- TF-IDF feature extraction
- KMeans clustering
- ML hazard prediction model

## Verification

After installation, run the script again:

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
python scripts/05_cleaning_and_eda.py
```

You should now see:
- ✅ Saved molecular weight histogram
- ✅ Saved correlation matrix
- ✅ Clustered X compounds into Y clusters
- ✅ ML model training results

Instead of:
- ⚠️ Visualization not available
- ⚠️ scikit-learn not available

## Current Output Files

Even without full dependencies, you should have:

1. **`05_compounds_cleaned.csv`** - Fully cleaned dataset with:
   - Parsed temperatures (Melting_Point_C, Boiling_Point_C)
   - Decomposes flags
   - One-hot encoded elements
   - One-hot encoded compound types
   - All original fields

2. **`05_cleaning_eda_report.md`** - Summary report

## Next Steps

1. **Install dependencies** using one of the options above
2. **Re-run the script** to generate visualizations and ML models
3. **Check output** in `plots/` directory for generated charts

## Troubleshooting

### "externally-managed-environment" Error
This means your system Python is protected. Use:
- Virtual environment (Option 2) - **Recommended**
- User installation (Option 1)
- System packages (Option 3)

### Packages Installed But Still Not Found
- Check Python path: `python3 -c "import sys; print(sys.path)"`
- Ensure you're using the same Python: `which python3`
- If using venv, make sure it's activated

### RDKit Issues
RDKit is optional and best installed via conda:
```bash
conda install -c conda-forge rdkit
```
The script works without RDKit, but molecular feature extraction will be limited.

