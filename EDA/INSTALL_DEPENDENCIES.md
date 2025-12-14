# Installing EDA Dependencies

## Quick Install

If you have system-wide access or are using a virtual environment:

```bash
pip install matplotlib seaborn scikit-learn
```

## User Installation (No Admin Access)

If you don't have admin access, install to user directory:

```bash
pip install --user matplotlib seaborn scikit-learn
```

## Virtual Environment (Recommended)

Create and activate a virtual environment:

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

pip install -r requirements.txt
```

## Required Packages

### Core Dependencies
- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computations
- **matplotlib** >= 3.7.0 - Basic plotting
- **seaborn** >= 0.12.0 - Statistical visualizations
- **scikit-learn** >= 1.3.0 - ML, clustering, TF-IDF

### Optional Dependencies
- **rdkit** - Molecular analysis (install via conda: `conda install -c conda-forge rdkit`)
- **pyarrow** - Parquet support

## Verify Installation

Test if packages are available:

```bash
python3 -c "import matplotlib; import seaborn; import sklearn; print('✅ All packages available')"
```

## Troubleshooting

### Import Errors
If you see "Warning: matplotlib/seaborn not available":
1. Check if packages are installed: `pip list | grep matplotlib`
2. Install missing packages: `pip install --user matplotlib seaborn scikit-learn`
3. If using venv, make sure it's activated

### RDKit Installation
RDKit is best installed via conda:
```bash
conda install -c conda-forge rdkit
```

If conda is not available, RDKit features will be limited but the script will still run.

## Running Scripts

After installation, scripts should work without warnings:

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
python scripts/05_cleaning_and_eda.py
```

You should see:
- ✅ Visualizations being created
- ✅ Clustering analysis
- ✅ ML model training

Instead of:
- ⚠️ Visualization not available
- ⚠️ scikit-learn not available

