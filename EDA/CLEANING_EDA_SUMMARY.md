# Cleaning and EDA Summary

## Overview

Script `05_cleaning_and_eda.py` provides comprehensive data cleaning, exploratory data analysis, clustering, and machine learning capabilities for the individual compounds dataset.

## Features Implemented

### ✅ Data Cleaning

1. **Temperature Parsing**
   - Parses Melting Point and Boiling Point strings to float values (Celsius)
   - Handles unit conversion (Fahrenheit to Celsius)
   - Flags compounds that decompose ("decomposes", "not applicable")
   - Creates separate columns: `Melting_Point_C`, `Boiling_Point_C`
   - Creates flag columns: `Melting_Point_Decomposes`, `Boiling_Point_Decomposes`

2. **One-Hot Encoding**
   - **Elements**: Creates binary columns for each unique element (e.g., `Element_Carbon`, `Element_Oxygen`)
   - **Compound Types**: Creates binary columns for each compound type (e.g., `Type_Carboxylic_acid`)

3. **RDKit Feature Extraction**
   - Extracts molecular descriptors when SMILES available:
     - `RDKit_NumAtoms` - Total number of atoms
     - `RDKit_NumHeavyAtoms` - Heavy atoms count
     - `RDKit_MolWt` - Molecular weight
     - `RDKit_LogP` - Lipophilicity
     - `RDKit_TPSA` - Topological polar surface area
     - `RDKit_NumRotatableBonds` - Rotatable bonds
     - `RDKit_NumAromaticRings` - Aromatic rings
     - `RDKit_NumSaturatedRings` - Saturated rings
     - `RDKit_NumHeteroatoms` - Heteroatoms count

### ✅ NLP Feature Extraction

1. **TF-IDF Vectorization**
   - Creates TF-IDF features from text fields:
     - `Common_Uses` - Up to 50 features
     - `Hazards` - Up to 50 features
   - Uses n-grams (1-2) and removes stop words
   - Useful for clustering and similarity analysis

### ✅ Univariate Analysis

1. **Histograms**
   - Molecular Weight distribution
   - Melting Point distribution
   - Boiling Point distribution
   - Saved to `plots/05_univariate/`

### ✅ Bivariate Analysis

1. **Correlation Analysis**
   - Correlation matrix heatmap for MW, MP, BP
   - Saved to `plots/05_bivariate/correlation_matrix.png`

2. **Scatter Plots**
   - Molecular Weight vs Melting Point
   - Molecular Weight vs Boiling Point
   - Saved to `plots/05_bivariate/`

3. **Boxplots**
   - Molecular Weight distribution by Compound Type
   - Saved to `plots/05_bivariate/molecular_weight_by_type_boxplot.png`

### ✅ Advanced Analysis

1. **Clustering**
   - KMeans clustering based on molecular features
   - Uses molecular weight, temperatures, and RDKit features
   - Standardizes features before clustering
   - PCA visualization of clusters (2D)
   - Adds `Cluster` column to dataset
   - Saved to `plots/05_clustering/compound_clusters_pca.png`

2. **Machine Learning - Hazard Prediction**
   - Random Forest classifier to predict presence of hazards
   - Features include:
     - Molecular properties (MW, MP, BP)
     - Element one-hot encodings (top 20)
     - Compound type one-hot encodings (top 10)
   - Generates:
     - Classification report
     - Feature importance ranking
     - Model performance metrics
   - Results saved to `05_ml_hazard_prediction_results.json`

## Output Files

### Data Files
- `05_compounds_cleaned.csv` - Cleaned dataset with all new features
- `05_compounds_clustered.csv` - Dataset with cluster labels
- `05_ml_hazard_prediction_results.json` - ML model results

### Visualizations
- `plots/05_univariate/` - Histograms
- `plots/05_bivariate/` - Correlation, scatter plots, boxplots
- `plots/05_clustering/` - Cluster visualizations

### Reports
- `05_cleaning_eda_report.md` - Comprehensive cleaning and analysis report

## Usage

```bash
cd /home/himanshu/MSC_FINAL/dev/EDA
python scripts/05_cleaning_and_eda.py
```

## Dependencies

**Required:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0 (for ML, clustering, TF-IDF)
- matplotlib >= 3.7.0 (for visualizations)
- seaborn >= 0.12.0 (for statistical plots)

**Optional:**
- rdkit (for molecular feature extraction)
  - Install: `conda install -c conda-forge rdkit`
  - Script works without RDKit but molecular features will be limited

## Data Flow

1. **Load** structured data from `04_compounds_structured.json`
2. **Clean** temperature columns (parse to float, flag decomposes)
3. **Encode** elements and compound types (one-hot)
4. **Extract** RDKit molecular features (if available)
5. **Create** TF-IDF features from text fields
6. **Analyze** univariate distributions (histograms)
7. **Analyze** bivariate relationships (correlations, scatter plots, boxplots)
8. **Cluster** compounds by molecular features
9. **Train** ML model for hazard prediction
10. **Save** cleaned data, visualizations, and reports

## Key Features

### Temperature Parsing
- Handles formats like "16.6°C (61.9°F)" or "117.9C (244.2F)"
- Automatically converts Fahrenheit to Celsius
- Flags "decomposes" and "not applicable" cases

### One-Hot Encoding
- Normalizes element names (title case)
- Creates binary features for presence/absence
- Handles missing values gracefully

### Clustering
- Adaptive number of clusters based on data size
- Standardizes features before clustering
- Provides PCA visualization for 2D representation

### ML Model
- Uses Random Forest for interpretability
- Feature importance analysis
- Stratified train-test split
- Classification report with precision/recall

## Notes

- **RDKit**: Most formulas cannot be directly converted to SMILES without proper parsing. RDKit features are only available if SMILES already exists in the data.
- **Missing Data**: All analyses handle missing data gracefully
- **Visualizations**: Uses non-interactive backend (saves to files)
- **TF-IDF**: Limited to 50 features per text field to avoid excessive dimensionality

## Integration

This script is integrated into the master EDA runner:
```bash
python scripts/run_all_eda.py
```

It runs after data extraction (script 04) and before advanced analysis scripts.

## Future Enhancements

1. **Better Formula Parsing**: Implement proper chemical formula parser for RDKit conversion
2. **More ML Models**: Add other classifiers (SVM, XGBoost, etc.)
3. **Deep Learning**: Use neural networks for text classification
4. **More Clustering Methods**: DBSCAN, hierarchical clustering
5. **Property Prediction**: Predict melting/boiling points from molecular features
6. **Interactive Dashboards**: Create Plotly/Bokeh dashboards

