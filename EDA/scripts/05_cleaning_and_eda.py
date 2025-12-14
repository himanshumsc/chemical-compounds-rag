#!/usr/bin/env python3
"""
Comprehensive Cleaning and EDA for Individual Compounds Dataset
Includes data cleaning, univariate/bivariate analysis, clustering, and ML.
"""

import json
import re
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    VIS_AVAILABLE = True
except ImportError as e:
    print("Warning: matplotlib/seaborn not available. Visualizations will be skipped.")
    print(f"  Install with: pip install --user matplotlib seaborn")
    print(f"  Or use: pip install --break-system-packages matplotlib seaborn")
    VIS_AVAILABLE = False

# ML and NLP
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("Warning: scikit-learn not available. ML features will be limited.")
    print(f"  Install with: pip install --user scikit-learn")
    print(f"  Or use: pip install --break-system-packages scikit-learn")
    SKLEARN_AVAILABLE = False

# RDKit for cheminformatics
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Cheminformatics features will be limited.")
    RDKIT_AVAILABLE = False

# Configuration
DATA_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/output")
STRUCTURED_DATA = DATA_DIR / "04_compounds_structured.json"
OUTPUT_DIR = DATA_DIR
PLOTS_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_structured_data() -> pd.DataFrame:
    """Load the structured DataFrame from JSON."""
    print("📁 Loading structured data...")
    
    with open(STRUCTURED_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} compounds with {len(df.columns)} columns")
    return df


def parse_temperature(temp_str: Optional[str]) -> Tuple[Optional[float], bool]:
    """
    Parse temperature string to float in Celsius.
    Returns: (temperature_value, decomposes_flag)
    """
    if pd.isna(temp_str) or not temp_str:
        return None, False
    
    temp_str = str(temp_str).strip()
    
    # Check for decomposes/not applicable
    if re.search(r'not applicable|n/a|decomposes', temp_str, re.IGNORECASE):
        return None, True
    
    # Extract numeric value (first number found)
    match = re.search(r'([0-9.]+)', temp_str)
    if not match:
        return None, False
    
    value = float(match.group(1))
    
    # Check if Fahrenheit (F) and convert to Celsius
    if 'F' in temp_str.upper() and 'C' not in temp_str.upper():
        value = (value - 32) * 5/9
    
    return value, False


def clean_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and parse temperature columns (Melting_Point, Boiling_Point)."""
    print("🔧 Cleaning temperature columns...")
    
    df_clean = df.copy()
    
    # Parse Melting Point
    mp_results = df_clean['Melting_Point'].apply(parse_temperature)
    df_clean['Melting_Point_C'] = mp_results.apply(lambda x: x[0])
    df_clean['Melting_Point_Decomposes'] = mp_results.apply(lambda x: x[1])
    
    # Parse Boiling Point
    bp_results = df_clean['Boiling_Point'].apply(parse_temperature)
    df_clean['Boiling_Point_C'] = bp_results.apply(lambda x: x[0])
    df_clean['Boiling_Point_Decomposes'] = bp_results.apply(lambda x: x[1])
    
    print(f"  ✅ Parsed {df_clean['Melting_Point_C'].notna().sum()} melting points")
    print(f"  ✅ Parsed {df_clean['Boiling_Point_C'].notna().sum()} boiling points")
    print(f"  ✅ {df_clean['Melting_Point_Decomposes'].sum()} compounds decompose at MP")
    print(f"  ✅ {df_clean['Boiling_Point_Decomposes'].sum()} compounds decompose at BP")
    
    return df_clean


def one_hot_encode_elements(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Elements column."""
    print("🔧 One-hot encoding Elements...")
    
    df_clean = df.copy()
    
    # Get all unique elements (normalize case)
    all_elements = set()
    for elem_list in df_clean['Elements']:
        if isinstance(elem_list, list):
            all_elements.update([e.strip().title() for e in elem_list])
    
    # Create one-hot columns
    for element in sorted(all_elements):
        col_name = f'Element_{element.replace(" ", "_")}'
        df_clean[col_name] = df_clean['Elements'].apply(
            lambda x: 1 if isinstance(x, list) and element in [e.strip().title() for e in x] else 0
        )
    
    print(f"  ✅ Created {len(all_elements)} element one-hot columns")
    return df_clean


def one_hot_encode_compound_types(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Compound_Type column."""
    print("🔧 One-hot encoding Compound Types...")
    
    df_clean = df.copy()
    
    # Get unique compound types
    unique_types = df_clean['Compound_Type'].dropna().unique()
    
    # Create one-hot columns
    for comp_type in unique_types:
        if pd.notna(comp_type):
            # Clean type name for column
            col_name = f'Type_{re.sub(r'[^a-zA-Z0-9]', '_', str(comp_type))}'
            df_clean[col_name] = (df_clean['Compound_Type'] == comp_type).astype(int)
    
    print(f"  ✅ Created {len(unique_types)} compound type one-hot columns")
    return df_clean


def extract_rdkit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract molecular features using RDKit."""
    if not RDKIT_AVAILABLE:
        print("⚠️  RDKit not available. Skipping molecular feature extraction.")
        return df
    
    print("🔧 Extracting RDKit molecular features...")
    
    df_clean = df.copy()
    
    # Initialize feature columns
    rdkit_features = []
    
    for idx, row in df_clean.iterrows():
        formula = row.get('Formula')
        features = {}
        
        if formula:
            try:
                # Try to create molecule from formula (limited - would need proper parser)
                # For now, we'll try to get basic info if SMILES exists
                smiles = row.get('SMILES')
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        features['RDKit_NumAtoms'] = mol.GetNumAtoms()
                        features['RDKit_NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
                        features['RDKit_MolWt'] = Descriptors.MolWt(mol)
                        features['RDKit_LogP'] = Descriptors.MolLogP(mol)
                        features['RDKit_TPSA'] = Descriptors.TPSA(mol)
                        features['RDKit_NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
                        features['RDKit_NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
                        features['RDKit_NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
                        features['RDKit_NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
            except Exception:
                pass
        
        # Fill with None if features not extracted
        for key in ['RDKit_NumAtoms', 'RDKit_NumHeavyAtoms', 'RDKit_MolWt', 'RDKit_LogP',
                   'RDKit_TPSA', 'RDKit_NumRotatableBonds', 'RDKit_NumAromaticRings',
                   'RDKit_NumSaturatedRings', 'RDKit_NumHeteroatoms']:
            if key not in features:
                features[key] = None
        
        rdkit_features.append(features)
    
    # Add RDKit features to dataframe
    rdkit_df = pd.DataFrame(rdkit_features)
    df_clean = pd.concat([df_clean, rdkit_df], axis=1)
    
    extracted_count = rdkit_df.notna().any(axis=1).sum()
    print(f"  ✅ Extracted RDKit features for {extracted_count} compounds")
    
    return df_clean


def create_tfidf_features(df: pd.DataFrame, text_column: str, max_features: int = 100) -> pd.DataFrame:
    """Create TF-IDF features from text column."""
    if not SKLEARN_AVAILABLE:
        print(f"⚠️  scikit-learn not available. Skipping TF-IDF for {text_column}.")
        return df
    
    print(f"🔧 Creating TF-IDF features for {text_column}...")
    
    df_clean = df.copy()
    
    # Fill NaN with empty string
    texts = df_clean[text_column].fillna('').astype(str)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'TFIDF_{text_column}_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        df_clean = pd.concat([df_clean, tfidf_df], axis=1)
        print(f"  ✅ Created {tfidf_matrix.shape[1]} TF-IDF features")
    except Exception as e:
        print(f"  ⚠️  Error creating TF-IDF: {e}")
    
    return df_clean


def univariate_analysis(df: pd.DataFrame):
    """Perform univariate analysis and create histograms."""
    if not VIS_AVAILABLE:
        print("⚠️  Visualization not available. Skipping univariate analysis plots.")
        return
    
    print("📊 Performing univariate analysis...")
    
    # Create plots directory
    univariate_dir = PLOTS_DIR / "05_univariate"
    univariate_dir.mkdir(parents=True, exist_ok=True)
    
    # Molecular Weight histogram
    if 'Molecular_Weight' in df.columns:
        plt.figure(figsize=(10, 6))
        mw_data = df['Molecular_Weight'].dropna()
        plt.hist(mw_data, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Molecular Weight (g/mol)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Molecular Weight')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(univariate_dir / 'molecular_weight_distribution.png', dpi=300)
        plt.close()
        print(f"  ✅ Saved molecular weight histogram")
    
    # Melting Point histogram
    if 'Melting_Point_C' in df.columns:
        plt.figure(figsize=(10, 6))
        mp_data = df['Melting_Point_C'].dropna()
        if len(mp_data) > 0:
            plt.hist(mp_data, bins=30, edgecolor='black', alpha=0.7, color='orange')
            plt.xlabel('Melting Point (°C)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Melting Point')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(univariate_dir / 'melting_point_distribution.png', dpi=300)
            plt.close()
            print(f"  ✅ Saved melting point histogram")
    
    # Boiling Point histogram
    if 'Boiling_Point_C' in df.columns:
        plt.figure(figsize=(10, 6))
        bp_data = df['Boiling_Point_C'].dropna()
        if len(bp_data) > 0:
            plt.hist(bp_data, bins=30, edgecolor='black', alpha=0.7, color='green')
            plt.xlabel('Boiling Point (°C)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Boiling Point')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(univariate_dir / 'boiling_point_distribution.png', dpi=300)
            plt.close()
            print(f"  ✅ Saved boiling point histogram")


def bivariate_analysis(df: pd.DataFrame):
    """Perform bivariate analysis (correlations, boxplots)."""
    if not VIS_AVAILABLE:
        print("⚠️  Visualization not available. Skipping bivariate analysis plots.")
        return
    
    print("📊 Performing bivariate analysis...")
    
    bivariate_dir = PLOTS_DIR / "05_bivariate"
    bivariate_dir.mkdir(parents=True, exist_ok=True)
    
    # Correlation matrix
    numeric_cols = ['Molecular_Weight', 'Melting_Point_C', 'Boiling_Point_C']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(numeric_cols) > 1:
        corr_data = df[numeric_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix: Molecular Weight, Melting Point, Boiling Point')
        plt.tight_layout()
        plt.savefig(bivariate_dir / 'correlation_matrix.png', dpi=300)
        plt.close()
        print(f"  ✅ Saved correlation matrix")
    
    # Scatter plots
    if 'Molecular_Weight' in df.columns and 'Melting_Point_C' in df.columns:
        plt.figure(figsize=(10, 6))
        scatter_data = df[['Molecular_Weight', 'Melting_Point_C']].dropna()
        if len(scatter_data) > 0:
            plt.scatter(scatter_data['Molecular_Weight'], scatter_data['Melting_Point_C'],
                       alpha=0.6, s=50)
            plt.xlabel('Molecular Weight (g/mol)')
            plt.ylabel('Melting Point (°C)')
            plt.title('Molecular Weight vs Melting Point')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(bivariate_dir / 'mw_vs_mp_scatter.png', dpi=300)
            plt.close()
            print(f"  ✅ Saved MW vs MP scatter plot")
    
    if 'Molecular_Weight' in df.columns and 'Boiling_Point_C' in df.columns:
        plt.figure(figsize=(10, 6))
        scatter_data = df[['Molecular_Weight', 'Boiling_Point_C']].dropna()
        if len(scatter_data) > 0:
            plt.scatter(scatter_data['Molecular_Weight'], scatter_data['Boiling_Point_C'],
                       alpha=0.6, s=50, color='green')
            plt.xlabel('Molecular Weight (g/mol)')
            plt.ylabel('Boiling Point (°C)')
            plt.title('Molecular Weight vs Boiling Point')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(bivariate_dir / 'mw_vs_bp_scatter.png', dpi=300)
            plt.close()
            print(f"  ✅ Saved MW vs BP scatter plot")
    
    # Boxplots by Compound Type
    if 'Compound_Type' in df.columns and 'Molecular_Weight' in df.columns:
        # Get top compound types
        top_types = df['Compound_Type'].value_counts().head(10).index.tolist()
        boxplot_data = df[df['Compound_Type'].isin(top_types)]
        
        if len(boxplot_data) > 0:
            plt.figure(figsize=(14, 8))
            boxplot_data.boxplot(column='Molecular_Weight', by='Compound_Type',
                               ax=plt.gca(), rot=45, fontsize=8)
            plt.title('Molecular Weight Distribution by Compound Type')
            plt.suptitle('')  # Remove default title
            plt.xlabel('Compound Type')
            plt.ylabel('Molecular Weight (g/mol)')
            plt.tight_layout()
            plt.savefig(bivariate_dir / 'molecular_weight_by_type_boxplot.png', dpi=300)
            plt.close()
            print(f"  ✅ Saved molecular weight boxplot by type")


def cluster_compounds(df: pd.DataFrame):
    """Cluster compounds by molecular features."""
    if not SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn not available. Skipping clustering.")
        return None
    
    print("🔍 Clustering compounds by molecular features...")
    
    # Select features for clustering
    feature_cols = ['Molecular_Weight', 'Melting_Point_C', 'Boiling_Point_C']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Add RDKit features if available
    rdkit_cols = [col for col in df.columns if col.startswith('RDKit_')]
    feature_cols.extend(rdkit_cols)
    
    # Prepare data
    cluster_data = df[feature_cols].dropna()
    
    if len(cluster_data) < 5:
        print("  ⚠️  Not enough data for clustering")
        return None
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # KMeans clustering
    n_clusters = min(5, len(cluster_data) // 3)  # Adaptive number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to dataframe
    cluster_indices = cluster_data.index
    df_clustered = df.copy()
    df_clustered.loc[cluster_indices, 'Cluster'] = cluster_labels
    
    print(f"  ✅ Clustered {len(cluster_data)} compounds into {n_clusters} clusters")
    
    # Visualize clusters if possible
    if VIS_AVAILABLE and len(feature_cols) >= 2:
        try:
            # PCA for 2D visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels,
                                cmap='viridis', alpha=0.6, s=50)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('Compound Clusters (PCA Visualization)')
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            cluster_dir = PLOTS_DIR / "05_clustering"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(cluster_dir / 'compound_clusters_pca.png', dpi=300)
            plt.close()
            print(f"  ✅ Saved cluster visualization")
        except Exception as e:
            print(f"  ⚠️  Could not create cluster visualization: {e}")
    
    return df_clustered


def predict_hazards_ml(df: pd.DataFrame):
    """Predict hazards using ML models."""
    if not SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn not available. Skipping hazard prediction.")
        return None
    
    print("🤖 Training ML model for hazard prediction...")
    
    # Create binary target: has hazards or not
    df_ml = df.copy()
    df_ml['Has_Hazards'] = df_ml['Hazards'].notna().astype(int)
    
    # Select features
    feature_cols = ['Molecular_Weight', 'Melting_Point_C', 'Boiling_Point_C']
    feature_cols = [col for col in feature_cols if col in df_ml.columns]
    
    # Add one-hot encoded elements (top elements)
    element_cols = [col for col in df_ml.columns if col.startswith('Element_')]
    feature_cols.extend(element_cols[:20])  # Limit to top 20 elements
    
    # Add compound type one-hot
    type_cols = [col for col in df_ml.columns if col.startswith('Type_')]
    feature_cols.extend(type_cols[:10])  # Limit to top 10 types
    
    # Prepare data
    ml_data = df_ml[feature_cols + ['Has_Hazards']].dropna()
    
    if len(ml_data) < 20:
        print("  ⚠️  Not enough data for ML model")
        return None
    
    X = ml_data[feature_cols]
    y = ml_data['Has_Hazards']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Evaluation
    print("\n  📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Hazards', 'Has Hazards']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n  📊 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save results
    results = {
        'model_type': 'RandomForest',
        'n_samples': len(ml_data),
        'n_features': len(feature_cols),
        'test_accuracy': (y_pred == y_test).mean(),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    results_path = OUTPUT_DIR / "05_ml_hazard_prediction_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  ✅ Saved ML results to {results_path}")
    
    return results


def generate_cleaning_report(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, output_path: Path):
    """Generate a comprehensive cleaning and EDA report."""
    report = []
    report.append("# Cleaning and EDA Report\n\n")
    report.append("---\n\n")
    
    # Data Overview
    report.append("## Data Overview\n\n")
    report.append(f"- **Original Compounds:** {len(df_original)}\n")
    report.append(f"- **Cleaned Compounds:** {len(df_cleaned)}\n")
    report.append(f"- **Original Columns:** {len(df_original.columns)}\n")
    report.append(f"- **Cleaned Columns:** {len(df_cleaned.columns)}\n")
    report.append("\n")
    
    # Cleaning Summary
    report.append("## Cleaning Summary\n\n")
    
    # Temperature parsing
    if 'Melting_Point_C' in df_cleaned.columns:
        mp_parsed = df_cleaned['Melting_Point_C'].notna().sum()
        mp_decomp = df_cleaned['Melting_Point_Decomposes'].sum()
        report.append(f"- **Melting Points Parsed:** {mp_parsed}\n")
        report.append(f"- **Compounds that Decompose at MP:** {mp_decomp}\n")
    
    if 'Boiling_Point_C' in df_cleaned.columns:
        bp_parsed = df_cleaned['Boiling_Point_C'].notna().sum()
        bp_decomp = df_cleaned['Boiling_Point_Decomposes'].sum()
        report.append(f"- **Boiling Points Parsed:** {bp_parsed}\n")
        report.append(f"- **Compounds that Decompose at BP:** {bp_decomp}\n")
    
    # One-hot encoding
    element_cols = [col for col in df_cleaned.columns if col.startswith('Element_')]
    type_cols = [col for col in df_cleaned.columns if col.startswith('Type_')]
    report.append(f"- **Element One-Hot Columns:** {len(element_cols)}\n")
    report.append(f"- **Compound Type One-Hot Columns:** {len(type_cols)}\n")
    
    # RDKit features
    rdkit_cols = [col for col in df_cleaned.columns if col.startswith('RDKit_')]
    if rdkit_cols:
        rdkit_count = df_cleaned[rdkit_cols[0]].notna().sum()
        report.append(f"- **RDKit Features Extracted:** {len(rdkit_cols)} features for {rdkit_count} compounds\n")
    
    report.append("\n")
    
    # Statistical Summary
    report.append("## Statistical Summary\n\n")
    numeric_cols = ['Molecular_Weight', 'Melting_Point_C', 'Boiling_Point_C']
    numeric_cols = [col for col in numeric_cols if col in df_cleaned.columns]
    
    if numeric_cols:
        report.append("| Column | Mean | Median | Std | Min | Max |\n")
        report.append("|--------|------|--------|-----|-----|-----|\n")
        for col in numeric_cols:
            data = df_cleaned[col].dropna()
            if len(data) > 0:
                report.append(f"| {col} | {data.mean():.2f} | {data.median():.2f} | "
                            f"{data.std():.2f} | {data.min():.2f} | {data.max():.2f} |\n")
    
    report.append("\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ Cleaning report saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Comprehensive Cleaning and EDA - Individual Compounds Dataset")
    print("=" * 60)
    print()
    
    # Load data
    df_original = load_structured_data()
    
    # Cleaning
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60 + "\n")
    
    df_cleaned = clean_temperature_columns(df_original)
    df_cleaned = one_hot_encode_elements(df_cleaned)
    df_cleaned = one_hot_encode_compound_types(df_cleaned)
    df_cleaned = extract_rdkit_features(df_cleaned)
    
    # NLP Features
    print("\n" + "=" * 60)
    print("NLP FEATURE EXTRACTION")
    print("=" * 60 + "\n")
    
    df_cleaned = create_tfidf_features(df_cleaned, 'Common_Uses', max_features=50)
    df_cleaned = create_tfidf_features(df_cleaned, 'Hazards', max_features=50)
    
    # Save cleaned data
    print("\n💾 Saving cleaned data...")
    cleaned_csv_path = OUTPUT_DIR / "05_compounds_cleaned.csv"
    df_cleaned.to_csv(cleaned_csv_path, index=False, encoding='utf-8')
    print(f"✅ Saved cleaned data to: {cleaned_csv_path}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("UNIVARIATE & BIVARIATE ANALYSIS")
    print("=" * 60 + "\n")
    
    univariate_analysis(df_cleaned)
    bivariate_analysis(df_cleaned)
    
    # Advanced Analysis
    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS")
    print("=" * 60 + "\n")
    
    df_clustered = cluster_compounds(df_cleaned)
    if df_clustered is not None:
        cluster_csv_path = OUTPUT_DIR / "05_compounds_clustered.csv"
        df_clustered.to_csv(cluster_csv_path, index=False, encoding='utf-8')
        print(f"✅ Saved clustered data to: {cluster_csv_path}")
    
    ml_results = predict_hazards_ml(df_cleaned)
    
    # Generate report
    report_path = OUTPUT_DIR / "05_cleaning_eda_report.md"
    generate_cleaning_report(df_original, df_cleaned, report_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Cleaning completed")
    print(f"✅ Analysis completed")
    print(f"✅ Visualizations saved to: {PLOTS_DIR}")
    print(f"✅ Reports saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

