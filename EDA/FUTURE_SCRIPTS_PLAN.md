# Future Scripts Plan - EDA for Individual Compounds Dataset

## Overview

This document outlines the plan for additional EDA scripts to further analyze and enrich the structured compound dataset.

---

## Completed Scripts

### ✅ 01_basic_statistics.py
- Dataset overview and basic distributions
- Field presence analysis
- Naming patterns

### ✅ 02_content_analysis.py
- Text metrics (characters, words, sentences)
- Chemical information extraction
- Content structure analysis

### ✅ 03_reference_analysis.py
- Reference count statistics
- Reference type distribution
- Page reference patterns

### ✅ 04_extract_structured_data.py
- Structured data extraction to DataFrame
- RDKit enrichment (SMILES, molecular features)
- Missing value flags
- Export to CSV, JSON, Parquet

---

## Planned Future Scripts

### 05_advanced_rdkit_analysis.py

**Purpose**: Deep molecular analysis using RDKit

**Features**:
- Convert formulas to SMILES (using PubChem API or formula parsers)
- Calculate molecular descriptors:
  - LogP (lipophilicity)
  - TPSA (topological polar surface area)
  - NumRotatableBonds
  - NumAromaticRings
  - NumSaturatedRings
  - NumAliphaticRings
  - NumHeteroatoms
  - NumHeavyAtoms
  - FractionCsp3
  - NumRadicalElectrons
- Generate molecular fingerprints (Morgan, RDKit, MACCS)
- Calculate molecular similarity matrices
- Identify structural patterns and substructures

**Dependencies**:
- rdkit
- pubchempy (for formula to SMILES conversion)
- numpy, pandas

**Outputs**:
- `05_rdkit_descriptors.csv` - All molecular descriptors
- `05_rdkit_analysis_report.md` - Analysis report
- `05_molecular_similarity_matrix.csv` - Similarity matrix

---

### 06_chemical_property_analysis.py

**Purpose**: Analyze extracted chemical properties and relationships

**Features**:
- Melting point analysis:
  - Extract numeric values from strings
  - Temperature unit conversion (Celsius, Fahrenheit, Kelvin)
  - Distribution analysis
  - Correlation with molecular weight
- Boiling point analysis:
  - Extract numeric values
  - Unit conversion
  - Distribution and correlations
- Solubility pattern analysis:
  - Categorize solubility descriptions
  - Extract solvents mentioned
  - Create solubility matrix
- Molecular weight analysis:
  - Distribution statistics
  - Relationship with other properties
  - Outlier detection
- State analysis:
  - Distribution
  - Temperature-dependent state transitions
  - Relationship with molecular properties

**Dependencies**:
- pandas
- numpy
- scipy (for statistical analysis)
- matplotlib, seaborn (for visualizations)

**Outputs**:
- `06_property_analysis.json` - Property statistics
- `06_property_correlations.csv` - Correlation matrix
- `06_property_analysis_report.md` - Analysis report
- `plots/06_property_distributions/` - Distribution plots

---

### 07_text_semantic_analysis.py

**Purpose**: Semantic analysis of text fields (Overview, Uses, Hazards, etc.)

**Features**:
- Text length statistics per field
- Word frequency analysis
- Topic modeling (LDA, NMF) on text fields
- Named Entity Recognition (NER) for:
  - Chemical names
  - Company names
  - Researcher names
  - Dates/years
- Sentiment analysis (if applicable)
- Keyword extraction
- Text similarity analysis between compounds
- Clustering based on text similarity

**Dependencies**:
- pandas
- nltk
- spacy (for NER)
- scikit-learn (for topic modeling, clustering)
- gensim (for topic modeling)

**Outputs**:
- `07_text_analysis.json` - Text statistics
- `07_topic_modeling_results.json` - Topic modeling results
- `07_named_entities.json` - Extracted entities
- `07_text_analysis_report.md` - Analysis report

---

### 08_compound_classification.py

**Purpose**: Classify compounds into categories and analyze patterns

**Features**:
- Compound type distribution analysis
- Hierarchical classification:
  - Organic vs Inorganic
  - By functional groups
  - By application (pharmaceutical, industrial, etc.)
- Pattern recognition:
  - Common structural patterns
  - Element combinations
  - Property clusters
- Classification based on:
  - Compound type
  - State
  - Molecular weight ranges
  - Element composition

**Dependencies**:
- pandas
- numpy
- scikit-learn (for clustering)
- matplotlib, seaborn

**Outputs**:
- `08_compound_classifications.csv` - Classification results
- `08_classification_report.md` - Classification report
- `plots/08_classification_visualizations/` - Classification plots

---

### 09_cross_reference_network.py

**Purpose**: Analyze cross-references and create network graphs

**Features**:
- Build reference network:
  - Nodes: Compounds
  - Edges: Cross-references
- Network metrics:
  - Degree centrality
  - Betweenness centrality
  - Clustering coefficient
  - Community detection
- Identify:
  - Most referenced compounds
  - Reference clusters
  - Isolated compounds
- Visualize network graph

**Dependencies**:
- pandas
- networkx (for network analysis)
- matplotlib, plotly (for visualization)

**Outputs**:
- `09_reference_network.json` - Network data
- `09_network_metrics.csv` - Network metrics
- `09_network_analysis_report.md` - Analysis report
- `plots/09_network_graph.html` - Interactive network visualization

---

### 10_data_quality_assessment.py

**Purpose**: Comprehensive data quality assessment

**Features**:
- Completeness analysis:
  - Missing value patterns
  - Field completeness scores
  - Compound completeness scores
- Consistency checks:
  - Formula vs Elements consistency
  - Molecular weight validation
  - State vs temperature consistency
  - Cross-field validation
- Accuracy checks:
  - Formula format validation
  - Element name validation
  - Temperature format validation
- Anomaly detection:
  - Outlier identification
  - Unusual patterns
  - Data integrity issues
- Quality scoring:
  - Per-compound quality score
  - Per-field quality score
  - Overall dataset quality score

**Dependencies**:
- pandas
- numpy
- scipy

**Outputs**:
- `10_quality_assessment.json` - Quality metrics
- `10_quality_scores.csv` - Quality scores per compound
- `10_anomalies.json` - Detected anomalies
- `10_quality_report.md` - Quality assessment report

---

### 11_visualization_dashboard.py

**Purpose**: Generate comprehensive visualization dashboard

**Features**:
- Distribution plots:
  - Molecular weight distribution
  - Melting/boiling point distributions
  - Reference count distribution
  - Text length distributions
- Relationship plots:
  - Scatter plots (MW vs MP, MW vs BP)
  - Correlation heatmaps
  - Pair plots
- Categorical plots:
  - State distribution (pie/bar)
  - Compound type distribution
  - Element frequency
- Time series (if applicable):
  - Reference patterns over pages
- Interactive dashboards:
  - Plotly dashboards
  - HTML reports with embedded plots

**Dependencies**:
- pandas
- matplotlib
- seaborn
- plotly
- bokeh (optional)

**Outputs**:
- `plots/11_dashboard/` - All visualization files
- `11_interactive_dashboard.html` - Interactive HTML dashboard
- `11_visualization_report.md` - Visualization report

---

### 12_comprehensive_report_generator.py

**Purpose**: Generate comprehensive EDA report combining all analyses

**Features**:
- Aggregate results from all previous scripts
- Create executive summary
- Generate detailed sections:
  - Dataset overview
  - Data quality assessment
  - Chemical property analysis
  - Text analysis findings
  - Network analysis insights
  - Recommendations
- Export to multiple formats:
  - Markdown
  - PDF (using markdown to PDF converter)
  - HTML
- Include visualizations
- Create data dictionary

**Dependencies**:
- pandas
- markdown
- weasyprint or reportlab (for PDF)

**Outputs**:
- `reports/comprehensive_eda_report.md` - Full markdown report
- `reports/comprehensive_eda_report.html` - HTML version
- `reports/comprehensive_eda_report.pdf` - PDF version
- `reports/data_dictionary.md` - Data dictionary

---

## Implementation Priority

### Phase 1 (High Priority)
1. ✅ 04_extract_structured_data.py - **COMPLETED**
2. 05_advanced_rdkit_analysis.py - Molecular descriptors
3. 06_chemical_property_analysis.py - Property relationships
4. 10_data_quality_assessment.py - Quality checks

### Phase 2 (Medium Priority)
5. 07_text_semantic_analysis.py - Text insights
6. 08_compound_classification.py - Classification
7. 11_visualization_dashboard.py - Visualizations

### Phase 3 (Lower Priority)
8. 09_cross_reference_network.py - Network analysis
9. 12_comprehensive_report_generator.py - Final report

---

## Script Template Structure

Each future script should follow this structure:

```python
#!/usr/bin/env python3
"""
[Script Name] - [Purpose]
[Description]
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration
DATA_DIR = Path("/home/himanshu/MSC_FINAL/dev/test/test/data/processed/individual_compounds")
STRUCTURED_DATA = Path("/home/himanshu/MSC_FINAL/dev/EDA/output/04_compounds_structured.csv")
OUTPUT_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/output")

def load_structured_data() -> pd.DataFrame:
    """Load the structured DataFrame."""
    return pd.read_csv(STRUCTURED_DATA)

def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform analysis."""
    # Analysis logic here
    pass

def generate_report(analysis: Dict[str, Any], output_path: Path):
    """Generate markdown report."""
    pass

def main():
    """Main execution."""
    # Load data
    # Analyze
    # Save results
    # Generate report
    pass

if __name__ == "__main__":
    main()
```

---

## Integration with Master Script

Update `run_all_eda.py` to include new scripts:

```python
ANALYSIS_SCRIPTS = [
    "01_basic_statistics.py",
    "02_content_analysis.py",
    "03_reference_analysis.py",
    "04_extract_structured_data.py",  # NEW
    "05_advanced_rdkit_analysis.py",  # FUTURE
    "06_chemical_property_analysis.py",  # FUTURE
    # ... etc
]
```

---

## Dependencies Summary

### Core (Already in requirements.txt)
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0

### Additional (To be added)
- rdkit (conda install -c conda-forge rdkit)
- pubchempy (pip install pubchempy)
- networkx (pip install networkx)
- plotly (pip install plotly)
- nltk (pip install nltk)
- spacy (pip install spacy)
- scikit-learn (pip install scikit-learn)
- gensim (pip install gensim)
- weasyprint or reportlab (for PDF generation)

---

## Notes

1. **RDKit Installation**: RDKit is best installed via conda, not pip
2. **Data Dependencies**: Scripts 05-12 should use the structured DataFrame from script 04
3. **Error Handling**: All scripts should handle missing data gracefully
4. **Progress Reporting**: Long-running scripts should include progress indicators
5. **Modularity**: Scripts should be independent but can build on previous outputs

---

## Success Criteria

Each script should:
- ✅ Load data successfully
- ✅ Handle missing/incomplete data
- ✅ Generate meaningful insights
- ✅ Export results in multiple formats
- ✅ Include comprehensive documentation
- ✅ Generate markdown reports
- ✅ Be executable independently
- ✅ Integrate with master script

---

## Timeline Estimate

- **Phase 1**: 2-3 weeks (4 scripts)
- **Phase 2**: 2-3 weeks (3 scripts)
- **Phase 3**: 1-2 weeks (2 scripts)

**Total**: ~6-8 weeks for complete implementation

---

## Next Steps

1. ✅ Complete script 04 (extract structured data)
2. Implement script 05 (RDKit analysis)
3. Implement script 06 (property analysis)
4. Implement script 10 (quality assessment)
5. Continue with remaining scripts based on priority

