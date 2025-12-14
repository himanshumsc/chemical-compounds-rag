#!/usr/bin/env python3
"""
Structured Data Extraction and Enrichment for Individual Compounds Dataset
Extracts structured fields from JSON files and creates a DataFrame with RDKit enrichment.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas and numpy are required. Install with: pip install pandas numpy")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")
    print("Continuing without RDKit enrichment...")
    RDKIT_AVAILABLE = False

# Configuration
DATA_DIR = Path("/home/himanshu/MSC_FINAL/dev/test/test/data/processed/individual_compounds")
OUTPUT_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might cause issues
    text = text.replace('\u0002', '')  # Remove special character
    return text


def extract_other_names(text: str) -> List[str]:
    """Extract other names from 'OTHER NAMES:' section."""
    pattern = r'OTHER NAMES:\s*([^\n]+(?:\n(?!FORMULA:)[^\n]+)*)'
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if match:
        names_text = match.group(1)
        # Split by semicolons, commas, or newlines
        names = re.split(r'[;\n,]+', names_text)
        names = [clean_text(name) for name in names if clean_text(name)]
        return names
    return []


def extract_formula(text: str) -> Optional[str]:
    """Extract chemical formula from 'FORMULA:' section."""
    pattern = r'FORMULA:\s*([^\n]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        formula = clean_text(match.group(1))
        # Remove common artifacts
        formula = re.sub(r'\s+', '', formula)
        return formula if formula else None
    return None


def extract_elements(text: str) -> List[str]:
    """Extract elements from 'ELEMENTS:' section."""
    pattern = r'ELEMENTS:\s*([^\n]+(?:\n(?!COMPOUND)[^\n]+)*)'
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if match:
        elements_text = match.group(1)
        # Split by commas
        elements = [clean_text(elem) for elem in re.split(r'[,;]', elements_text)]
        elements = [e for e in elements if e and e.lower() not in ['and', 'or']]
        return elements
    return []


def extract_compound_type(text: str) -> Optional[str]:
    """Extract compound type from 'COMPOUND TYPE:' section."""
    pattern = r'COMPOUND TYPE:\s*([^\n]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return clean_text(match.group(1))
    return None


def extract_state(text: str) -> Optional[str]:
    """Extract state from 'STATE:' section."""
    pattern = r'STATE:\s*([^\n]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return clean_text(match.group(1))
    return None


def extract_molecular_weight(text: str) -> Optional[float]:
    """Extract molecular weight from 'MOLECULAR WEIGHT:' section."""
    pattern = r'MOLECULAR WEIGHT:\s*([0-9.]+)\s*g/mol'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_melting_point(text: str) -> Optional[str]:
    """Extract melting point from 'MELTING POINT:' section."""
    pattern = r'MELTING POINT:\s*([^\n]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        mp = clean_text(match.group(1))
        # Check if it's "Not applicable" or similar
        if re.search(r'not applicable|n/a|decomposes', mp, re.IGNORECASE):
            return None
        return mp
    return None


def extract_boiling_point(text: str) -> Optional[str]:
    """Extract boiling point from 'BOILING POINT:' section."""
    pattern = r'BOILING POINT:\s*([^\n]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        bp = clean_text(match.group(1))
        # Check if it's "Not applicable" or similar
        if re.search(r'not applicable|n/a|decomposes', bp, re.IGNORECASE):
            return None
        return bp
    return None


def extract_solubility(text: str) -> Optional[str]:
    """Extract solubility from 'SOLUBILITY:' section."""
    pattern = r'SOLUBILITY:\s*([^\n]+(?:\n(?!OVERVIEW|HOW|COMMON|Interesting)[^\n]+)*)'
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if match:
        return clean_text(match.group(1))
    return None


def extract_section(text: str, section_name: str) -> Optional[str]:
    """Extract a section by name (e.g., OVERVIEW, HOW IT IS MADE)."""
    # Pattern to match section header and content until next section
    patterns = {
        'OVERVIEW': r'OVERVIEW\s*\n(.*?)(?=\n(?:HOW IT IS MADE|COMMON USES|K E Y F A C T S|$))',
        'HOW IT IS MADE': r'HOW IT IS MADE\s*\n(.*?)(?=\n(?:COMMON USES|POTENTIAL HAZARDS|Interesting Facts|$))',
        'COMMON USES': r'COMMON USES(?: AND POTENTIAL HAZARDS)?\s*\n(.*?)(?=\n(?:POTENTIAL HAZARDS|Interesting Facts|$))',
        'HAZARDS': r'POTENTIAL HAZARDS\s*\n(.*?)(?=\n(?:Interesting Facts|COMMON USES|$))',
        'INTERESTING FACTS': r'Interesting Facts\s*\n(.*?)(?=\n(?:COMMON USES|POTENTIAL HAZARDS|$))'
    }
    
    pattern = patterns.get(section_name.upper())
    if pattern:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Clean up the content
            content = re.sub(r'\n+', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            return content if content else None
    return None


def formula_to_smiles(formula: str) -> Optional[str]:
    """Attempt to convert chemical formula to SMILES using RDKit."""
    if not RDKIT_AVAILABLE or not formula:
        return None
    
    try:
        # Try to parse as SMILES directly (some formulas might already be SMILES-like)
        mol = Chem.MolFromSmiles(formula)
        if mol:
            return formula
        
        # Try common formula patterns
        # This is a simplified approach - full conversion would need a formula parser
        # For now, return None if direct parsing fails
        return None
    except Exception:
        return None


def enrich_with_rdkit(formula: str) -> Dict[str, Any]:
    """Enrich compound data with RDKit features."""
    enrichment = {
        'num_atoms': None,
        'smiles': None,
        'molecular_formula_rdkit': None,
        'molecular_weight_rdkit': None,
        'rdkit_available': False
    }
    
    if not RDKIT_AVAILABLE:
        return enrichment
    
    if not formula:
        return enrichment
    
    try:
        # Try to get SMILES from formula
        smiles = formula_to_smiles(formula)
        if not smiles:
            # Try to create molecule from formula string (limited success)
            # Most formulas need proper parsing
            return enrichment
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            enrichment['smiles'] = smiles
            enrichment['num_atoms'] = mol.GetNumAtoms()
            enrichment['molecular_formula_rdkit'] = Chem.rdMolDescriptors.CalcMolFormula(mol)
            enrichment['molecular_weight_rdkit'] = Descriptors.MolWt(mol)
            enrichment['rdkit_available'] = True
    except Exception as e:
        # Silently fail - not all formulas can be converted
        pass
    
    return enrichment


def extract_compound_data(compound: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all structured data from a compound JSON."""
    main_entry = compound.get("main_entry_content", "")
    
    # Basic fields
    data = {
        'Compound_ID': compound.get("compound_id"),
        'Name': clean_text(compound.get("name", "")),
        'Other_Names': extract_other_names(main_entry),
        'Formula': extract_formula(main_entry),
        'Elements': extract_elements(main_entry),
        'Compound_Type': extract_compound_type(main_entry),
        'State': extract_state(main_entry),
        'Molecular_Weight': extract_molecular_weight(main_entry),
        'Melting_Point': extract_melting_point(main_entry),
        'Boiling_Point': extract_boiling_point(main_entry),
        'Solubility': extract_solubility(main_entry),
        'Overview': extract_section(main_entry, 'OVERVIEW'),
        'How_It_Is_Made': extract_section(main_entry, 'HOW IT IS MADE'),
        'Common_Uses': extract_section(main_entry, 'COMMON USES'),
        'Hazards': extract_section(main_entry, 'HAZARDS'),
        'Interesting_Facts': extract_section(main_entry, 'INTERESTING FACTS'),
        'Total_References': compound.get("total_references", 0),
    }
    
    # Add missing value flags
    data['Has_Other_Names'] = len(data['Other_Names']) > 0
    data['Has_Formula'] = data['Formula'] is not None
    data['Has_Elements'] = len(data['Elements']) > 0
    data['Has_Compound_Type'] = data['Compound_Type'] is not None
    data['Has_State'] = data['State'] is not None
    data['Has_Molecular_Weight'] = data['Molecular_Weight'] is not None
    data['Has_Melting_Point'] = data['Melting_Point'] is not None
    data['Has_Boiling_Point'] = data['Boiling_Point'] is not None
    data['Has_Solubility'] = data['Solubility'] is not None
    data['Has_Overview'] = data['Overview'] is not None
    data['Has_How_It_Is_Made'] = data['How_It_Is_Made'] is not None
    data['Has_Common_Uses'] = data['Common_Uses'] is not None
    data['Has_Hazards'] = data['Hazards'] is not None
    data['Has_Interesting_Facts'] = data['Interesting_Facts'] is not None
    
    # RDKit enrichment
    if data['Formula']:
        rdkit_data = enrich_with_rdkit(data['Formula'])
        data.update({
            'Num_Atoms': rdkit_data.get('num_atoms'),
            'SMILES': rdkit_data.get('smiles'),
            'Molecular_Formula_RDKit': rdkit_data.get('molecular_formula_rdkit'),
            'Molecular_Weight_RDKit': rdkit_data.get('molecular_weight_rdkit'),
            'RDKit_Available': rdkit_data.get('rdkit_available', False)
        })
    else:
        data.update({
            'Num_Atoms': None,
            'SMILES': None,
            'Molecular_Formula_RDKit': None,
            'Molecular_Weight_RDKit': None,
            'RDKit_Available': False
        })
    
    return data


def load_all_compounds() -> List[Dict[str, Any]]:
    """Load all compound JSON files."""
    compounds = []
    json_files = sorted(DATA_DIR.glob("*.json"))
    
    print(f"📁 Loading {len(json_files)} compound files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                compounds.append(data)
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
    
    return compounds


def create_dataframe(compounds: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create DataFrame from extracted compound data."""
    print("📊 Extracting structured data...")
    
    rows = []
    for i, compound in enumerate(compounds):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(compounds)} compounds...")
        
        row_data = extract_compound_data(compound)
        rows.append(row_data)
    
    print(f"✅ Extracted data from {len(rows)} compounds")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert Elements and Other_Names lists to strings for CSV compatibility
    # Keep as lists in the DataFrame for JSON export
    df['Elements_Str'] = df['Elements'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    df['Other_Names_Str'] = df['Other_Names'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    
    return df


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generate a summary report of the extracted data."""
    report = []
    report.append("# Structured Data Extraction Summary\n\n")
    report.append("---\n\n")
    
    # Basic Statistics
    report.append("## Dataset Overview\n\n")
    report.append(f"- **Total Compounds:** {len(df)}\n")
    report.append(f"- **Total Columns:** {len(df.columns)}\n")
    report.append("\n")
    
    # Completeness Statistics
    report.append("## Data Completeness\n\n")
    report.append("| Field | Present | Missing | Percentage |\n")
    report.append("|-------|---------|---------|------------|\n")
    
    flag_columns = [col for col in df.columns if col.startswith('Has_')]
    for flag_col in flag_columns:
        field_name = flag_col.replace('Has_', '').replace('_', ' ')
        present = df[flag_col].sum()
        missing = len(df) - present
        pct = (present / len(df)) * 100 if len(df) > 0 else 0
        report.append(f"| {field_name} | {present} | {missing} | {pct:.1f}% |\n")
    
    report.append("\n")
    
    # RDKit Statistics
    if 'RDKit_Available' in df.columns:
        rdkit_count = df['RDKit_Available'].sum()
        report.append("## RDKit Enrichment\n\n")
        report.append(f"- **Compounds with RDKit Data:** {rdkit_count} ({rdkit_count/len(df)*100:.1f}%)\n")
        report.append(f"- **Compounds without RDKit Data:** {len(df) - rdkit_count}\n")
        report.append("\n")
    
    # Formula Statistics
    if 'Formula' in df.columns:
        formula_count = df['Formula'].notna().sum()
        report.append("## Formula Statistics\n\n")
        report.append(f"- **Compounds with Formula:** {formula_count} ({formula_count/len(df)*100:.1f}%)\n")
        report.append("\n")
    
    # Element Statistics
    if 'Elements' in df.columns:
        all_elements = []
        for elem_list in df['Elements']:
            if isinstance(elem_list, list):
                all_elements.extend(elem_list)
        element_counts = pd.Series(all_elements).value_counts()
        report.append("## Most Common Elements\n\n")
        for elem, count in element_counts.head(10).items():
            report.append(f"- **{elem}:** {count} compounds\n")
        report.append("\n")
    
    # State Distribution
    if 'State' in df.columns:
        state_counts = df['State'].value_counts()
        report.append("## State Distribution\n\n")
        for state, count in state_counts.items():
            if pd.notna(state):
                report.append(f"- **{state}:** {count} compounds\n")
        report.append("\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ Summary report saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Structured Data Extraction - Individual Compounds Dataset")
    print("=" * 60)
    print()
    
    if not RDKIT_AVAILABLE:
        print("⚠️  RDKit not available. Enrichment features will be limited.")
        print("   Install with: conda install -c conda-forge rdkit\n")
    
    # Load compounds
    compounds = load_all_compounds()
    
    if not compounds:
        print("❌ No compounds loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(compounds)} compounds\n")
    
    # Create DataFrame
    df = create_dataframe(compounds)
    
    # Save DataFrame
    # CSV version (with string representations of lists)
    csv_path = OUTPUT_DIR / "04_compounds_structured.csv"
    df_csv = df.copy()
    # Replace list columns with string versions for CSV
    df_csv['Elements'] = df_csv['Elements_Str']
    df_csv['Other_Names'] = df_csv['Other_Names_Str']
    df_csv = df_csv.drop(columns=['Elements_Str', 'Other_Names_Str'])
    df_csv.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV saved to: {csv_path}")
    
    # JSON version (preserves lists)
    json_path = OUTPUT_DIR / "04_compounds_structured.json"
    # Convert DataFrame to records, handling lists properly
    records = df.to_dict('records')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ JSON saved to: {json_path}")
    
    # Parquet version (efficient, preserves types)
    parquet_path = OUTPUT_DIR / "04_compounds_structured.parquet"
    try:
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        print(f"✅ Parquet saved to: {parquet_path}")
    except Exception as e:
        print(f"⚠️  Could not save Parquet: {e}")
    
    # Generate summary report
    report_path = OUTPUT_DIR / "04_extraction_summary_report.md"
    generate_summary_report(df, report_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Compounds: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    if 'Formula' in df.columns:
        print(f"Compounds with Formula: {df['Formula'].notna().sum()}")
    if 'RDKit_Available' in df.columns:
        print(f"Compounds with RDKit Data: {df['RDKit_Available'].sum()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

