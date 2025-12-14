#!/usr/bin/env python3
"""
Content Analysis for Individual Compounds Dataset
Analyzes text content, chemical information, and content structure.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set
import statistics

# Configuration
DATA_DIR = Path("/home/himanshu/MSC_FINAL/dev/test/test/data/processed/individual_compounds")
OUTPUT_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_compounds() -> List[Dict[str, Any]]:
    """Load all compound JSON files."""
    compounds = []
    json_files = sorted(DATA_DIR.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                compounds.append(data)
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
    
    return compounds


def extract_chemical_formula(text: str) -> List[str]:
    """Extract chemical formulas from text using regex patterns."""
    # Pattern for chemical formulas (e.g., CH3COOH, Al(OH)3, C20H30O)
    # Matches: letters followed by optional numbers, parentheses with subscripts
    pattern = r'\b[A-Z][a-z]?(?:[0-9]+)?(?:\([A-Z][a-z]?[0-9]*\)[0-9]*)*(?:[A-Z][a-z]?(?:[0-9]+)?)*\b'
    formulas = re.findall(pattern, text)
    # Filter out common words that match the pattern
    common_words = {'OVERVIEW', 'STATE', 'FORMULA', 'ELEMENTS', 'OTHER', 'NAMES', 'KEY', 'FACTS'}
    formulas = [f for f in formulas if f not in common_words and len(f) > 1]
    return formulas


def extract_elements(text: str) -> Set[str]:
    """Extract element names from text."""
    # Common chemical elements
    elements = {
        'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon', 'Nitrogen', 'Oxygen',
        'Fluorine', 'Neon', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur',
        'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Iron', 'Copper', 'Zinc', 'Silver', 'Gold',
        'Mercury', 'Lead', 'Tin', 'Iodine', 'Bromine', 'Chlorine', 'Sulfur', 'Phosphorus'
    }
    
    found_elements = set()
    text_upper = text.upper()
    
    for element in elements:
        if element.upper() in text_upper or element in text:
            found_elements.add(element)
    
    return found_elements


def extract_compound_type(text: str) -> str:
    """Extract compound type from text."""
    # Look for "COMPOUND TYPE:" pattern
    pattern = r'COMPOUND TYPE:\s*([^\n]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_state(text: str) -> str:
    """Extract state (Solid/Liquid/Gas) from text."""
    # Look for "STATE:" pattern
    pattern = r'STATE:\s*([^\n]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_molecular_weight(text: str) -> float:
    """Extract molecular weight from text."""
    # Look for "MOLECULAR WEIGHT:" pattern
    pattern = r'MOLECULAR WEIGHT:\s*([0-9.]+)\s*g/mol'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def count_sentences(text: str) -> int:
    """Approximate sentence count."""
    if not text:
        return 0
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+\s+', text)
    return len([s for s in sentences if s.strip()])


def analyze_content(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze content of all compounds."""
    
    analysis = {
        "text_metrics": {},
        "chemical_information": {},
        "content_structure": {},
        "section_presence": {}
    }
    
    # Text metrics
    main_entry_lengths = []
    comprehensive_lengths = []
    main_entry_words = []
    comprehensive_words = []
    main_entry_sentences = []
    comprehensive_sentences = []
    
    # Chemical information
    all_formulas = []
    all_elements = Counter()
    compound_types = Counter()
    states = Counter()
    molecular_weights = []
    
    # Content structure
    sections_found = defaultdict(int)
    
    for compound in compounds:
        # Main entry analysis
        main_entry = compound.get("main_entry_content", "")
        if main_entry:
            main_entry_lengths.append(len(main_entry))
            main_entry_words.append(count_words(main_entry))
            main_entry_sentences.append(count_sentences(main_entry))
            
            # Extract chemical information
            formulas = extract_chemical_formula(main_entry)
            all_formulas.extend(formulas)
            
            elements = extract_elements(main_entry)
            all_elements.update(elements)
            
            comp_type = extract_compound_type(main_entry)
            if comp_type:
                compound_types[comp_type] += 1
            
            state = extract_state(main_entry)
            if state:
                states[state] += 1
            
            mw = extract_molecular_weight(main_entry)
            if mw:
                molecular_weights.append(mw)
            
            # Check for key sections
            if "FORMULA:" in main_entry:
                sections_found["FORMULA"] += 1
            if "ELEMENTS:" in main_entry:
                sections_found["ELEMENTS"] += 1
            if "COMPOUND TYPE:" in main_entry:
                sections_found["COMPOUND_TYPE"] += 1
            if "STATE:" in main_entry:
                sections_found["STATE"] += 1
            if "MOLECULAR WEIGHT:" in main_entry:
                sections_found["MOLECULAR_WEIGHT"] += 1
            if "MELTING POINT:" in main_entry:
                sections_found["MELTING_POINT"] += 1
            if "BOILING POINT:" in main_entry:
                sections_found["BOILING_POINT"] += 1
            if "SOLUBILITY:" in main_entry:
                sections_found["SOLUBILITY"] += 1
            if "OVERVIEW" in main_entry:
                sections_found["OVERVIEW"] += 1
            if "HOW IT IS MADE" in main_entry:
                sections_found["HOW_IT_IS_MADE"] += 1
            if "COMMON USES" in main_entry:
                sections_found["COMMON_USES"] += 1
        
        # Comprehensive text analysis
        comprehensive = compound.get("comprehensive_text", "")
        if comprehensive:
            comprehensive_lengths.append(len(comprehensive))
            comprehensive_words.append(count_words(comprehensive))
            comprehensive_sentences.append(count_sentences(comprehensive))
    
    # Compile text metrics
    if main_entry_lengths:
        analysis["text_metrics"]["main_entry"] = {
            "character_count": {
                "min": min(main_entry_lengths),
                "max": max(main_entry_lengths),
                "mean": statistics.mean(main_entry_lengths),
                "median": statistics.median(main_entry_lengths),
                "total": sum(main_entry_lengths)
            },
            "word_count": {
                "min": min(main_entry_words) if main_entry_words else 0,
                "max": max(main_entry_words) if main_entry_words else 0,
                "mean": statistics.mean(main_entry_words) if main_entry_words else 0,
                "median": statistics.median(main_entry_words) if main_entry_words else 0,
                "total": sum(main_entry_words) if main_entry_words else 0
            },
            "sentence_count": {
                "min": min(main_entry_sentences) if main_entry_sentences else 0,
                "max": max(main_entry_sentences) if main_entry_sentences else 0,
                "mean": statistics.mean(main_entry_sentences) if main_entry_sentences else 0,
                "median": statistics.median(main_entry_sentences) if main_entry_sentences else 0
            }
        }
    
    if comprehensive_lengths:
        analysis["text_metrics"]["comprehensive"] = {
            "character_count": {
                "min": min(comprehensive_lengths),
                "max": max(comprehensive_lengths),
                "mean": statistics.mean(comprehensive_lengths),
                "median": statistics.median(comprehensive_lengths),
                "total": sum(comprehensive_lengths)
            },
            "word_count": {
                "min": min(comprehensive_words) if comprehensive_words else 0,
                "max": max(comprehensive_words) if comprehensive_words else 0,
                "mean": statistics.mean(comprehensive_words) if comprehensive_words else 0,
                "median": statistics.median(comprehensive_words) if comprehensive_words else 0,
                "total": sum(comprehensive_words) if comprehensive_words else 0
            },
            "sentence_count": {
                "min": min(comprehensive_sentences) if comprehensive_sentences else 0,
                "max": max(comprehensive_sentences) if comprehensive_sentences else 0,
                "mean": statistics.mean(comprehensive_sentences) if comprehensive_sentences else 0,
                "median": statistics.median(comprehensive_sentences) if comprehensive_sentences else 0
            }
        }
    
    # Chemical information
    formula_counter = Counter(all_formulas)
    analysis["chemical_information"] = {
        "unique_formulas": len(formula_counter),
        "total_formula_mentions": len(all_formulas),
        "top_formulas": dict(formula_counter.most_common(20)),
        "element_frequency": dict(all_elements.most_common()),
        "compound_types": dict(compound_types.most_common()),
        "states": dict(states.most_common()),
        "molecular_weights": {
            "count": len(molecular_weights),
            "min": min(molecular_weights) if molecular_weights else None,
            "max": max(molecular_weights) if molecular_weights else None,
            "mean": statistics.mean(molecular_weights) if molecular_weights else None,
            "median": statistics.median(molecular_weights) if molecular_weights else None
        }
    }
    
    # Content structure
    analysis["section_presence"] = dict(sections_found)
    analysis["content_structure"] = {
        "total_compounds": len(compounds),
        "sections_present": len(sections_found),
        "section_coverage": {k: (v / len(compounds)) * 100 for k, v in sections_found.items()}
    }
    
    return analysis


def generate_content_report(analysis: Dict[str, Any], output_path: Path):
    """Generate a markdown content analysis report."""
    
    report = []
    report.append("# Content Analysis Report - Individual Compounds Dataset\n\n")
    report.append("---\n\n")
    
    # Text Metrics
    report.append("## Text Metrics\n\n")
    if "main_entry" in analysis["text_metrics"]:
        me = analysis["text_metrics"]["main_entry"]
        report.append("### Main Entry Content\n\n")
        if "character_count" in me:
            cc = me["character_count"]
            report.append(f"- **Characters:** Min={cc['min']:,}, Max={cc['max']:,}, Mean={cc['mean']:.1f}, Median={cc['median']:.1f}\n")
            report.append(f"- **Total Characters:** {cc['total']:,}\n")
        if "word_count" in me:
            wc = me["word_count"]
            report.append(f"- **Words:** Min={wc['min']:,}, Max={wc['max']:,}, Mean={wc['mean']:.1f}, Median={wc['median']:.1f}\n")
            report.append(f"- **Total Words:** {wc['total']:,}\n")
        if "sentence_count" in me:
            sc = me["sentence_count"]
            report.append(f"- **Sentences:** Min={sc['min']}, Max={sc['max']}, Mean={sc['mean']:.1f}, Median={sc['median']:.1f}\n")
        report.append("\n")
    
    if "comprehensive" in analysis["text_metrics"]:
        comp = analysis["text_metrics"]["comprehensive"]
        report.append("### Comprehensive Text\n\n")
        if "character_count" in comp:
            cc = comp["character_count"]
            report.append(f"- **Characters:** Min={cc['min']:,}, Max={cc['max']:,}, Mean={cc['mean']:.1f}, Median={cc['median']:.1f}\n")
            report.append(f"- **Total Characters:** {cc['total']:,}\n")
        if "word_count" in comp:
            wc = comp["word_count"]
            report.append(f"- **Words:** Min={wc['min']:,}, Max={wc['max']:,}, Mean={wc['mean']:.1f}, Median={wc['median']:.1f}\n")
        report.append("\n")
    
    # Chemical Information
    report.append("## Chemical Information\n\n")
    chem = analysis["chemical_information"]
    report.append(f"- **Unique Formulas Found:** {chem['unique_formulas']}\n")
    report.append(f"- **Total Formula Mentions:** {chem['total_formula_mentions']}\n")
    
    if chem.get("top_formulas"):
        report.append("\n### Top Formulas\n\n")
        for formula, count in list(chem["top_formulas"].items())[:10]:
            report.append(f"- `{formula}`: {count} occurrences\n")
    
    if chem.get("element_frequency"):
        report.append("\n### Element Frequency\n\n")
        for element, count in list(chem["element_frequency"].items())[:15]:
            report.append(f"- **{element}:** {count} occurrences\n")
    
    if chem.get("compound_types"):
        report.append("\n### Compound Types\n\n")
        for comp_type, count in list(chem["compound_types"].items())[:10]:
            report.append(f"- **{comp_type}:** {count} compounds\n")
    
    if chem.get("states"):
        report.append("\n### States\n\n")
        for state, count in chem["states"].items():
            report.append(f"- **{state}:** {count} compounds\n")
    
    if chem.get("molecular_weights") and chem["molecular_weights"]["count"] > 0:
        mw = chem["molecular_weights"]
        report.append("\n### Molecular Weights\n\n")
        report.append(f"- **Compounds with MW:** {mw['count']}\n")
        report.append(f"- **Range:** {mw['min']:.2f} - {mw['max']:.2f} g/mol\n")
        report.append(f"- **Mean:** {mw['mean']:.2f} g/mol\n")
        report.append(f"- **Median:** {mw['median']:.2f} g/mol\n")
    
    report.append("\n")
    
    # Content Structure
    report.append("## Content Structure\n\n")
    struct = analysis["content_structure"]
    report.append(f"- **Total Compounds:** {struct['total_compounds']}\n")
    report.append(f"- **Unique Sections Found:** {struct['sections_present']}\n")
    
    if analysis.get("section_presence"):
        report.append("\n### Section Presence\n\n")
        report.append("| Section | Present | Coverage % |\n")
        report.append("|---------|---------|------------|\n")
        for section, count in sorted(analysis["section_presence"].items(), key=lambda x: x[1], reverse=True):
            coverage = struct["section_coverage"].get(section, 0)
            report.append(f"| {section} | {count} | {coverage:.1f}% |\n")
    
    report.append("\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ Content analysis report saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Content Analysis - Individual Compounds Dataset")
    print("=" * 60)
    print()
    
    # Load data
    compounds = load_all_compounds()
    
    if not compounds:
        print("❌ No compounds loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(compounds)} compounds\n")
    
    # Analyze content
    print("📊 Analyzing content...")
    analysis = analyze_content(compounds)
    
    # Save analysis as JSON
    analysis_path = OUTPUT_DIR / "02_content_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"✅ Analysis saved to: {analysis_path}")
    
    # Generate report
    report_path = OUTPUT_DIR / "02_content_analysis_report.md"
    generate_content_report(analysis, report_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "text_metrics" in analysis and "main_entry" in analysis["text_metrics"]:
        me = analysis["text_metrics"]["main_entry"]
        if "word_count" in me:
            print(f"Main Entry Words: Avg={me['word_count']['mean']:.1f}")
    if "chemical_information" in analysis:
        chem = analysis["chemical_information"]
        print(f"Unique Formulas: {chem['unique_formulas']}")
        print(f"Unique Elements: {len(chem.get('element_frequency', {}))}")
    if "content_structure" in analysis:
        print(f"Sections Found: {analysis['content_structure']['sections_present']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

