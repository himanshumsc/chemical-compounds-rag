#!/usr/bin/env python3
"""
Reference Analysis for Individual Compounds Dataset
Analyzes reference patterns, types, and cross-references.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
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


def analyze_references(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze references across all compounds."""
    
    analysis = {
        "reference_counts": {},
        "reference_types": {},
        "reference_pages": {},
        "reference_variations": {},
        "cross_reference_patterns": {}
    }
    
    # Reference count statistics
    total_refs = []
    ref_types_found = Counter()
    page_numbers = []
    variation_counts = []
    
    # Reference type distribution
    ref_type_combinations = Counter()
    
    # Cross-reference analysis
    compounds_with_refs = 0
    compounds_without_refs = 0
    
    for compound in compounds:
        compound_id = compound.get("compound_id")
        total_ref_count = compound.get("total_references", 0)
        total_refs.append(total_ref_count)
        
        if total_ref_count > 0:
            compounds_with_refs += 1
        else:
            compounds_without_refs += 1
        
        # Reference types
        ref_types = compound.get("reference_types_found", [])
        if ref_types:
            for ref_type in ref_types:
                ref_types_found[ref_type] += 1
            # Track combinations
            ref_type_combinations[tuple(sorted(ref_types))] += 1
        
        # Analyze other references
        refs_breakdown = compound.get("references_breakdown", {})
        other_refs = refs_breakdown.get("other_references", [])
        
        for ref in other_refs:
            # Page numbers
            page_num = ref.get("page_number")
            if page_num:
                page_numbers.append(page_num)
            
            # Reference types
            ref_type = ref.get("reference_type")
            if ref_type:
                ref_types_found[ref_type] += 1
            
            # Variations
            variations = ref.get("found_variations", [])
            if variations:
                variation_counts.append(len(variations))
    
    # Compile statistics
    if total_refs:
        analysis["reference_counts"] = {
            "total_compounds": len(compounds),
            "compounds_with_refs": compounds_with_refs,
            "compounds_without_refs": compounds_without_refs,
            "statistics": {
                "min": min(total_refs),
                "max": max(total_refs),
                "mean": statistics.mean(total_refs),
                "median": statistics.median(total_refs),
                "total": sum(total_refs)
            },
            "distribution": {
                "zero": sum(1 for r in total_refs if r == 0),
                "one_to_five": sum(1 for r in total_refs if 1 <= r <= 5),
                "six_to_ten": sum(1 for r in total_refs if 6 <= r <= 10),
                "eleven_to_twenty": sum(1 for r in total_refs if 11 <= r <= 20),
                "twenty_one_plus": sum(1 for r in total_refs if r > 20)
            }
        }
    
    # Reference types
    analysis["reference_types"] = {
        "unique_types": len(ref_types_found),
        "frequency": dict(ref_types_found.most_common()),
        "combinations": {", ".join(k): v for k, v in ref_type_combinations.most_common(10)}
    }
    
    # Page numbers
    if page_numbers:
        analysis["reference_pages"] = {
            "total_pages_referenced": len(page_numbers),
            "unique_pages": len(set(page_numbers)),
            "statistics": {
                "min": min(page_numbers),
                "max": max(page_numbers),
                "mean": statistics.mean(page_numbers),
                "median": statistics.median(page_numbers)
            },
            "most_referenced_pages": dict(Counter(page_numbers).most_common(20))
        }
    
    # Variation analysis
    if variation_counts:
        analysis["reference_variations"] = {
            "statistics": {
                "min": min(variation_counts),
                "max": max(variation_counts),
                "mean": statistics.mean(variation_counts),
                "median": statistics.median(variation_counts)
            },
            "compounds_with_variations": sum(1 for c in variation_counts if c > 0)
        }
    
    # Detailed reference analysis per compound
    detailed_refs = []
    for compound in compounds:
        compound_id = compound.get("compound_id")
        name = compound.get("name", "").strip()
        total_refs_count = compound.get("total_references", 0)
        ref_types = compound.get("reference_types_found", [])
        
        detailed_refs.append({
            "compound_id": compound_id,
            "name": name,
            "total_references": total_refs_count,
            "reference_types": ref_types
        })
    
    analysis["detailed_references"] = detailed_refs
    
    return analysis


def generate_reference_report(analysis: Dict[str, Any], output_path: Path):
    """Generate a markdown reference analysis report."""
    
    report = []
    report.append("# Reference Analysis Report - Individual Compounds Dataset\n\n")
    report.append("---\n\n")
    
    # Reference Counts
    if "reference_counts" in analysis:
        report.append("## Reference Count Statistics\n\n")
        ref_counts = analysis["reference_counts"]
        report.append(f"- **Total Compounds:** {ref_counts['total_compounds']}\n")
        report.append(f"- **Compounds with References:** {ref_counts['compounds_with_refs']}\n")
        report.append(f"- **Compounds without References:** {ref_counts['compounds_without_refs']}\n")
        
        if "statistics" in ref_counts:
            stats = ref_counts["statistics"]
            report.append(f"\n### Statistics\n\n")
            report.append(f"- **Min:** {stats['min']}\n")
            report.append(f"- **Max:** {stats['max']}\n")
            report.append(f"- **Mean:** {stats['mean']:.2f}\n")
            report.append(f"- **Median:** {stats['median']:.2f}\n")
            report.append(f"- **Total References:** {stats['total']}\n")
        
        if "distribution" in ref_counts:
            dist = ref_counts["distribution"]
            report.append(f"\n### Distribution\n\n")
            report.append(f"- **0 references:** {dist['zero']} compounds\n")
            report.append(f"- **1-5 references:** {dist['one_to_five']} compounds\n")
            report.append(f"- **6-10 references:** {dist['six_to_ten']} compounds\n")
            report.append(f"- **11-20 references:** {dist['eleven_to_twenty']} compounds\n")
            report.append(f"- **21+ references:** {dist['twenty_one_plus']} compounds\n")
        
        report.append("\n")
    
    # Reference Types
    if "reference_types" in analysis:
        report.append("## Reference Types\n\n")
        ref_types = analysis["reference_types"]
        report.append(f"- **Unique Types:** {ref_types['unique_types']}\n")
        
        if ref_types.get("frequency"):
            report.append("\n### Type Frequency\n\n")
            for ref_type, count in list(ref_types["frequency"].items()):
                report.append(f"- **{ref_type}:** {count} occurrences\n")
        
        if ref_types.get("combinations"):
            report.append("\n### Type Combinations\n\n")
            for combo, count in list(ref_types["combinations"].items())[:10]:
                report.append(f"- **{combo}:** {count} compounds\n")
        
        report.append("\n")
    
    # Reference Pages
    if "reference_pages" in analysis:
        report.append("## Reference Page Analysis\n\n")
        ref_pages = analysis["reference_pages"]
        report.append(f"- **Total Page References:** {ref_pages['total_pages_referenced']}\n")
        report.append(f"- **Unique Pages:** {ref_pages['unique_pages']}\n")
        
        if "statistics" in ref_pages:
            stats = ref_pages["statistics"]
            report.append(f"\n### Page Statistics\n\n")
            report.append(f"- **Min Page:** {stats['min']}\n")
            report.append(f"- **Max Page:** {stats['max']}\n")
            report.append(f"- **Mean Page:** {stats['mean']:.1f}\n")
            report.append(f"- **Median Page:** {stats['median']:.1f}\n")
        
        if ref_pages.get("most_referenced_pages"):
            report.append(f"\n### Most Referenced Pages\n\n")
            for page, count in list(ref_pages["most_referenced_pages"].items())[:15]:
                report.append(f"- **Page {page}:** {count} references\n")
        
        report.append("\n")
    
    # Variation Analysis
    if "reference_variations" in analysis:
        report.append("## Reference Variation Analysis\n\n")
        variations = analysis["reference_variations"]
        if "statistics" in variations:
            stats = variations["statistics"]
            report.append(f"- **Variations per Reference:** Min={stats['min']}, Max={stats['max']}, Mean={stats['mean']:.2f}\n")
        report.append("\n")
    
    # Top compounds by reference count
    if "detailed_references" in analysis:
        report.append("## Top Compounds by Reference Count\n\n")
        detailed = sorted(analysis["detailed_references"], 
                         key=lambda x: x.get("total_references", 0), 
                         reverse=True)
        report.append("| Compound ID | Name | References | Types |\n")
        report.append("|-------------|------|------------|-------|\n")
        for comp in detailed[:20]:
            ref_types_str = ", ".join(comp.get("reference_types", []))
            report.append(f"| {comp.get('compound_id')} | {comp.get('name', '')[:40]} | {comp.get('total_references', 0)} | {ref_types_str} |\n")
        report.append("\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ Reference analysis report saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Reference Analysis - Individual Compounds Dataset")
    print("=" * 60)
    print()
    
    # Load data
    compounds = load_all_compounds()
    
    if not compounds:
        print("❌ No compounds loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(compounds)} compounds\n")
    
    # Analyze references
    print("📊 Analyzing references...")
    analysis = analyze_references(compounds)
    
    # Save analysis as JSON
    analysis_path = OUTPUT_DIR / "03_reference_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"✅ Analysis saved to: {analysis_path}")
    
    # Generate report
    report_path = OUTPUT_DIR / "03_reference_analysis_report.md"
    generate_reference_report(analysis, report_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "reference_counts" in analysis:
        rc = analysis["reference_counts"]
        print(f"Compounds with refs: {rc['compounds_with_refs']}")
        if "statistics" in rc:
            print(f"Avg references: {rc['statistics']['mean']:.2f}")
    if "reference_types" in analysis:
        print(f"Unique reference types: {analysis['reference_types']['unique_types']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

