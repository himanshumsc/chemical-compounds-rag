#!/usr/bin/env python3
"""
Basic Statistics Analysis for Individual Compounds Dataset
Computes fundamental statistics about the dataset structure and completeness.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import statistics

# Add parent directory to path for imports if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
DATA_DIR = Path("/home/himanshu/MSC_FINAL/dev/test/test/data/processed/individual_compounds")
OUTPUT_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_compounds() -> List[Dict[str, Any]]:
    """Load all compound JSON files."""
    compounds = []
    json_files = sorted(DATA_DIR.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {DATA_DIR}")
        return compounds
    
    print(f"📁 Loading {len(json_files)} compound files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                compounds.append(data)
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
    
    return compounds


def analyze_basic_statistics(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute basic statistics about the dataset."""
    
    stats = {
        "total_compounds": len(compounds),
        "file_analysis": {},
        "field_presence": {},
        "data_completeness": {},
        "naming_patterns": {}
    }
    
    # Field presence analysis
    all_fields = set()
    for compound in compounds:
        all_fields.update(compound.keys())
    
    field_presence = {}
    for field in all_fields:
        present_count = sum(1 for c in compounds if field in c and c[field] is not None)
        field_presence[field] = {
            "present": present_count,
            "missing": len(compounds) - present_count,
            "percentage": (present_count / len(compounds)) * 100 if compounds else 0
        }
    
    stats["field_presence"] = field_presence
    
    # Compound ID analysis
    compound_ids = [c.get("compound_id") for c in compounds if c.get("compound_id") is not None]
    if compound_ids:
        stats["compound_id"] = {
            "min": min(compound_ids),
            "max": max(compound_ids),
            "unique_count": len(set(compound_ids)),
            "duplicates": len(compound_ids) - len(set(compound_ids)),
            "missing": len(compounds) - len(compound_ids)
        }
    
    # Name analysis
    names = [c.get("name", "").strip() for c in compounds]
    name_lengths = [len(name) for name in names if name]
    if name_lengths:
        stats["name"] = {
            "total": len(names),
            "non_empty": len([n for n in names if n]),
            "length_stats": {
                "min": min(name_lengths),
                "max": max(name_lengths),
                "mean": statistics.mean(name_lengths),
                "median": statistics.median(name_lengths)
            }
        }
    
    # Page information analysis
    arabic_starts = [c.get("arabic_start_page") for c in compounds if c.get("arabic_start_page") is not None]
    arabic_ends = [c.get("arabic_end_page") for c in compounds if c.get("arabic_end_page") is not None]
    pdf_pages = [c.get("pdf_start_page") for c in compounds if c.get("pdf_start_page") is not None]
    total_pages = [c.get("total_pages") for c in compounds if c.get("total_pages") is not None]
    
    if arabic_starts:
        stats["pages"] = {
            "arabic_start": {
                "min": min(arabic_starts),
                "max": max(arabic_starts),
                "mean": statistics.mean(arabic_starts),
                "median": statistics.median(arabic_starts)
            },
            "arabic_end": {
                "min": min(arabic_ends) if arabic_ends else None,
                "max": max(arabic_ends) if arabic_ends else None,
                "mean": statistics.mean(arabic_ends) if arabic_ends else None,
                "median": statistics.median(arabic_ends) if arabic_ends else None
            },
            "pdf_start": {
                "min": min(pdf_pages) if pdf_pages else None,
                "max": max(pdf_pages) if pdf_pages else None,
                "mean": statistics.mean(pdf_pages) if pdf_pages else None,
                "median": statistics.median(pdf_pages) if pdf_pages else None
            },
            "total_pages": {
                "min": min(total_pages) if total_pages else None,
                "max": max(total_pages) if total_pages else None,
                "mean": statistics.mean(total_pages) if total_pages else None,
                "median": statistics.median(total_pages) if total_pages else None
            }
        }
    
    # Reference analysis
    total_refs = [c.get("total_references", 0) for c in compounds]
    if total_refs:
        stats["references"] = {
            "min": min(total_refs),
            "max": max(total_refs),
            "mean": statistics.mean(total_refs),
            "median": statistics.median(total_refs),
            "zero_refs": sum(1 for r in total_refs if r == 0)
        }
    
    # Text length analysis
    main_lengths = [c.get("main_entry_length", 0) for c in compounds]
    comp_lengths = [c.get("comprehensive_text_length", 0) for c in compounds]
    
    if main_lengths:
        stats["text_lengths"] = {
            "main_entry": {
                "min": min(main_lengths),
                "max": max(main_lengths),
                "mean": statistics.mean(main_lengths),
                "median": statistics.median(main_lengths),
                "total": sum(main_lengths)
            },
            "comprehensive": {
                "min": min(comp_lengths) if comp_lengths else 0,
                "max": max(comp_lengths) if comp_lengths else 0,
                "mean": statistics.mean(comp_lengths) if comp_lengths else 0,
                "median": statistics.median(comp_lengths) if comp_lengths else 0,
                "total": sum(comp_lengths) if comp_lengths else 0
            }
        }
    
    # Naming pattern analysis
    file_patterns = Counter()
    for compound in compounds:
        name = compound.get("name", "")
        if name:
            # Check for common patterns
            if " " in name:
                file_patterns["contains_space"] += 1
            if "." in name:
                file_patterns["contains_dot"] += 1
            if "(" in name or ")" in name:
                file_patterns["contains_parentheses"] += 1
            if "-" in name:
                file_patterns["contains_hyphen"] += 1
    
    stats["naming_patterns"] = dict(file_patterns)
    
    return stats


def generate_summary_report(stats: Dict[str, Any], output_path: Path):
    """Generate a markdown summary report."""
    
    report = []
    report.append("# Basic Statistics Report - Individual Compounds Dataset\n")
    report.append(f"**Generated:** {Path(__file__).name}\n")
    report.append("---\n\n")
    
    # Overview
    report.append("## Dataset Overview\n\n")
    report.append(f"- **Total Compounds:** {stats['total_compounds']}\n")
    
    if "compound_id" in stats:
        cid = stats["compound_id"]
        report.append(f"- **Compound IDs:** {cid['min']} to {cid['max']} ({cid['unique_count']} unique)\n")
        if cid['duplicates'] > 0:
            report.append(f"  - ⚠️ **Duplicates:** {cid['duplicates']}\n")
        if cid['missing'] > 0:
            report.append(f"  - ⚠️ **Missing IDs:** {cid['missing']}\n")
    
    report.append("\n")
    
    # Field Presence
    report.append("## Field Presence Analysis\n\n")
    report.append("| Field | Present | Missing | Percentage |\n")
    report.append("|-------|---------|---------|------------|\n")
    
    for field, info in sorted(stats["field_presence"].items()):
        status = "✅" if info["percentage"] == 100 else "⚠️" if info["percentage"] >= 90 else "❌"
        report.append(f"| {field} | {info['present']} | {info['missing']} | {info['percentage']:.1f}% {status} |\n")
    
    report.append("\n")
    
    # Name Statistics
    if "name" in stats:
        report.append("## Name Statistics\n\n")
        name_stats = stats["name"]
        report.append(f"- **Total Names:** {name_stats['total']}\n")
        report.append(f"- **Non-empty Names:** {name_stats['non_empty']}\n")
        if "length_stats" in name_stats:
            ls = name_stats["length_stats"]
            report.append(f"- **Length:** Min={ls['min']}, Max={ls['max']}, Mean={ls['mean']:.1f}, Median={ls['median']:.1f}\n")
        report.append("\n")
    
    # Page Statistics
    if "pages" in stats:
        report.append("## Page Statistics\n\n")
        pages = stats["pages"]
        if "arabic_start" in pages:
            as_start = pages["arabic_start"]
            report.append(f"- **Arabic Start Page:** Min={as_start['min']}, Max={as_start['max']}, Mean={as_start['mean']:.1f}\n")
        if "total_pages" in pages and pages["total_pages"]["min"] is not None:
            tp = pages["total_pages"]
            report.append(f"- **Total Pages:** Min={tp['min']}, Max={tp['max']}, Mean={tp['mean']:.1f}, Median={tp['median']:.1f}\n")
        report.append("\n")
    
    # Reference Statistics
    if "references" in stats:
        report.append("## Reference Statistics\n\n")
        refs = stats["references"]
        report.append(f"- **Total References:** Min={refs['min']}, Max={refs['max']}, Mean={refs['mean']:.1f}, Median={refs['median']:.1f}\n")
        report.append(f"- **Compounds with Zero References:** {refs['zero_refs']}\n")
        report.append("\n")
    
    # Text Length Statistics
    if "text_lengths" in stats:
        report.append("## Text Length Statistics\n\n")
        tl = stats["text_lengths"]
        if "main_entry" in tl:
            me = tl["main_entry"]
            report.append(f"### Main Entry\n")
            report.append(f"- **Length:** Min={me['min']:,}, Max={me['max']:,}, Mean={me['mean']:.1f}, Median={me['median']:.1f}\n")
            report.append(f"- **Total Characters:** {me['total']:,}\n")
        if "comprehensive" in tl:
            comp = tl["comprehensive"]
            report.append(f"### Comprehensive Text\n")
            report.append(f"- **Length:** Min={comp['min']:,}, Max={comp['max']:,}, Mean={comp['mean']:.1f}, Median={comp['median']:.1f}\n")
            report.append(f"- **Total Characters:** {comp['total']:,}\n")
        report.append("\n")
    
    # Naming Patterns
    if "naming_patterns" in stats and stats["naming_patterns"]:
        report.append("## Naming Patterns\n\n")
        for pattern, count in sorted(stats["naming_patterns"].items()):
            report.append(f"- **{pattern.replace('_', ' ').title()}:** {count}\n")
        report.append("\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ Summary report saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Basic Statistics Analysis - Individual Compounds Dataset")
    print("=" * 60)
    print()
    
    # Load data
    compounds = load_all_compounds()
    
    if not compounds:
        print("❌ No compounds loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(compounds)} compounds\n")
    
    # Compute statistics
    print("📊 Computing basic statistics...")
    stats = analyze_basic_statistics(compounds)
    
    # Save statistics as JSON
    stats_path = OUTPUT_DIR / "01_basic_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✅ Statistics saved to: {stats_path}")
    
    # Generate summary report
    report_path = OUTPUT_DIR / "01_basic_statistics_report.md"
    generate_summary_report(stats, report_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Compounds: {stats['total_compounds']}")
    if "compound_id" in stats:
        print(f"Compound ID Range: {stats['compound_id']['min']} - {stats['compound_id']['max']}")
    if "text_lengths" in stats and "main_entry" in stats["text_lengths"]:
        me = stats["text_lengths"]["main_entry"]
        print(f"Main Entry Length: Avg={me['mean']:.1f}, Median={me['median']:.1f}")
    if "references" in stats:
        print(f"References: Avg={stats['references']['mean']:.1f}, Max={stats['references']['max']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

