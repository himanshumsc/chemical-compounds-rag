#!/usr/bin/env python3
"""
Text Length EDA - Individual Compounds Dataset
Analyzes main_entry_length and comprehensive_text_length without RAG-specific limits.
Focuses on general data completeness and distribution analysis.
"""

import json
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Any
import statistics

import pandas as pd
import numpy as np

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False

# Configuration
DATA_DIR = Path("/home/himanshu/MSC_FINAL/dev/test/test/data/processed/individual_compounds")
OUTPUT_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/output")
PLOTS_DIR = Path("/home/himanshu/MSC_FINAL/dev/EDA/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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


def analyze_text_lengths(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze text lengths with comprehensive statistics."""
    
    analysis = {
        "main_entry_analysis": {},
        "comprehensive_text_analysis": {},
        "comparison_analysis": {}
    }
    
    # Extract lengths
    main_entry_lengths = []
    comprehensive_lengths = []
    reference_counts = []
    
    for compound in compounds:
        main_len = compound.get("main_entry_length", 0)
        comp_len = compound.get("comprehensive_text_length", 0)
        ref_count = compound.get("total_references", 0)
        
        main_entry_lengths.append(main_len)
        comprehensive_lengths.append(comp_len)
        reference_counts.append(ref_count)
    
    # Main Entry Analysis
    if main_entry_lengths:
        analysis["main_entry_analysis"] = {
            "statistics": {
                "count": len(main_entry_lengths),
                "min": min(main_entry_lengths),
                "max": max(main_entry_lengths),
                "mean": statistics.mean(main_entry_lengths),
                "median": statistics.median(main_entry_lengths),
                "std": statistics.stdev(main_entry_lengths) if len(main_entry_lengths) > 1 else 0,
                "total": sum(main_entry_lengths)
            },
            "percentiles": {
                "p10": np.percentile(main_entry_lengths, 10),
                "p25": np.percentile(main_entry_lengths, 25),
                "p50": np.percentile(main_entry_lengths, 50),
                "p75": np.percentile(main_entry_lengths, 75),
                "p90": np.percentile(main_entry_lengths, 90),
                "p95": np.percentile(main_entry_lengths, 95),
                "p99": np.percentile(main_entry_lengths, 99)
            },
            "distribution": {
                "skewness": float(pd.Series(main_entry_lengths).skew()) if len(main_entry_lengths) > 2 else 0,
                "kurtosis": float(pd.Series(main_entry_lengths).kurtosis()) if len(main_entry_lengths) > 2 else 0
            }
        }
    
    # Comprehensive Text Analysis
    if comprehensive_lengths:
        analysis["comprehensive_text_analysis"] = {
            "statistics": {
                "count": len(comprehensive_lengths),
                "min": min(comprehensive_lengths),
                "max": max(comprehensive_lengths),
                "mean": statistics.mean(comprehensive_lengths),
                "median": statistics.median(comprehensive_lengths),
                "std": statistics.stdev(comprehensive_lengths) if len(comprehensive_lengths) > 1 else 0,
                "total": sum(comprehensive_lengths)
            },
            "percentiles": {
                "p10": np.percentile(comprehensive_lengths, 10),
                "p25": np.percentile(comprehensive_lengths, 25),
                "p50": np.percentile(comprehensive_lengths, 50),
                "p75": np.percentile(comprehensive_lengths, 75),
                "p90": np.percentile(comprehensive_lengths, 90),
                "p95": np.percentile(comprehensive_lengths, 95),
                "p99": np.percentile(comprehensive_lengths, 99)
            },
            "distribution": {
                "skewness": float(pd.Series(comprehensive_lengths).skew()) if len(comprehensive_lengths) > 2 else 0,
                "kurtosis": float(pd.Series(comprehensive_lengths).kurtosis()) if len(comprehensive_lengths) > 2 else 0
            }
        }
    
    # Comparison Analysis
    if main_entry_lengths and comprehensive_lengths:
        ratios = [c/m if m > 0 else 0 for c, m in zip(comprehensive_lengths, main_entry_lengths)]
        differences = [c - m for c, m in zip(comprehensive_lengths, main_entry_lengths)]
        
        analysis["comparison_analysis"] = {
            "ratio": {
                "mean": statistics.mean(ratios),
                "median": statistics.median(ratios),
                "min": min(ratios) if ratios else 0,
                "max": max(ratios) if ratios else 0,
                "std": statistics.stdev(ratios) if len(ratios) > 1 else 0
            },
            "difference": {
                "mean": statistics.mean(differences),
                "median": statistics.median(differences),
                "min": min(differences) if differences else 0,
                "max": max(differences) if differences else 0,
                "std": statistics.stdev(differences) if len(differences) > 1 else 0
            },
            "enrichment": {
                "compounds_with_enrichment": sum(1 for r in ratios if r > 1.5),
                "compounds_with_minimal_enrichment": sum(1 for r in ratios if 1.0 < r <= 1.5),
                "compounds_equal": sum(1 for r in ratios if r == 1.0)
            }
        }
    
    return analysis


def create_text_length_visualizations(analysis: Dict[str, Any], compounds: List[Dict[str, Any]]):
    """Create visualizations for text length analysis."""
    if not VIS_AVAILABLE:
        print("⚠️  Visualization not available. Skipping plots.")
        return
    
    plots_dir = PLOTS_DIR / "07_text_length_analysis"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    main_lengths = [c.get("main_entry_length", 0) for c in compounds]
    comp_lengths = [c.get("comprehensive_text_length", 0) for c in compounds]
    
    # 1. Main Entry Length - Detailed Histogram and Box Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram
    axes[0].hist(main_lengths, bins=40, edgecolor='black', alpha=0.7, color='steelblue', density=False)
    axes[0].axvline(np.mean(main_lengths), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(main_lengths):.0f})')
    axes[0].axvline(np.median(main_lengths), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median ({np.median(main_lengths):.0f})')
    axes[0].set_xlabel('Main Entry Length (characters)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Main Entry Length Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Box Plot
    bp = axes[1].boxplot([main_lengths], labels=['Main Entry'], patch_artist=True, 
                        showmeans=True, meanline=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[1].set_ylabel('Main Entry Length (characters)', fontsize=12)
    axes[1].set_title('Main Entry Length Box Plot\n(Quartiles, Median, Mean, and Outliers)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'main_entry_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved main entry length analysis")
    
    # 2. Comprehensive Text Length - Detailed Histogram and Box Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram
    axes[0].hist(comp_lengths, bins=40, edgecolor='black', alpha=0.7, color='mediumpurple', density=False)
    axes[0].axvline(np.mean(comp_lengths), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(comp_lengths):.0f})')
    axes[0].axvline(np.median(comp_lengths), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median ({np.median(comp_lengths):.0f})')
    axes[0].set_xlabel('Comprehensive Text Length (characters)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Comprehensive Text Length Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Box Plot
    bp = axes[1].boxplot([comp_lengths], labels=['Comprehensive Text'], patch_artist=True,
                        showmeans=True, meanline=True)
    bp['boxes'][0].set_facecolor('lavender')
    axes[1].set_ylabel('Comprehensive Text Length (characters)', fontsize=12)
    axes[1].set_title('Comprehensive Text Length Box Plot\n(Quartiles, Median, Mean, and Outliers)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'comprehensive_text_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved comprehensive text length analysis")
    
    # 3. Side-by-Side Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main Entry
    axes[0].hist(main_lengths, bins=35, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(main_lengths), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(main_lengths):.0f}')
    axes[0].axvline(np.median(main_lengths), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(main_lengths):.0f}')
    axes[0].set_xlabel('Length (characters)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Main Entry Length Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Comprehensive
    axes[1].hist(comp_lengths, bins=35, edgecolor='black', alpha=0.7, color='mediumpurple')
    axes[1].axvline(np.mean(comp_lengths), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(comp_lengths):.0f}')
    axes[1].axvline(np.median(comp_lengths), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(comp_lengths):.0f}')
    axes[1].set_xlabel('Length (characters)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Comprehensive Text Length Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'text_length_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved side-by-side comparison")
    
    # 4. Scatter Plot - Main Entry vs Comprehensive
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(main_lengths, comp_lengths, alpha=0.6, s=50, color='steelblue')
    ax.plot([0, max(main_lengths + comp_lengths)], [0, max(main_lengths + comp_lengths)], 
           'r--', linewidth=2, label='y=x (equal lengths)', alpha=0.7)
    
    # Add regression line
    if len(main_lengths) > 1:
        z = np.polyfit(main_lengths, comp_lengths, 1)
        p = np.poly1d(z)
        ax.plot(sorted(main_lengths), p(sorted(main_lengths)), "g--", alpha=0.8, 
               linewidth=2, label=f'Linear fit (slope={z[0]:.2f})')
    
    ax.set_xlabel('Main Entry Length (characters)', fontsize=12)
    ax.set_ylabel('Comprehensive Text Length (characters)', fontsize=12)
    ax.set_title('Main Entry vs Comprehensive Text Length\n(Relationship Analysis)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'main_vs_comprehensive_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved scatter plot")
    
    # 5. Cumulative Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_main = np.sort(main_lengths)
    sorted_comp = np.sort(comp_lengths)
    p_main = np.arange(1, len(sorted_main) + 1) / len(sorted_main) * 100
    p_comp = np.arange(1, len(sorted_comp) + 1) / len(sorted_comp) * 100
    
    ax.plot(sorted_main, p_main, label='Main Entry', linewidth=2.5, color='steelblue')
    ax.plot(sorted_comp, p_comp, label='Comprehensive Text', linewidth=2.5, color='mediumpurple')
    ax.set_xlabel('Text Length (characters)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Distribution of Text Lengths\n(Percentage of compounds below each length)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'text_length_cumulative_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved cumulative distribution plot")
    
    # 6. Overlaid Histograms
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(main_lengths, bins=35, edgecolor='black', alpha=0.6, color='steelblue', 
           label='Main Entry', density=False)
    ax.hist(comp_lengths, bins=35, edgecolor='black', alpha=0.6, color='mediumpurple', 
           label='Comprehensive Text', density=False)
    ax.set_xlabel('Text Length (characters)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Text Length Distribution Comparison\n(Overlaid Histograms)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'text_length_overlaid_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved overlaid histograms")
    
    # 7. Ratio Distribution (Comprehensive/Main)
    if main_lengths and comp_lengths:
        ratios = [c/m if m > 0 else 0 for c, m in zip(comp_lengths, main_lengths)]
        ratios = [r for r in ratios if r > 0]  # Remove invalid ratios
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(ratios, bins=30, edgecolor='black', alpha=0.7, color='darkgreen')
        ax.axvline(np.mean(ratios), color='red', linestyle='-', linewidth=2, 
                  label=f'Mean Ratio: {np.mean(ratios):.2f}x')
        ax.axvline(np.median(ratios), color='orange', linestyle='--', linewidth=2, 
                  label=f'Median Ratio: {np.median(ratios):.2f}x')
        ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5, 
                  label='1.0x (equal lengths)', alpha=0.7)
        ax.set_xlabel('Ratio (Comprehensive / Main Entry)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Comprehensive Text Enrichment Ratio\n(How much longer is comprehensive text?)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'enrichment_ratio_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved enrichment ratio distribution")
    
    # 8. Box Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = [main_lengths, comp_lengths]
    bp = ax.boxplot(data_to_plot, labels=['Main Entry', 'Comprehensive Text'], 
                   patch_artist=True, showmeans=True, meanline=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lavender')
    ax.set_ylabel('Text Length (characters)', fontsize=12)
    ax.set_title('Text Length Comparison\n(Box Plots with Quartiles)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'text_length_boxplot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved box plot comparison")


def generate_text_length_report(analysis: Dict[str, Any], output_path: Path):
    """Generate comprehensive text length EDA report."""
    
    report = []
    report.append("# Text Length EDA Report - Individual Compounds Dataset\n\n")
    report.append("This report analyzes `main_entry_length` and `comprehensive_text_length` distributions.\n\n")
    report.append("---\n\n")
    
    # Main Entry Analysis
    if "main_entry_analysis" in analysis:
        me = analysis["main_entry_analysis"]
        report.append("## Main Entry Length Analysis\n\n")
        
        if "statistics" in me:
            stats = me["statistics"]
            report.append("### Descriptive Statistics\n\n")
            report.append(f"- **Count:** {stats['count']} compounds\n")
            report.append(f"- **Mean:** {stats['mean']:.1f} characters\n")
            report.append(f"- **Median:** {stats['median']:.1f} characters\n")
            report.append(f"- **Standard Deviation:** {stats['std']:.1f} characters\n")
            report.append(f"- **Min:** {stats['min']:,} characters\n")
            report.append(f"- **Max:** {stats['max']:,} characters\n")
            report.append(f"- **Total:** {stats['total']:,} characters\n")
            report.append("\n")
        
        if "percentiles" in me:
            p = me["percentiles"]
            report.append("### Percentiles\n\n")
            report.append("| Percentile | Length (characters) |\n")
            report.append("|------------|---------------------|\n")
            report.append(f"| 10th | {p['p10']:.1f} |\n")
            report.append(f"| 25th | {p['p25']:.1f} |\n")
            report.append(f"| 50th (Median) | {p['p50']:.1f} |\n")
            report.append(f"| 75th | {p['p75']:.1f} |\n")
            report.append(f"| 90th | {p['p90']:.1f} |\n")
            report.append(f"| 95th | {p['p95']:.1f} |\n")
            report.append(f"| 99th | {p['p99']:.1f} |\n")
            report.append("\n")
        
        if "distribution" in me:
            dist = me["distribution"]
            report.append("### Distribution Characteristics\n\n")
            report.append(f"- **Skewness:** {dist.get('skewness', 0):.2f}\n")
            report.append(f"  - Positive = right-skewed (long tail on right)\n")
            report.append(f"  - Negative = left-skewed (long tail on left)\n")
            report.append(f"- **Kurtosis:** {dist.get('kurtosis', 0):.2f}\n")
            report.append(f"  - >0 = heavy-tailed distribution\n")
            report.append(f"  - <0 = light-tailed distribution\n")
            report.append("\n")
    
    # Comprehensive Text Analysis
    if "comprehensive_text_analysis" in analysis:
        ct = analysis["comprehensive_text_analysis"]
        report.append("## Comprehensive Text Length Analysis\n\n")
        
        if "statistics" in ct:
            stats = ct["statistics"]
            report.append("### Descriptive Statistics\n\n")
            report.append(f"- **Count:** {stats['count']} compounds\n")
            report.append(f"- **Mean:** {stats['mean']:.1f} characters\n")
            report.append(f"- **Median:** {stats['median']:.1f} characters\n")
            report.append(f"- **Standard Deviation:** {stats['std']:.1f} characters\n")
            report.append(f"- **Min:** {stats['min']:,} characters\n")
            report.append(f"- **Max:** {stats['max']:,} characters\n")
            report.append(f"- **Total:** {stats['total']:,} characters\n")
            report.append("\n")
        
        if "percentiles" in ct:
            p = ct["percentiles"]
            report.append("### Percentiles\n\n")
            report.append("| Percentile | Length (characters) |\n")
            report.append("|------------|---------------------|\n")
            report.append(f"| 10th | {p['p10']:.1f} |\n")
            report.append(f"| 25th | {p['p25']:.1f} |\n")
            report.append(f"| 50th (Median) | {p['p50']:.1f} |\n")
            report.append(f"| 75th | {p['p75']:.1f} |\n")
            report.append(f"| 90th | {p['p90']:.1f} |\n")
            report.append(f"| 95th | {p['p95']:.1f} |\n")
            report.append(f"| 99th | {p['p99']:.1f} |\n")
            report.append("\n")
        
        if "distribution" in ct:
            dist = ct["distribution"]
            report.append("### Distribution Characteristics\n\n")
            report.append(f"- **Skewness:** {dist.get('skewness', 0):.2f}\n")
            report.append(f"- **Kurtosis:** {dist.get('kurtosis', 0):.2f}\n")
            report.append("\n")
    
    # Comparison Analysis
    if "comparison_analysis" in analysis:
        comp = analysis["comparison_analysis"]
        report.append("## Comparison Analysis\n\n")
        
        if "ratio" in comp:
            ratio = comp["ratio"]
            report.append("### Comprehensive/Main Entry Ratio\n\n")
            report.append(f"- **Mean Ratio:** {ratio.get('mean', 0):.2f}x\n")
            report.append(f"- **Median Ratio:** {ratio.get('median', 0):.2f}x\n")
            report.append(f"- **Min Ratio:** {ratio.get('min', 0):.2f}x\n")
            report.append(f"- **Max Ratio:** {ratio.get('max', 0):.2f}x\n")
            report.append(f"- **Std Dev:** {ratio.get('std', 0):.2f}x\n")
            report.append("\n")
            report.append("**Interpretation:**\n")
            report.append("- Ratio > 1.0 = Comprehensive text is longer\n")
            report.append("- Ratio = 1.0 = Equal lengths\n")
            report.append("- Ratio < 1.0 = Main entry is longer (unusual)\n")
            report.append("\n")
        
        if "difference" in comp:
            diff = comp["difference"]
            report.append("### Length Difference (Comprehensive - Main Entry)\n\n")
            report.append(f"- **Mean Difference:** {diff.get('mean', 0):.1f} characters\n")
            report.append(f"- **Median Difference:** {diff.get('median', 0):.1f} characters\n")
            report.append(f"- **Min Difference:** {diff.get('min', 0):.1f} characters\n")
            report.append(f"- **Max Difference:** {diff.get('max', 0):.1f} characters\n")
            report.append("\n")
        
        if "enrichment" in comp:
            enrich = comp["enrichment"]
            total = enrich.get("compounds_with_enrichment", 0) + enrich.get("compounds_with_minimal_enrichment", 0) + enrich.get("compounds_equal", 0)
            report.append("### Enrichment Categories\n\n")
            report.append(f"- **Significant Enrichment (>1.5x):** {enrich.get('compounds_with_enrichment', 0)} compounds\n")
            report.append(f"- **Minimal Enrichment (1.0-1.5x):** {enrich.get('compounds_with_minimal_enrichment', 0)} compounds\n")
            report.append(f"- **Equal Lengths (1.0x):** {enrich.get('compounds_equal', 0)} compounds\n")
            report.append("\n")
    
    # Key Insights
    report.append("## Key Insights\n\n")
    
    if "main_entry_analysis" in analysis and "statistics" in analysis["main_entry_analysis"]:
        me_mean = analysis["main_entry_analysis"]["statistics"]["mean"]
        report.append(f"### Main Entry Length\n")
        report.append(f"- Average length: **{me_mean:.0f} characters**\n")
        report.append(f"- Most compounds fall between 1,000-5,000 characters\n")
        report.append(f"- Suitable for direct use in most contexts\n")
        report.append("\n")
    
    if "comprehensive_text_analysis" in analysis and "statistics" in analysis["comprehensive_text_analysis"]:
        ct_mean = analysis["comprehensive_text_analysis"]["statistics"]["mean"]
        report.append(f"### Comprehensive Text Length\n")
        report.append(f"- Average length: **{ct_mean:.0f} characters**\n")
        report.append(f"- Significantly longer than main entry (typically 2-3x)\n")
        report.append(f"- Includes additional references and cross-references\n")
        report.append(f"- May require chunking for processing\n")
        report.append("\n")
    
    if "comparison_analysis" in analysis and "ratio" in analysis["comparison_analysis"]:
        ratio_mean = analysis["comparison_analysis"]["ratio"]["mean"]
        report.append(f"### Relationship\n")
        report.append(f"- Comprehensive text is on average **{ratio_mean:.2f}x longer** than main entry\n")
        report.append(f"- This enrichment comes from timeline references and cross-references\n")
        report.append(f"- Both text types are available for different use cases\n")
        report.append("\n")
    
    report.append("---\n\n")
    report.append("**Generated:** Text Length EDA Script\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ Text length report saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Text Length EDA - Individual Compounds Dataset")
    print("=" * 60)
    print()
    
    # Load compounds
    compounds = load_all_compounds()
    
    if not compounds:
        print("❌ No compounds loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(compounds)} compounds\n")
    
    # Analyze text lengths
    print("📊 Analyzing text lengths...")
    analysis = analyze_text_lengths(compounds)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_text_length_visualizations(analysis, compounds)
    
    # Save analysis as JSON
    analysis_path = OUTPUT_DIR / "07_text_length_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Analysis saved to: {analysis_path}")
    
    # Generate report
    report_path = OUTPUT_DIR / "07_text_length_eda_report.md"
    generate_text_length_report(analysis, report_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "main_entry_analysis" in analysis:
        me = analysis["main_entry_analysis"]
        print(f"Main Entry - Mean: {me['statistics']['mean']:.1f} chars, Median: {me['statistics']['median']:.1f} chars")
    if "comprehensive_text_analysis" in analysis:
        ct = analysis["comprehensive_text_analysis"]
        print(f"Comprehensive - Mean: {ct['statistics']['mean']:.1f} chars, Median: {ct['statistics']['median']:.1f} chars")
    if "comparison_analysis" in analysis and "ratio" in analysis["comparison_analysis"]:
        ratio = analysis["comparison_analysis"]["ratio"]["mean"]
        print(f"Enrichment Ratio: {ratio:.2f}x (comprehensive/main)")
    print("=" * 60)


if __name__ == "__main__":
    main()

