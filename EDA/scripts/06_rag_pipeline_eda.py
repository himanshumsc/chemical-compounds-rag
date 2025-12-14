#!/usr/bin/env python3
"""
RAG Pipeline EDA - Individual Compounds Dataset
Analyzes data completeness and suitability for multimodal RAG pipeline.
Focuses on text lengths, context windows, and RAG-relevant metrics.
"""

import json
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
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

# RAG Pipeline Constants (from codebase analysis)
CHAR_LIMIT_Q1 = 600   # Q1 character limit
CHAR_LIMIT_Q2 = 1000  # Q2 character limit
CHAR_LIMIT_Q3 = 1800  # Q3 character limit
CHAR_LIMIT_Q4 = 2000  # Q4 character limit

MAX_TOKENS_Q1 = 200
MAX_TOKENS_Q2 = 333
MAX_TOKENS_Q3 = 600
MAX_TOKENS_Q4 = 666

# Typical token-to-character ratios (approximate)
TOKEN_CHAR_RATIO = 4  # ~4 characters per token (conservative estimate)


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


def analyze_text_lengths_for_rag(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze text lengths in context of RAG pipeline requirements."""
    
    analysis = {
        "main_entry_analysis": {},
        "comprehensive_text_analysis": {},
        "rag_suitability": {},
        "context_window_analysis": {},
        "chunking_analysis": {}
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
                "p25": np.percentile(main_entry_lengths, 25),
                "p50": np.percentile(main_entry_lengths, 50),
                "p75": np.percentile(main_entry_lengths, 75),
                "p90": np.percentile(main_entry_lengths, 90),
                "p95": np.percentile(main_entry_lengths, 95),
                "p99": np.percentile(main_entry_lengths, 99)
            },
            "rag_compatibility": {
                "fits_q1": sum(1 for x in main_entry_lengths if x <= CHAR_LIMIT_Q1),
                "fits_q2": sum(1 for x in main_entry_lengths if x <= CHAR_LIMIT_Q2),
                "fits_q3": sum(1 for x in main_entry_lengths if x <= CHAR_LIMIT_Q3),
                "fits_q4": sum(1 for x in main_entry_lengths if x <= CHAR_LIMIT_Q4),
                "exceeds_q4": sum(1 for x in main_entry_lengths if x > CHAR_LIMIT_Q4)
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
                "p25": np.percentile(comprehensive_lengths, 25),
                "p50": np.percentile(comprehensive_lengths, 50),
                "p75": np.percentile(comprehensive_lengths, 75),
                "p90": np.percentile(comprehensive_lengths, 90),
                "p95": np.percentile(comprehensive_lengths, 95),
                "p99": np.percentile(comprehensive_lengths, 99)
            },
            "rag_compatibility": {
                "fits_q1": sum(1 for x in comprehensive_lengths if x <= CHAR_LIMIT_Q1),
                "fits_q2": sum(1 for x in comprehensive_lengths if x <= CHAR_LIMIT_Q2),
                "fits_q3": sum(1 for x in comprehensive_lengths if x <= CHAR_LIMIT_Q3),
                "fits_q4": sum(1 for x in comprehensive_lengths if x <= CHAR_LIMIT_Q4),
                "exceeds_q4": sum(1 for x in comprehensive_lengths if x > CHAR_LIMIT_Q4)
            }
        }
    
    # RAG Suitability Analysis
    analysis["rag_suitability"] = {
        "main_entry_vs_comprehensive": {
            "ratio_mean": statistics.mean([c/m if m > 0 else 0 for c, m in zip(comprehensive_lengths, main_entry_lengths)]),
            "ratio_median": statistics.median([c/m if m > 0 else 0 for c, m in zip(comprehensive_lengths, main_entry_lengths)]),
            "comprehensive_advantage": sum(1 for c, m in zip(comprehensive_lengths, main_entry_lengths) if c > m * 1.5)
        },
        "reference_enrichment": {
            "mean_references": statistics.mean(reference_counts) if reference_counts else 0,
            "compounds_with_refs": sum(1 for r in reference_counts if r > 0),
            "compounds_without_refs": sum(1 for r in reference_counts if r == 0),
            "high_reference_compounds": sum(1 for r in reference_counts if r > 20)
        }
    }
    
    # Context Window Analysis (token-based)
    analysis["context_window_analysis"] = {
        "main_entry_tokens": {
            "mean": statistics.mean([x / TOKEN_CHAR_RATIO for x in main_entry_lengths]) if main_entry_lengths else 0,
            "median": statistics.median([x / TOKEN_CHAR_RATIO for x in main_entry_lengths]) if main_entry_lengths else 0,
            "max": max([x / TOKEN_CHAR_RATIO for x in main_entry_lengths]) if main_entry_lengths else 0,
            "fits_q1_tokens": sum(1 for x in main_entry_lengths if (x / TOKEN_CHAR_RATIO) <= MAX_TOKENS_Q1),
            "fits_q2_tokens": sum(1 for x in main_entry_lengths if (x / TOKEN_CHAR_RATIO) <= MAX_TOKENS_Q2),
            "fits_q3_tokens": sum(1 for x in main_entry_lengths if (x / TOKEN_CHAR_RATIO) <= MAX_TOKENS_Q3),
            "fits_q4_tokens": sum(1 for x in main_entry_lengths if (x / TOKEN_CHAR_RATIO) <= MAX_TOKENS_Q4)
        },
        "comprehensive_tokens": {
            "mean": statistics.mean([x / TOKEN_CHAR_RATIO for x in comprehensive_lengths]) if comprehensive_lengths else 0,
            "median": statistics.median([x / TOKEN_CHAR_RATIO for x in comprehensive_lengths]) if comprehensive_lengths else 0,
            "max": max([x / TOKEN_CHAR_RATIO for x in comprehensive_lengths]) if comprehensive_lengths else 0
        }
    }
    
    # Chunking Analysis (for RAG retrieval)
    # Typical chunk size: 500-1000 characters
    CHUNK_SIZE = 800  # Average chunk size
    analysis["chunking_analysis"] = {
        "main_entry_chunks": {
            "mean_chunks": statistics.mean([max(1, x / CHUNK_SIZE) for x in main_entry_lengths]) if main_entry_lengths else 0,
            "median_chunks": statistics.median([max(1, x / CHUNK_SIZE) for x in main_entry_lengths]) if main_entry_lengths else 0,
            "max_chunks": max([max(1, int(x / CHUNK_SIZE)) for x in main_entry_lengths]) if main_entry_lengths else 0,
            "single_chunk": sum(1 for x in main_entry_lengths if x <= CHUNK_SIZE),
            "multi_chunk": sum(1 for x in main_entry_lengths if x > CHUNK_SIZE)
        },
        "comprehensive_chunks": {
            "mean_chunks": statistics.mean([max(1, x / CHUNK_SIZE) for x in comprehensive_lengths]) if comprehensive_lengths else 0,
            "median_chunks": statistics.median([max(1, x / CHUNK_SIZE) for x in comprehensive_lengths]) if comprehensive_lengths else 0,
            "max_chunks": max([max(1, int(x / CHUNK_SIZE)) for x in comprehensive_lengths]) if comprehensive_lengths else 0,
            "single_chunk": sum(1 for x in comprehensive_lengths if x <= CHUNK_SIZE),
            "multi_chunk": sum(1 for x in comprehensive_lengths if x > CHUNK_SIZE)
        }
    }
    
    return analysis


def analyze_content_structure_for_rag(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze content structure relevant to RAG retrieval."""
    
    analysis = {
        "section_presence": {},
        "reference_quality": {},
        "content_completeness": {}
    }
    
    # Section presence
    sections_found = Counter()
    compounds_with_sections = 0
    
    for compound in compounds:
        main_entry = compound.get("main_entry_content", "")
        has_sections = False
        
        if "FORMULA:" in main_entry:
            sections_found["FORMULA"] += 1
            has_sections = True
        if "ELEMENTS:" in main_entry:
            sections_found["ELEMENTS"] += 1
            has_sections = True
        if "COMPOUND TYPE:" in main_entry:
            sections_found["COMPOUND_TYPE"] += 1
            has_sections = True
        if "STATE:" in main_entry:
            sections_found["STATE"] += 1
            has_sections = True
        if "MOLECULAR WEIGHT:" in main_entry:
            sections_found["MOLECULAR_WEIGHT"] += 1
            has_sections = True
        if "MELTING POINT:" in main_entry:
            sections_found["MELTING_POINT"] += 1
            has_sections = True
        if "BOILING POINT:" in main_entry:
            sections_found["BOILING_POINT"] += 1
            has_sections = True
        if "SOLUBILITY:" in main_entry:
            sections_found["SOLUBILITY"] += 1
            has_sections = True
        if "OVERVIEW" in main_entry:
            sections_found["OVERVIEW"] += 1
            has_sections = True
        if "HOW IT IS MADE" in main_entry:
            sections_found["HOW_IT_IS_MADE"] += 1
            has_sections = True
        if "COMMON USES" in main_entry:
            sections_found["COMMON_USES"] += 1
            has_sections = True
        if "POTENTIAL HAZARDS" in main_entry or "HAZARDS" in main_entry:
            sections_found["HAZARDS"] += 1
            has_sections = True
        
        if has_sections:
            compounds_with_sections += 1
    
    analysis["section_presence"] = {
        "total_compounds": len(compounds),
        "compounds_with_sections": compounds_with_sections,
        "section_frequency": dict(sections_found),
        "section_coverage": {k: (v / len(compounds)) * 100 for k, v in sections_found.items()}
    }
    
    # Reference quality
    ref_types = Counter()
    total_refs = []
    
    for compound in compounds:
        ref_types_list = compound.get("reference_types_found", [])
        for ref_type in ref_types_list:
            ref_types[ref_type] += 1
        total_refs.append(compound.get("total_references", 0))
    
    analysis["reference_quality"] = {
        "reference_types": dict(ref_types),
        "mean_references": statistics.mean(total_refs) if total_refs else 0,
        "compounds_with_timeline_refs": ref_types.get("timeline", 0)
    }
    
    # Content completeness
    completeness_scores = []
    for compound in compounds:
        score = 0
        main_entry = compound.get("main_entry_content", "")
        
        # Check for key sections
        if "FORMULA:" in main_entry:
            score += 1
        if "ELEMENTS:" in main_entry:
            score += 1
        if "COMPOUND TYPE:" in main_entry:
            score += 1
        if "STATE:" in main_entry:
            score += 1
        if "MOLECULAR WEIGHT:" in main_entry:
            score += 1
        if "OVERVIEW" in main_entry:
            score += 1
        if "HOW IT IS MADE" in main_entry:
            score += 1
        if "COMMON USES" in main_entry:
            score += 1
        
        completeness_scores.append(score)
    
    analysis["content_completeness"] = {
        "mean_score": statistics.mean(completeness_scores) if completeness_scores else 0,
        "max_score": max(completeness_scores) if completeness_scores else 0,
        "high_completeness": sum(1 for s in completeness_scores if s >= 7),
        "low_completeness": sum(1 for s in completeness_scores if s < 4)
    }
    
    return analysis


def create_rag_visualizations(analysis: Dict[str, Any], compounds: List[Dict[str, Any]]):
    """Create visualizations for RAG pipeline analysis."""
    if not VIS_AVAILABLE:
        print("⚠️  Visualization not available. Skipping plots.")
        return
    
    rag_plots_dir = PLOTS_DIR / "06_rag_analysis"
    rag_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    main_lengths = [c.get("main_entry_length", 0) for c in compounds]
    comp_lengths = [c.get("comprehensive_text_length", 0) for c in compounds]
    
    # 1. Text Length Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(main_lengths, bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0].axvline(CHAR_LIMIT_Q1, color='red', linestyle='--', label=f'Q1 Limit ({CHAR_LIMIT_Q1})')
    axes[0].axvline(CHAR_LIMIT_Q2, color='orange', linestyle='--', label=f'Q2 Limit ({CHAR_LIMIT_Q2})')
    axes[0].axvline(CHAR_LIMIT_Q4, color='green', linestyle='--', label=f'Q4 Limit ({CHAR_LIMIT_Q4})')
    axes[0].set_xlabel('Main Entry Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Main Entry Length Distribution\nwith RAG Character Limits')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(comp_lengths, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[1].axvline(CHAR_LIMIT_Q1, color='red', linestyle='--', label=f'Q1 Limit ({CHAR_LIMIT_Q1})')
    axes[1].axvline(CHAR_LIMIT_Q2, color='orange', linestyle='--', label=f'Q2 Limit ({CHAR_LIMIT_Q2})')
    axes[1].axvline(CHAR_LIMIT_Q4, color='green', linestyle='--', label=f'Q4 Limit ({CHAR_LIMIT_Q4})')
    axes[1].set_xlabel('Comprehensive Text Length (characters)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Comprehensive Text Length Distribution\nwith RAG Character Limits')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(rag_plots_dir / 'text_length_distributions_with_limits.png', dpi=300)
    plt.close()
    print(f"  ✅ Saved text length distributions")
    
    # 2. RAG Compatibility Chart
    if "main_entry_analysis" in analysis and "rag_compatibility" in analysis["main_entry_analysis"]:
        compat = analysis["main_entry_analysis"]["rag_compatibility"]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        questions = ['Q1 (600)', 'Q2 (1000)', 'Q3 (1800)', 'Q4 (2000)', 'Exceeds Q4']
        counts = [
            compat.get("fits_q1", 0),
            compat.get("fits_q2", 0),
            compat.get("fits_q3", 0),
            compat.get("fits_q4", 0),
            compat.get("exceeds_q4", 0)
        ]
        
        bars = ax.bar(questions, counts, color=['green', 'blue', 'orange', 'purple', 'red'])
        ax.set_ylabel('Number of Compounds')
        ax.set_title('Main Entry Length Compatibility with RAG Character Limits')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(rag_plots_dir / 'rag_compatibility_chart.png', dpi=300)
        plt.close()
        print(f"  ✅ Saved RAG compatibility chart")
    
    # 3. Main Entry vs Comprehensive Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(main_lengths, comp_lengths, alpha=0.6, s=50)
    ax.plot([0, max(main_lengths + comp_lengths)], [0, max(main_lengths + comp_lengths)], 
           'r--', label='y=x (equal lengths)')
    ax.set_xlabel('Main Entry Length (characters)')
    ax.set_ylabel('Comprehensive Text Length (characters)')
    ax.set_title('Main Entry vs Comprehensive Text Length\n(For RAG Context Selection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(rag_plots_dir / 'main_vs_comprehensive_scatter.png', dpi=300)
    plt.close()
    print(f"  ✅ Saved main vs comprehensive comparison")
    
    # 4. Boxplot by Reference Count
    ref_counts = [c.get("total_references", 0) for c in compounds]
    ref_bins = ['0', '1-5', '6-10', '11-20', '21+']
    ref_bin_values = []
    main_lengths_by_ref = {bin_name: [] for bin_name in ref_bins}
    
    for ref_count, main_len in zip(ref_counts, main_lengths):
        if ref_count == 0:
            main_lengths_by_ref['0'].append(main_len)
        elif 1 <= ref_count <= 5:
            main_lengths_by_ref['1-5'].append(main_len)
        elif 6 <= ref_count <= 10:
            main_lengths_by_ref['6-10'].append(main_len)
        elif 11 <= ref_count <= 20:
            main_lengths_by_ref['11-20'].append(main_len)
        else:
            main_lengths_by_ref['21+'].append(main_len)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [main_lengths_by_ref[bin_name] for bin_name in ref_bins if main_lengths_by_ref[bin_name]]
    labels_to_plot = [bin_name for bin_name in ref_bins if main_lengths_by_ref[bin_name]]
    
    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels_to_plot)
        ax.set_xlabel('Reference Count Range')
        ax.set_ylabel('Main Entry Length (characters)')
        ax.set_title('Main Entry Length Distribution by Reference Count\n(More references = richer context for RAG)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(rag_plots_dir / 'length_by_reference_count.png', dpi=300)
        plt.close()
        print(f"  ✅ Saved length by reference count")
    
    # 5. Detailed Histograms - Main Entry Length
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main Entry - Histogram with density
    axes[0].hist(main_lengths, bins=40, edgecolor='black', alpha=0.7, color='steelblue', density=False)
    axes[0].axvline(np.mean(main_lengths), color='red', linestyle='-', linewidth=2, label=f'Mean ({np.mean(main_lengths):.0f})')
    axes[0].axvline(np.median(main_lengths), color='orange', linestyle='--', linewidth=2, label=f'Median ({np.median(main_lengths):.0f})')
    axes[0].axvline(CHAR_LIMIT_Q1, color='green', linestyle=':', linewidth=1.5, label=f'Q1 Limit ({CHAR_LIMIT_Q1})')
    axes[0].axvline(CHAR_LIMIT_Q2, color='blue', linestyle=':', linewidth=1.5, label=f'Q2 Limit ({CHAR_LIMIT_Q2})')
    axes[0].axvline(CHAR_LIMIT_Q3, color='purple', linestyle=':', linewidth=1.5, label=f'Q3 Limit ({CHAR_LIMIT_Q3})')
    axes[0].axvline(CHAR_LIMIT_Q4, color='brown', linestyle=':', linewidth=1.5, label=f'Q4 Limit ({CHAR_LIMIT_Q4})')
    axes[0].set_xlabel('Main Entry Length (characters)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Main Entry Length Distribution\n(Detailed with RAG Character Limits)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Main Entry - Box Plot
    bp = axes[1].boxplot([main_lengths], labels=['Main Entry'], patch_artist=True, 
                        showmeans=True, meanline=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[1].axhline(CHAR_LIMIT_Q1, color='green', linestyle=':', linewidth=1.5, label=f'Q1 Limit ({CHAR_LIMIT_Q1})')
    axes[1].axhline(CHAR_LIMIT_Q2, color='blue', linestyle=':', linewidth=1.5, label=f'Q2 Limit ({CHAR_LIMIT_Q2})')
    axes[1].axhline(CHAR_LIMIT_Q3, color='purple', linestyle=':', linewidth=1.5, label=f'Q3 Limit ({CHAR_LIMIT_Q3})')
    axes[1].axhline(CHAR_LIMIT_Q4, color='brown', linestyle=':', linewidth=1.5, label=f'Q4 Limit ({CHAR_LIMIT_Q4})')
    axes[1].set_ylabel('Main Entry Length (characters)', fontsize=12)
    axes[1].set_title('Main Entry Length Box Plot\n(Showing Quartiles and Outliers)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(rag_plots_dir / 'main_entry_length_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved detailed main entry length plots")
    
    # 6. Detailed Histograms - Comprehensive Text Length
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Comprehensive - Histogram with density
    axes[0].hist(comp_lengths, bins=40, edgecolor='black', alpha=0.7, color='mediumpurple', density=False)
    axes[0].axvline(np.mean(comp_lengths), color='red', linestyle='-', linewidth=2, label=f'Mean ({np.mean(comp_lengths):.0f})')
    axes[0].axvline(np.median(comp_lengths), color='orange', linestyle='--', linewidth=2, label=f'Median ({np.median(comp_lengths):.0f})')
    axes[0].axvline(CHAR_LIMIT_Q1, color='green', linestyle=':', linewidth=1.5, label=f'Q1 Limit ({CHAR_LIMIT_Q1})')
    axes[0].axvline(CHAR_LIMIT_Q2, color='blue', linestyle=':', linewidth=1.5, label=f'Q2 Limit ({CHAR_LIMIT_Q2})')
    axes[0].axvline(CHAR_LIMIT_Q3, color='purple', linestyle=':', linewidth=1.5, label=f'Q3 Limit ({CHAR_LIMIT_Q3})')
    axes[0].axvline(CHAR_LIMIT_Q4, color='brown', linestyle=':', linewidth=1.5, label=f'Q4 Limit ({CHAR_LIMIT_Q4})')
    axes[0].set_xlabel('Comprehensive Text Length (characters)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Comprehensive Text Length Distribution\n(Detailed with RAG Character Limits)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Comprehensive - Box Plot
    bp = axes[1].boxplot([comp_lengths], labels=['Comprehensive Text'], patch_artist=True,
                        showmeans=True, meanline=True)
    bp['boxes'][0].set_facecolor('lavender')
    axes[1].axhline(CHAR_LIMIT_Q1, color='green', linestyle=':', linewidth=1.5, label=f'Q1 Limit ({CHAR_LIMIT_Q1})')
    axes[1].axhline(CHAR_LIMIT_Q2, color='blue', linestyle=':', linewidth=1.5, label=f'Q2 Limit ({CHAR_LIMIT_Q2})')
    axes[1].axhline(CHAR_LIMIT_Q3, color='purple', linestyle=':', linewidth=1.5, label=f'Q3 Limit ({CHAR_LIMIT_Q3})')
    axes[1].axhline(CHAR_LIMIT_Q4, color='brown', linestyle=':', linewidth=1.5, label=f'Q4 Limit ({CHAR_LIMIT_Q4})')
    axes[1].set_ylabel('Comprehensive Text Length (characters)', fontsize=12)
    axes[1].set_title('Comprehensive Text Length Box Plot\n(Showing Quartiles and Outliers)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(rag_plots_dir / 'comprehensive_text_length_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved detailed comprehensive text length plots")
    
    # 7. Side-by-Side Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main Entry
    axes[0].hist(main_lengths, bins=35, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(main_lengths), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(main_lengths):.0f}')
    axes[0].axvline(np.median(main_lengths), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(main_lengths):.0f}')
    axes[0].set_xlabel('Length (characters)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Main Entry Length Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Comprehensive
    axes[1].hist(comp_lengths, bins=35, edgecolor='black', alpha=0.7, color='mediumpurple')
    axes[1].axvline(np.mean(comp_lengths), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(comp_lengths):.0f}')
    axes[1].axvline(np.median(comp_lengths), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(comp_lengths):.0f}')
    axes[1].set_xlabel('Length (characters)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Comprehensive Text Length Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(rag_plots_dir / 'text_length_comparison_side_by_side.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved side-by-side comparison")
    
    # 8. Cumulative Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_main = np.sort(main_lengths)
    sorted_comp = np.sort(comp_lengths)
    p_main = np.arange(1, len(sorted_main) + 1) / len(sorted_main) * 100
    p_comp = np.arange(1, len(sorted_comp) + 1) / len(sorted_comp) * 100
    
    ax.plot(sorted_main, p_main, label='Main Entry', linewidth=2, color='steelblue')
    ax.plot(sorted_comp, p_comp, label='Comprehensive Text', linewidth=2, color='mediumpurple')
    ax.axvline(CHAR_LIMIT_Q4, color='red', linestyle='--', linewidth=1.5, label=f'Q4 Limit ({CHAR_LIMIT_Q4})')
    ax.set_xlabel('Text Length (characters)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Distribution of Text Lengths\n(What % of compounds are below each length)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(rag_plots_dir / 'text_length_cumulative_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved cumulative distribution plot")


def generate_rag_report(analysis: Dict[str, Any], structure_analysis: Dict[str, Any], output_path: Path):
    """Generate comprehensive RAG pipeline EDA report."""
    
    report = []
    report.append("# RAG Pipeline EDA Report - Individual Compounds Dataset\n\n")
    report.append("This report analyzes data completeness and suitability for the multimodal RAG pipeline.\n\n")
    report.append("---\n\n")
    
    # Main Entry Analysis
    if "main_entry_analysis" in analysis:
        me = analysis["main_entry_analysis"]
        report.append("## Main Entry Length Analysis\n\n")
        
        if "statistics" in me:
            stats = me["statistics"]
            report.append("### Statistics\n\n")
            report.append(f"- **Count:** {stats['count']} compounds\n")
            report.append(f"- **Mean:** {stats['mean']:.1f} characters\n")
            report.append(f"- **Median:** {stats['median']:.1f} characters\n")
            report.append(f"- **Min:** {stats['min']:,} characters\n")
            report.append(f"- **Max:** {stats['max']:,} characters\n")
            report.append(f"- **Std Dev:** {stats['std']:.1f} characters\n")
            report.append("\n")
        
        if "percentiles" in me:
            p = me["percentiles"]
            report.append("### Percentiles\n\n")
            report.append(f"- **25th:** {p['p25']:.1f} characters\n")
            report.append(f"- **50th (Median):** {p['p50']:.1f} characters\n")
            report.append(f"- **75th:** {p['p75']:.1f} characters\n")
            report.append(f"- **90th:** {p['p90']:.1f} characters\n")
            report.append(f"- **95th:** {p['p95']:.1f} characters\n")
            report.append(f"- **99th:** {p['p99']:.1f} characters\n")
            report.append("\n")
        
        if "rag_compatibility" in me:
            compat = me["rag_compatibility"]
            total = compat.get("fits_q4", 0) + compat.get("exceeds_q4", 0)
            report.append("### RAG Character Limit Compatibility\n\n")
            report.append("| Question | Character Limit | Compounds That Fit | Percentage |\n")
            report.append("|----------|----------------|-------------------|------------|\n")
            report.append(f"| Q1 | {CHAR_LIMIT_Q1} | {compat.get('fits_q1', 0)} | {(compat.get('fits_q1', 0)/total*100):.1f}% |\n")
            report.append(f"| Q2 | {CHAR_LIMIT_Q2} | {compat.get('fits_q2', 0)} | {(compat.get('fits_q2', 0)/total*100):.1f}% |\n")
            report.append(f"| Q3 | {CHAR_LIMIT_Q3} | {compat.get('fits_q3', 0)} | {(compat.get('fits_q3', 0)/total*100):.1f}% |\n")
            report.append(f"| Q4 | {CHAR_LIMIT_Q4} | {compat.get('fits_q4', 0)} | {(compat.get('fits_q4', 0)/total*100):.1f}% |\n")
            report.append(f"| Exceeds Q4 | >{CHAR_LIMIT_Q4} | {compat.get('exceeds_q4', 0)} | {(compat.get('exceeds_q4', 0)/total*100):.1f}% |\n")
            report.append("\n")
    
    # Comprehensive Text Analysis
    if "comprehensive_text_analysis" in analysis:
        ct = analysis["comprehensive_text_analysis"]
        report.append("## Comprehensive Text Length Analysis\n\n")
        
        if "statistics" in ct:
            stats = ct["statistics"]
            report.append("### Statistics\n\n")
            report.append(f"- **Count:** {stats['count']} compounds\n")
            report.append(f"- **Mean:** {stats['mean']:.1f} characters\n")
            report.append(f"- **Median:** {stats['median']:.1f} characters\n")
            report.append(f"- **Min:** {stats['min']:,} characters\n")
            report.append(f"- **Max:** {stats['max']:,} characters\n")
            report.append("\n")
        
        if "rag_compatibility" in ct:
            compat = ct["rag_compatibility"]
            total = compat.get("fits_q4", 0) + compat.get("exceeds_q4", 0)
            report.append("### RAG Character Limit Compatibility\n\n")
            report.append("| Question | Character Limit | Compounds That Fit | Percentage |\n")
            report.append("|----------|----------------|-------------------|------------|\n")
            report.append(f"| Q1 | {CHAR_LIMIT_Q1} | {compat.get('fits_q1', 0)} | {(compat.get('fits_q1', 0)/total*100):.1f}% |\n")
            report.append(f"| Q2 | {CHAR_LIMIT_Q2} | {compat.get('fits_q2', 0)} | {(compat.get('fits_q2', 0)/total*100):.1f}% |\n")
            report.append(f"| Q3 | {CHAR_LIMIT_Q3} | {compat.get('fits_q3', 0)} | {(compat.get('fits_q3', 0)/total*100):.1f}% |\n")
            report.append(f"| Q4 | {CHAR_LIMIT_Q4} | {compat.get('fits_q4', 0)} | {(compat.get('fits_q4', 0)/total*100):.1f}% |\n")
            report.append(f"| Exceeds Q4 | >{CHAR_LIMIT_Q4} | {compat.get('exceeds_q4', 0)} | {(compat.get('exceeds_q4', 0)/total*100):.1f}% |\n")
            report.append("\n")
    
    # RAG Suitability
    if "rag_suitability" in analysis:
        rs = analysis["rag_suitability"]
        report.append("## RAG Suitability Analysis\n\n")
        
        if "main_entry_vs_comprehensive" in rs:
            mevc = rs["main_entry_vs_comprehensive"]
            report.append("### Main Entry vs Comprehensive Text\n\n")
            report.append(f"- **Mean Ratio (comp/main):** {mevc.get('ratio_mean', 0):.2f}x\n")
            report.append(f"- **Median Ratio:** {mevc.get('ratio_median', 0):.2f}x\n")
            report.append(f"- **Compounds with >1.5x enrichment:** {mevc.get('comprehensive_advantage', 0)}\n")
            report.append("\n")
            report.append("**Insight:** Comprehensive text provides significantly more context for RAG retrieval.\n\n")
        
        if "reference_enrichment" in rs:
            re = rs["reference_enrichment"]
            report.append("### Reference Enrichment\n\n")
            report.append(f"- **Mean References per Compound:** {re.get('mean_references', 0):.1f}\n")
            report.append(f"- **Compounds with References:** {re.get('compounds_with_refs', 0)}\n")
            report.append(f"- **Compounds without References:** {re.get('compounds_without_refs', 0)}\n")
            report.append(f"- **High Reference Compounds (>20):** {re.get('high_reference_compounds', 0)}\n")
            report.append("\n")
    
    # Context Window Analysis
    if "context_window_analysis" in analysis:
        cwa = analysis["context_window_analysis"]
        report.append("## Context Window Analysis (Token-based)\n\n")
        
        if "main_entry_tokens" in cwa:
            met = cwa["main_entry_tokens"]
            report.append("### Main Entry (Estimated Tokens)\n\n")
            report.append(f"- **Mean:** {met.get('mean', 0):.1f} tokens\n")
            report.append(f"- **Median:** {met.get('median', 0):.1f} tokens\n")
            report.append(f"- **Max:** {met.get('max', 0):.1f} tokens\n")
            report.append("\n")
            report.append("### Token Limit Compatibility\n\n")
            report.append("| Question | Token Limit | Compounds That Fit |\n")
            report.append("|----------|-------------|-------------------|\n")
            report.append(f"| Q1 | {MAX_TOKENS_Q1} | {met.get('fits_q1_tokens', 0)} |\n")
            report.append(f"| Q2 | {MAX_TOKENS_Q2} | {met.get('fits_q2_tokens', 0)} |\n")
            report.append(f"| Q3 | {MAX_TOKENS_Q3} | {met.get('fits_q3_tokens', 0)} |\n")
            report.append(f"| Q4 | {MAX_TOKENS_Q4} | {met.get('fits_q4_tokens', 0)} |\n")
            report.append("\n")
    
    # Chunking Analysis
    if "chunking_analysis" in analysis:
        ca = analysis["chunking_analysis"]
        report.append("## Chunking Analysis (for RAG Retrieval)\n\n")
        report.append("Assuming average chunk size of 800 characters:\n\n")
        
        if "main_entry_chunks" in ca:
            mec = ca["main_entry_chunks"]
            report.append("### Main Entry Chunking\n\n")
            report.append(f"- **Mean Chunks per Compound:** {mec.get('mean_chunks', 0):.1f}\n")
            report.append(f"- **Median Chunks:** {mec.get('median_chunks', 0):.1f}\n")
            report.append(f"- **Max Chunks:** {mec.get('max_chunks', 0)}\n")
            report.append(f"- **Single Chunk Compounds:** {mec.get('single_chunk', 0)}\n")
            report.append(f"- **Multi-Chunk Compounds:** {mec.get('multi_chunk', 0)}\n")
            report.append("\n")
    
    # Content Structure
    if "section_presence" in structure_analysis:
        sp = structure_analysis["section_presence"]
        report.append("## Content Structure Analysis\n\n")
        report.append("### Section Presence (Important for RAG Retrieval)\n\n")
        report.append("| Section | Present | Coverage % |\n")
        report.append("|---------|---------|------------|\n")
        
        for section, count in sorted(sp.get("section_frequency", {}).items(), key=lambda x: x[1], reverse=True):
            coverage = sp.get("section_coverage", {}).get(section, 0)
            report.append(f"| {section} | {count} | {coverage:.1f}% |\n")
        
        report.append("\n")
    
    # Content Completeness
    if "content_completeness" in structure_analysis:
        cc = structure_analysis["content_completeness"]
        report.append("### Content Completeness Score\n\n")
        report.append("(Based on presence of key sections: FORMULA, ELEMENTS, TYPE, STATE, MW, OVERVIEW, HOW_IT_IS_MADE, COMMON_USES)\n\n")
        report.append(f"- **Mean Score:** {cc.get('mean_score', 0):.1f}/8\n")
        report.append(f"- **Max Score:** {cc.get('max_score', 0)}/8\n")
        report.append(f"- **High Completeness (≥7):** {cc.get('high_completeness', 0)} compounds\n")
        report.append(f"- **Low Completeness (<4):** {cc.get('low_completeness', 0)} compounds\n")
        report.append("\n")
    
    # Recommendations
    report.append("## Recommendations for RAG Pipeline\n\n")
    
    if "main_entry_analysis" in analysis and "rag_compatibility" in analysis["main_entry_analysis"]:
        compat = analysis["main_entry_analysis"]["rag_compatibility"]
        exceeds = compat.get("exceeds_q4", 0)
        if exceeds > 0:
            report.append(f"⚠️ **{exceeds} compounds exceed Q4 character limit** - Consider truncation or chunking strategies\n\n")
    
    if "chunking_analysis" in analysis and "main_entry_chunks" in analysis["chunking_analysis"]:
        mec = analysis["chunking_analysis"]["main_entry_chunks"]
        multi_chunk = mec.get("multi_chunk", 0)
        if multi_chunk > 0:
            report.append(f"📦 **{multi_chunk} compounds require multi-chunk retrieval** - Ensure RAG system handles chunk aggregation\n\n")
    
    if "rag_suitability" in analysis and "main_entry_vs_comprehensive" in analysis["rag_suitability"]:
        mevc = analysis["rag_suitability"]["main_entry_vs_comprehensive"]
        if mevc.get("ratio_mean", 0) > 1.5:
            report.append("✅ **Comprehensive text provides significantly more context** - Consider using comprehensive text for richer RAG retrieval\n\n")
    
    report.append("---\n\n")
    report.append("**Generated:** RAG Pipeline EDA Script\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✅ RAG pipeline report saved to: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("RAG Pipeline EDA - Individual Compounds Dataset")
    print("=" * 60)
    print()
    
    # Load compounds
    compounds = load_all_compounds()
    
    if not compounds:
        print("❌ No compounds loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(compounds)} compounds\n")
    
    # Analyze text lengths for RAG
    print("📊 Analyzing text lengths for RAG pipeline...")
    analysis = analyze_text_lengths_for_rag(compounds)
    
    # Analyze content structure
    print("📊 Analyzing content structure...")
    structure_analysis = analyze_content_structure_for_rag(compounds)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_rag_visualizations(analysis, compounds)
    
    # Save analysis as JSON
    analysis_path = OUTPUT_DIR / "06_rag_pipeline_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump({
            "text_analysis": analysis,
            "structure_analysis": structure_analysis
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Analysis saved to: {analysis_path}")
    
    # Generate report
    report_path = OUTPUT_DIR / "06_rag_pipeline_eda_report.md"
    generate_rag_report(analysis, structure_analysis, report_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "main_entry_analysis" in analysis:
        me = analysis["main_entry_analysis"]
        print(f"Main Entry - Mean: {me['statistics']['mean']:.1f} chars, Median: {me['statistics']['median']:.1f} chars")
        compat = me.get("rag_compatibility", {})
        print(f"RAG Compatibility - Q1: {compat.get('fits_q1', 0)}, Q4: {compat.get('fits_q4', 0)}, Exceeds: {compat.get('exceeds_q4', 0)}")
    if "comprehensive_text_analysis" in analysis:
        ct = analysis["comprehensive_text_analysis"]
        print(f"Comprehensive - Mean: {ct['statistics']['mean']:.1f} chars, Median: {ct['statistics']['median']:.1f} chars")
    print("=" * 60)


if __name__ == "__main__":
    main()

