#!/usr/bin/env python3
"""
Answer Length Analysis: QWEN RAG vs OpenAI Baseline
Detailed per-question breakdown of answer lengths (characters and tokens)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import statistics

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas/numpy not installed. Install with: pip install pandas numpy")
    sys.exit(1)

# Directories
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Load the detailed results
BLEU_CSV = RESULTS_DIR / "per_question_bleu_rag.csv"
BERTSCORE_CSV = RESULTS_DIR / "per_question_bertscore_rag.csv"

def load_data():
    """Load the evaluation results"""
    if BLEU_CSV.exists():
        df = pd.read_csv(BLEU_CSV)
        return df
    elif BERTSCORE_CSV.exists():
        df = pd.read_csv(BERTSCORE_CSV)
        return df
    else:
        print(f"Error: No results file found. Expected {BLEU_CSV} or {BERTSCORE_CSV}")
        sys.exit(1)


def calculate_length_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate detailed length statistics grouped by question type.
    
    Returns:
        Dictionary with per-question statistics
    """
    stats = {}
    
    # Group by question_index (0=Q1, 1=Q2, 2=Q3, 3=Q4)
    for q_idx in range(4):
        q_data = df[df['question_index'] == q_idx]
        
        if len(q_data) == 0:
            continue
        
        # Character lengths (convert to list to avoid numpy type issues)
        qwen_chars = q_data['qwen_rag_length'].values.tolist()
        openai_chars = q_data['openai_length'].values.tolist()
        
        # Token lengths (convert to list to avoid numpy type issues)
        qwen_tokens = q_data['qwen_rag_tokens'].values.tolist()
        openai_tokens = q_data['openai_tokens'].values.tolist()
        
        # Calculate ratios
        char_ratios = [q/o if o > 0 else 0 for q, o in zip(qwen_chars, openai_chars)]
        token_ratios = [q/o if o > 0 else 0 for q, o in zip(qwen_tokens, openai_tokens)]
        
        stats[f'Q{q_idx+1}'] = {
            'count': len(q_data),
            'qwen_rag': {
                'chars': {
                    'mean': float(statistics.mean(qwen_chars)),
                    'median': float(statistics.median(qwen_chars)),
                    'min': int(min(qwen_chars)),
                    'max': int(max(qwen_chars)),
                    'std': float(statistics.stdev(qwen_chars)) if len(qwen_chars) > 1 else 0.0,
                },
                'tokens': {
                    'mean': float(statistics.mean(qwen_tokens)),
                    'median': float(statistics.median(qwen_tokens)),
                    'min': int(min(qwen_tokens)),
                    'max': int(max(qwen_tokens)),
                    'std': float(statistics.stdev(qwen_tokens)) if len(qwen_tokens) > 1 else 0.0,
                }
            },
            'openai': {
                'chars': {
                    'mean': float(statistics.mean(openai_chars)),
                    'median': float(statistics.median(openai_chars)),
                    'min': int(min(openai_chars)),
                    'max': int(max(openai_chars)),
                    'std': float(statistics.stdev(openai_chars)) if len(openai_chars) > 1 else 0.0,
                },
                'tokens': {
                    'mean': float(statistics.mean(openai_tokens)),
                    'median': float(statistics.median(openai_tokens)),
                    'min': int(min(openai_tokens)),
                    'max': int(max(openai_tokens)),
                    'std': float(statistics.stdev(openai_tokens)) if len(openai_tokens) > 1 else 0.0,
                }
            },
            'ratios': {
                'chars': {
                    'mean': float(statistics.mean(char_ratios)),
                    'median': float(statistics.median(char_ratios)),
                    'min': float(min(char_ratios)),
                    'max': float(max(char_ratios)),
                },
                'tokens': {
                    'mean': float(statistics.mean(token_ratios)),
                    'median': float(statistics.median(token_ratios)),
                    'min': float(min(token_ratios)),
                    'max': float(max(token_ratios)),
                }
            }
        }
    
    # Overall statistics (all questions combined)
    qwen_chars_all = df['qwen_rag_length'].values.tolist()
    openai_chars_all = df['openai_length'].values.tolist()
    qwen_tokens_all = df['qwen_rag_tokens'].values.tolist()
    openai_tokens_all = df['openai_tokens'].values.tolist()
    
    char_ratios_all = [q/o if o > 0 else 0 for q, o in zip(qwen_chars_all, openai_chars_all)]
    token_ratios_all = [q/o if o > 0 else 0 for q, o in zip(qwen_tokens_all, openai_tokens_all)]
    
    stats['Overall'] = {
        'count': len(df),
        'qwen_rag': {
            'chars': {
                'mean': float(statistics.mean(qwen_chars_all)),
                'median': float(statistics.median(qwen_chars_all)),
                'min': int(min(qwen_chars_all)),
                'max': int(max(qwen_chars_all)),
                'std': float(statistics.stdev(qwen_chars_all)) if len(qwen_chars_all) > 1 else 0.0,
            },
            'tokens': {
                'mean': float(statistics.mean(qwen_tokens_all)),
                'median': float(statistics.median(qwen_tokens_all)),
                'min': int(min(qwen_tokens_all)),
                'max': int(max(qwen_tokens_all)),
                'std': float(statistics.stdev(qwen_tokens_all)) if len(qwen_tokens_all) > 1 else 0.0,
            }
        },
        'openai': {
            'chars': {
                'mean': float(statistics.mean(openai_chars_all)),
                'median': float(statistics.median(openai_chars_all)),
                'min': int(min(openai_chars_all)),
                'max': int(max(openai_chars_all)),
                'std': float(statistics.stdev(openai_chars_all)) if len(openai_chars_all) > 1 else 0.0,
            },
            'tokens': {
                'mean': float(statistics.mean(openai_tokens_all)),
                'median': float(statistics.median(openai_tokens_all)),
                'min': int(min(openai_tokens_all)),
                'max': int(max(openai_tokens_all)),
                'std': float(statistics.stdev(openai_tokens_all)) if len(openai_tokens_all) > 1 else 0.0,
            }
        },
        'ratios': {
            'chars': {
                'mean': float(statistics.mean(char_ratios_all)),
                'median': float(statistics.median(char_ratios_all)),
                'min': float(min(char_ratios_all)),
                'max': float(max(char_ratios_all)),
            },
            'tokens': {
                'mean': float(statistics.mean(token_ratios_all)),
                'median': float(statistics.median(token_ratios_all)),
                'min': float(min(token_ratios_all)),
                'max': float(max(token_ratios_all)),
            }
        }
    }
    
    return stats


def print_detailed_table(stats: Dict):
    """Print a detailed table of length statistics"""
    print("\n" + "="*100)
    print("DETAILED ANSWER LENGTH COMPARISON: QWEN RAG vs OpenAI Baseline")
    print("="*100)
    
    # Question type descriptions
    question_descriptions = {
        'Q1': 'Image-based identification',
        'Q2': 'Formula/Type',
        'Q3': 'Production process',
        'Q4': 'Uses/Hazards',
        'Overall': 'All questions combined'
    }
    
    for q_type in ['Q1', 'Q2', 'Q3', 'Q4', 'Overall']:
        if q_type not in stats:
            continue
        
        q_stats = stats[q_type]
        desc = question_descriptions.get(q_type, '')
        
        print(f"\n{'='*100}")
        print(f"{q_type}: {desc}")
        print(f"Count: {q_stats['count']} question-answer pairs")
        print(f"{'='*100}")
        
        # Character length comparison
        print("\n📏 CHARACTER LENGTH (chars):")
        print(f"{'Metric':<15} {'QWEN RAG':<20} {'OpenAI Baseline':<20} {'Ratio (Q/O)':<15}")
        print("-" * 70)
        
        qwen_c = q_stats['qwen_rag']['chars']
        openai_c = q_stats['openai']['chars']
        ratio_c = q_stats['ratios']['chars']
        
        print(f"{'Mean':<15} {qwen_c['mean']:<20.1f} {openai_c['mean']:<20.1f} {ratio_c['mean']:<15.2f}x")
        print(f"{'Median':<15} {qwen_c['median']:<20.1f} {openai_c['median']:<20.1f} {ratio_c['median']:<15.2f}x")
        print(f"{'Min':<15} {qwen_c['min']:<20} {openai_c['min']:<20} {ratio_c['min']:<15.2f}x")
        print(f"{'Max':<15} {qwen_c['max']:<20} {openai_c['max']:<20} {ratio_c['max']:<15.2f}x")
        print(f"{'Std Dev':<15} {qwen_c['std']:<20.1f} {openai_c['std']:<20.1f} {'-':<15}")
        
        # Token length comparison
        print("\n🔤 TOKEN LENGTH (tokens):")
        print(f"{'Metric':<15} {'QWEN RAG':<20} {'OpenAI Baseline':<20} {'Ratio (Q/O)':<15}")
        print("-" * 70)
        
        qwen_t = q_stats['qwen_rag']['tokens']
        openai_t = q_stats['openai']['tokens']
        ratio_t = q_stats['ratios']['tokens']
        
        print(f"{'Mean':<15} {qwen_t['mean']:<20.1f} {openai_t['mean']:<20.1f} {ratio_t['mean']:<15.2f}x")
        print(f"{'Median':<15} {qwen_t['median']:<20.1f} {openai_t['median']:<20.1f} {ratio_t['median']:<15.2f}x")
        print(f"{'Min':<15} {qwen_t['min']:<20} {openai_t['min']:<20} {ratio_t['min']:<15.2f}x")
        print(f"{'Max':<15} {qwen_t['max']:<20} {openai_t['max']:<20} {ratio_t['max']:<15.2f}x")
        print(f"{'Std Dev':<15} {qwen_t['std']:<20.1f} {openai_t['std']:<20.1f} {'-':<15}")
        
        # Summary insights
        print(f"\n💡 Key Insights for {q_type}:")
        print(f"   • QWEN RAG is {ratio_c['mean']:.2f}x longer in characters ({ratio_t['mean']:.2f}x in tokens)")
        print(f"   • QWEN RAG range: {qwen_c['min']}-{qwen_c['max']} chars ({qwen_t['min']}-{qwen_t['max']} tokens)")
        print(f"   • OpenAI range: {openai_c['min']}-{openai_c['max']} chars ({openai_t['min']}-{openai_t['max']} tokens)")
        if ratio_c['mean'] > 5:
            print(f"   • ⚠️  Very high length ratio suggests QWEN RAG provides much more comprehensive answers")
        elif ratio_c['mean'] > 3:
            print(f"   • ✓ Moderate length ratio indicates QWEN RAG provides detailed but reasonable answers")
        else:
            print(f"   • ✓ Lower length ratio suggests answers are more aligned in length")
    
    print("\n" + "="*100)


def save_results(stats: Dict):
    """Save detailed statistics to JSON"""
    output_path = RESULTS_DIR / "answer_length_analysis_rag.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Detailed length statistics saved to: {output_path}")


def main():
    """Main execution function"""
    print("="*100)
    print("Answer Length Analysis: QWEN RAG vs OpenAI Baseline")
    print("="*100)
    
    # Load data
    print("\nLoading evaluation results...")
    df = load_data()
    print(f"✅ Loaded {len(df)} question-answer pairs")
    
    # Calculate statistics
    print("\nCalculating detailed length statistics...")
    stats = calculate_length_stats(df)
    
    # Print detailed table
    print_detailed_table(stats)
    
    # Save results
    save_results(stats)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

