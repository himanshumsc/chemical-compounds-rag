#!/usr/bin/env python3
"""
BERTScore Calculator: Regenerated QWEN vs Regenerated OpenAI Answers
Compares regenerated QWEN-generated answers (candidate) against regenerated OpenAI-generated answers (reference)
using semantic similarity via BERT embeddings
Uses:
- QWEN: dev/output/qwen_regenerated (regenerated with vLLM, max_tokens=500)
- OpenAI: dev/test/data/processed/qa_pairs_individual_components_comprehensive (regenerated with comprehensive_text)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

try:
    from bert_score import score
    import torch
except ImportError:
    print("Error: bert-score not installed. Install with: pip install bert-score")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Warning: pandas/numpy not installed. Some features may be limited.")
    pd = None
    np = None

# Directories - UPDATED FOR REGENERATED DATA
QWEN_DIR = Path("/home/himanshu/dev/output/qwen_regenerated")
OPENAI_DIR = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive")
RESULTS_DIR = Path(__file__).parent.parent / "results"
VIS_DIR = Path(__file__).parent.parent / "visualizations"

# Create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# BERTScore model configuration
DEFAULT_MODEL = "microsoft/deberta-xlarge-mnli"  # Best for semantic similarity
# Alternatives: "roberta-large", "bert-base-uncased"


def extract_file_base(filename: str) -> str:
    """
    Extract base filename for matching.
    
    Examples:
        "37_Carbon_Dioxide__answers.json" -> "37_Carbon_Dioxide"
        "37_Carbon_Dioxide.json" -> "37_Carbon_Dioxide"
    """
    base = filename.replace("__answers.json", "").replace(".json", "")
    return base


def load_qwen_answers(qwen_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load all QWEN answer files.
    
    Returns:
        Dictionary mapping file_base -> list of answers
    """
    qwen_data = {}
    qwen_files = sorted(qwen_dir.glob("*__answers.json"))
    
    print(f"Loading {len(qwen_files)} QWEN answer files...")
    
    for qwen_file in qwen_files:
        try:
            with open(qwen_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_base = extract_file_base(qwen_file.name)
            qwen_data[file_base] = data.get('answers', [])
            
        except Exception as e:
            print(f"Error loading {qwen_file.name}: {e}")
    
    print(f"Loaded {len(qwen_data)} QWEN files")
    return qwen_data


def load_openai_answers(openai_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load all OpenAI answer files.
    
    Returns:
        Dictionary mapping file_base -> list of qa_pairs
    """
    openai_data = {}
    openai_files = sorted(openai_dir.glob("*.json"))
    
    print(f"Loading {len(openai_files)} OpenAI answer files...")
    
    for openai_file in openai_files:
        try:
            with open(openai_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_base = extract_file_base(openai_file.name)
            openai_data[file_base] = data.get('qa_pairs', [])
            
        except Exception as e:
            print(f"Error loading {openai_file.name}: {e}")
    
    print(f"Loaded {len(openai_data)} OpenAI files")
    return openai_data


def match_qa_pairs(qwen_data: Dict, openai_data: Dict) -> List[Dict]:
    """
    Match QWEN and OpenAI answers by file and question index.
    
    Returns:
        List of matched pairs with metadata
    """
    matched_pairs = []
    unmatched_qwen = []
    unmatched_openai = []
    
    print("\nMatching QA pairs...")
    
    # Find all unique file bases
    all_files = set(qwen_data.keys()) | set(openai_data.keys())
    
    for file_base in sorted(all_files):
        qwen_answers = qwen_data.get(file_base, [])
        openai_qa_pairs = openai_data.get(file_base, [])
        
        if not qwen_answers:
            unmatched_qwen.append(file_base)
            continue
        
        if not openai_qa_pairs:
            unmatched_openai.append(file_base)
            continue
        
        # Match by index (Q1=0, Q2=1, Q3=2, Q4=3)
        max_pairs = min(len(qwen_answers), len(openai_qa_pairs))
        
        for idx in range(max_pairs):
            qwen_answer_obj = qwen_answers[idx]
            openai_qa_obj = openai_qa_pairs[idx]
            
            qwen_question = qwen_answer_obj.get('question', '').strip()
            qwen_answer = qwen_answer_obj.get('answer', '').strip()
            openai_question = openai_qa_obj.get('question', '').strip()
            openai_answer = openai_qa_obj.get('answer', '').strip()
            
            # Optional: Verify questions match (log warning if not)
            if qwen_question and openai_question and qwen_question != openai_question:
                print(f"  ⚠️  Warning: Questions don't match for {file_base} Q{idx+1}")
            
            matched_pairs.append({
                'file_base': file_base,
                'question_index': idx,
                'question': openai_question or qwen_question,
                'qwen_answer': qwen_answer,
                'openai_answer': openai_answer,
                'qwen_length': len(qwen_answer),
                'openai_length': len(openai_answer),
            })
    
    print(f"\nMatched {len(matched_pairs)} question-answer pairs")
    if unmatched_qwen:
        print(f"  ⚠️  {len(unmatched_qwen)} QWEN files without OpenAI matches")
    if unmatched_openai:
        print(f"  ⚠️  {len(unmatched_openai)} OpenAI files without QWEN matches")
    
    return matched_pairs


def calculate_bertscore_scores(
    matched_pairs: List[Dict], 
    model_type: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Calculate BERTScore for all matched pairs.
    
    Args:
        matched_pairs: List of matched QA pairs
        model_type: BERT model to use
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu'), auto-detect if None
    
    Returns:
        List of dictionaries with BERTScore results added
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nCalculating BERTScore using model: {model_type}")
    print(f"Device: {device}")
    print(f"Processing {len(matched_pairs)} pairs in batches of {batch_size}...")
    
    # Extract candidates and references
    candidates = [pair['qwen_answer'] for pair in matched_pairs]
    references = [pair['openai_answer'] for pair in matched_pairs]
    
    # Handle empty answers
    for i, (cand, ref) in enumerate(zip(candidates, references)):
        if not cand.strip():
            candidates[i] = " "  # BERTScore needs non-empty string
        if not ref.strip():
            references[i] = " "
    
    try:
        # Calculate BERTScore
        # This will download the model on first run (~1.5GB for deberta-xlarge)
        print("  Loading BERT model (this may take a moment on first run)...")
        P, R, F1 = score(
            candidates,
            references,
            lang='en',
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=True,
            rescale_with_baseline=True  # Rescale scores to be more interpretable
        )
        
        print(f"  ✅ BERTScore calculation complete!")
        
        # Convert to lists and combine with metadata
        results = []
        for i, pair in enumerate(matched_pairs):
            results.append({
                **pair,
                'bertscore_precision': float(P[i].cpu().item()),
                'bertscore_recall': float(R[i].cpu().item()),
                'bertscore_f1': float(F1[i].cpu().item())
            })
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error calculating BERTScore: {e}")
        import traceback
        traceback.print_exc()
        return []


def aggregate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate aggregated statistics.
    
    Returns:
        Dictionary with various aggregated metrics
    """
    print("\nCalculating aggregated statistics...")
    
    # Extract BERTScore metrics
    precision_scores = [r['bertscore_precision'] for r in results]
    recall_scores = [r['bertscore_recall'] for r in results]
    f1_scores = [r['bertscore_f1'] for r in results]
    
    def calc_stats(scores):
        return {
            'mean': statistics.mean(scores),
            'median': statistics.median(scores),
            'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'min': min(scores),
            'max': max(scores),
            'count': len(scores)
        }
    
    # Overall statistics
    overall_stats = {
        'precision': calc_stats(precision_scores),
        'recall': calc_stats(recall_scores),
        'f1': calc_stats(f1_scores),
    }
    
    # Per-question-type statistics (Q1, Q2, Q3, Q4)
    per_question_stats = {}
    for q_idx in range(4):
        q_results = [r for r in results if r['question_index'] == q_idx]
        if q_results:
            q_precision = [r['bertscore_precision'] for r in q_results]
            q_recall = [r['bertscore_recall'] for r in q_results]
            q_f1 = [r['bertscore_f1'] for r in q_results]
            
            per_question_stats[f'Q{q_idx+1}'] = {
                'count': len(q_results),
                'mean_precision': statistics.mean(q_precision),
                'mean_recall': statistics.mean(q_recall),
                'mean_f1': statistics.mean(q_f1),
                'median_f1': statistics.median(q_f1),
                'std_f1': statistics.stdev(q_f1) if len(q_f1) > 1 else 0.0,
            }
    
    # Per-compound statistics
    per_compound_stats = {}
    compounds = defaultdict(list)
    for r in results:
        compounds[r['file_base']].append(r['bertscore_f1'])
    
    for compound, scores in compounds.items():
        per_compound_stats[compound] = {
            'mean_f1': statistics.mean(scores),
            'count': len(scores)
        }
    
    # Answer length statistics
    qwen_lengths = [r['qwen_length'] for r in results]
    openai_lengths = [r['openai_length'] for r in results]
    
    length_stats = {
        'qwen': {
            'mean': statistics.mean(qwen_lengths),
            'median': statistics.median(qwen_lengths),
            'min': min(qwen_lengths),
            'max': max(qwen_lengths),
        },
        'openai': {
            'mean': statistics.mean(openai_lengths),
            'median': statistics.median(openai_lengths),
            'min': min(openai_lengths),
            'max': max(openai_lengths),
        }
    }
    
    return {
        'overall': overall_stats,
        'per_question_type': per_question_stats,
        'per_compound': per_compound_stats,
        'answer_lengths': length_stats,
        'total_pairs': len(results)
    }


def load_bleu_results_for_comparison() -> Optional[Dict]:
    """
    Load BLEU results for comparison with BERTScore.
    
    Returns:
        Dictionary mapping (file_base, question_index) -> BLEU scores
    """
    bleu_csv = RESULTS_DIR / "per_question_bleu_regenerated.csv"
    if not bleu_csv.exists():
        print("  ⚠️  BLEU results not found for comparison")
        return None
    
    try:
        if pd is not None:
            df = pd.read_csv(bleu_csv)
            bleu_dict = {}
            for _, row in df.iterrows():
                key = (row['file_base'], row['question_index'])
                bleu_dict[key] = {
                    'bleu_1': row.get('bleu_1', 0),
                    'bleu_4': row.get('bleu_4', 0)
                }
            return bleu_dict
        else:
            # Fallback: load as JSON if available
            return None
    except Exception as e:
        print(f"  ⚠️  Error loading BLEU results: {e}")
        return None


def add_bleu_comparison(results: List[Dict], stats: Dict) -> Dict:
    """
    Add BLEU comparison to statistics.
    
    Returns:
        Updated statistics with BLEU comparison
    """
    bleu_data = load_bleu_results_for_comparison()
    if not bleu_data:
        return stats
    
    print("\nAdding BLEU comparison...")
    
    # Match BERTScore results with BLEU results
    bertscore_f1_scores = []
    bleu_4_scores = []
    bleu_1_scores = []
    
    for r in results:
        key = (r['file_base'], r['question_index'])
        if key in bleu_data:
            bertscore_f1_scores.append(r['bertscore_f1'])
            bleu_4_scores.append(bleu_data[key]['bleu_4'])
            bleu_1_scores.append(bleu_data[key]['bleu_1'])
    
    if bertscore_f1_scores and bleu_4_scores:
        # Calculate correlation (simple linear correlation)
        if len(bertscore_f1_scores) > 1:
            if np is not None:
                correlation_f1_bleu4 = np.corrcoef(bertscore_f1_scores, bleu_4_scores)[0, 1]
                correlation_f1_bleu1 = np.corrcoef(bertscore_f1_scores, bleu_1_scores)[0, 1]
            else:
                # Simple correlation calculation
                correlation_f1_bleu4 = statistics.correlation(bertscore_f1_scores, bleu_4_scores) if len(bertscore_f1_scores) > 1 else 0.0
                correlation_f1_bleu1 = statistics.correlation(bertscore_f1_scores, bleu_1_scores) if len(bertscore_f1_scores) > 1 else 0.0
        else:
            correlation_f1_bleu4 = 0.0
            correlation_f1_bleu1 = 0.0
        
        stats['comparison_with_bleu'] = {
            'correlation_f1_bleu4': float(correlation_f1_bleu4) if 'correlation_f1_bleu4' in locals() else 0.0,
            'correlation_f1_bleu1': float(correlation_f1_bleu1) if 'correlation_f1_bleu1' in locals() else 0.0,
            'matched_pairs': len(bertscore_f1_scores),
            'mean_bertscore_f1': statistics.mean(bertscore_f1_scores),
            'mean_bleu_4': statistics.mean(bleu_4_scores),
            'mean_bleu_1': statistics.mean(bleu_1_scores),
        }
    
    return stats


def save_results(results: List[Dict], stats: Dict):
    """
    Save results to CSV and JSON files.
    """
    print("\nSaving results...")
    
    # Save detailed CSV - use regenerated suffix
    if pd is not None:
        df = pd.DataFrame(results)
        csv_path = RESULTS_DIR / "per_question_bertscore_regenerated.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Saved detailed results to: {csv_path}")
    else:
        # Fallback: save as JSON
        json_path = RESULTS_DIR / "per_question_bertscore_regenerated.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved detailed results to: {json_path}")
    
    # Save summary statistics - use regenerated suffix
    summary_path = RESULTS_DIR / "summary_bertscore_regenerated.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved summary metrics to: {summary_path}")


def print_summary(stats: Dict):
    """
    Print summary statistics to console.
    """
    print("\n" + "="*70)
    print("BERTScore Summary - REGENERATED DATA")
    print("="*70)
    print("QWEN: Regenerated with vLLM (max_tokens=500)")
    print("OpenAI: Regenerated with comprehensive_text (max_tokens=500)")
    print("="*70)
    
    overall = stats['overall']
    print(f"\nTotal Question-Answer Pairs: {stats['total_pairs']}")
    
    print("\nOverall Corpus BERTScore:")
    for metric_type in ['precision', 'recall', 'f1']:
        s = overall[metric_type]
        print(f"  {metric_type.upper()}:")
        print(f"    Mean:   {s['mean']:.4f}")
        print(f"    Median: {s['median']:.4f}")
        print(f"    Std:    {s['std']:.4f}")
        print(f"    Range:  [{s['min']:.4f}, {s['max']:.4f}]")
    
    print("\nPer-Question-Type BERTScore F1:")
    for q_type, q_stats in stats['per_question_type'].items():
        print(f"  {q_type}:")
        print(f"    Mean F1:   {q_stats['mean_f1']:.4f}")
        print(f"    Median F1: {q_stats['median_f1']:.4f}")
        print(f"    Mean Precision: {q_stats['mean_precision']:.4f}")
        print(f"    Mean Recall:    {q_stats['mean_recall']:.4f}")
        print(f"    Count:  {q_stats['count']}")
    
    print("\nAnswer Length Statistics:")
    qwen_len = stats['answer_lengths']['qwen']
    openai_len = stats['answer_lengths']['openai']
    print(f"  QWEN (candidate, regenerated):")
    print(f"    Mean:   {qwen_len['mean']:.1f} chars")
    print(f"    Median: {qwen_len['median']:.1f} chars")
    print(f"    Range:  [{qwen_len['min']}, {qwen_len['max']}] chars")
    print(f"  OpenAI (reference, regenerated):")
    print(f"    Mean:   {openai_len['mean']:.1f} chars")
    print(f"    Median: {openai_len['median']:.1f} chars")
    print(f"    Range:  [{openai_len['min']}, {openai_len['max']}] chars")
    
    # BLEU comparison if available
    if 'comparison_with_bleu' in stats:
        bleu_comp = stats['comparison_with_bleu']
        print("\nComparison with BLEU:")
        print(f"  Correlation (BERTScore F1 vs BLEU-4): {bleu_comp['correlation_f1_bleu4']:.4f}")
        print(f"  Correlation (BERTScore F1 vs BLEU-1): {bleu_comp['correlation_f1_bleu1']:.4f}")
        print(f"  Mean BERTScore F1: {bleu_comp['mean_bertscore_f1']:.4f}")
        print(f"  Mean BLEU-4:        {bleu_comp['mean_bleu_4']:.4f}")
        print(f"  Mean BLEU-1:        {bleu_comp['mean_bleu_1']:.4f}")
        print(f"  Matched pairs:     {bleu_comp['matched_pairs']}")
    
    print("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate BERTScore for Regenerated QWEN vs OpenAI answers")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"BERT model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for BERTScore calculation (default: 32)")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help="Device to use (default: auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("BERTScore Calculator: Regenerated QWEN vs Regenerated OpenAI Answers")
    print("="*70)
    print(f"QWEN Directory: {QWEN_DIR}")
    print(f"OpenAI Directory: {OPENAI_DIR}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    
    # Load data
    qwen_data = load_qwen_answers(QWEN_DIR)
    openai_data = load_openai_answers(OPENAI_DIR)
    
    # Match pairs
    matched_pairs = match_qa_pairs(qwen_data, openai_data)
    
    if not matched_pairs:
        print("❌ No matched pairs found. Exiting.")
        return
    
    # Calculate BERTScore
    results = calculate_bertscore_scores(
        matched_pairs,
        model_type=args.model,
        batch_size=args.batch_size,
        device=device
    )
    
    if not results:
        print("❌ BERTScore calculation failed. Exiting.")
        return
    
    # Aggregate statistics
    stats = aggregate_statistics(results)
    
    # Add BLEU comparison if available
    stats = add_bleu_comparison(results, stats)
    
    # Save results
    save_results(results, stats)
    
    # Print summary
    print_summary(stats)
    
    print(f"\n✅ BERTScore calculation complete!")
    print(f"   Results saved to: {RESULTS_DIR}")
    print(f"   Files: per_question_bertscore_regenerated.csv, summary_bertscore_regenerated.json")


if __name__ == "__main__":
    main()

