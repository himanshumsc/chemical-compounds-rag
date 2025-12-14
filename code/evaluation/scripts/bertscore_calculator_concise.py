#!/usr/bin/env python3
"""
BERTScore Calculator: QWEN RAG Concise vs OpenAI Baseline
Compares QWEN RAG Concise-generated answers (candidate) against OpenAI-generated answers (reference)
using semantic similarity via BERT embeddings
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

# Directories
QWEN_CONCISE_DIR = Path("/home/himanshu/dev/output/qwen_rag_concise")
OPENAI_BASELINE_DIR = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive")
RESULTS_DIR = Path(__file__).parent.parent / "results"
VIS_DIR = Path(__file__).parent.parent / "visualizations"

# Create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# BERTScore model configuration
DEFAULT_MODEL = "microsoft/deberta-xlarge-mnli"


def extract_file_base(filename: str) -> str:
    """Extract base filename for matching."""
    base = filename.replace("__answers.json", "").replace(".json", "")
    return base


def load_qwen_concise_answers(qwen_dir: Path) -> Dict[str, List[Dict]]:
    """Load all QWEN RAG Concise answer files."""
    qwen_data = {}
    qwen_files = sorted(qwen_dir.glob("*__answers.json"))
    
    print(f"Loading {len(qwen_files)} QWEN RAG Concise answer files...")
    
    for qwen_file in qwen_files:
        try:
            with open(qwen_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_base = extract_file_base(qwen_file.name)
            qwen_data[file_base] = data.get('answers', [])
            
        except Exception as e:
            print(f"Error loading {qwen_file.name}: {e}")
    
    print(f"Loaded {len(qwen_data)} QWEN RAG Concise files")
    return qwen_data


def load_openai_baseline_answers(openai_dir: Path) -> Dict[str, List[Dict]]:
    """Load all OpenAI baseline answer files."""
    openai_data = {}
    openai_files = sorted(openai_dir.glob("*.json"))
    
    print(f"Loading {len(openai_files)} OpenAI baseline answer files...")
    
    for openai_file in openai_files:
        try:
            with open(openai_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_base = extract_file_base(openai_file.name)
            openai_data[file_base] = data.get('qa_pairs', [])
            
        except Exception as e:
            print(f"Error loading {openai_file.name}: {e}")
    
    print(f"Loaded {len(openai_data)} OpenAI baseline files")
    return openai_data


def match_qa_pairs(qwen_data: Dict, openai_data: Dict) -> List[Dict]:
    """Match QWEN RAG Concise and OpenAI baseline answers by file and question index."""
    matched_pairs = []
    unmatched_qwen = []
    unmatched_openai = []
    
    print("\nMatching QA pairs...")
    
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
        
        max_pairs = min(len(qwen_answers), len(openai_qa_pairs))
        
        for idx in range(max_pairs):
            qwen_answer_obj = qwen_answers[idx]
            openai_qa_obj = openai_qa_pairs[idx]
            
            qwen_question = qwen_answer_obj.get('question', '').strip()
            qwen_answer = qwen_answer_obj.get('answer', '').strip()
            openai_question = openai_qa_obj.get('question', '').strip()
            openai_answer = openai_qa_obj.get('answer', '').strip()
            
            if qwen_question and openai_question and qwen_question != openai_question:
                print(f"  ⚠️  Warning: Questions don't match for {file_base} Q{idx+1}")
            
            matched_pairs.append({
                'file_base': file_base,
                'question_index': idx,
                'question': openai_question or qwen_question,
                'qwen_concise_answer': qwen_answer,
                'openai_answer': openai_answer,
                'qwen_concise_length': len(qwen_answer),
                'openai_length': len(openai_answer),
                'qwen_concise_tokens': len(qwen_answer.split()),
                'openai_tokens': len(openai_answer.split()),
            })
    
    print(f"\nMatched {len(matched_pairs)} question-answer pairs")
    if unmatched_qwen:
        print(f"  ⚠️  {len(unmatched_qwen)} QWEN Concise files without OpenAI matches")
    if unmatched_openai:
        print(f"  ⚠️  {len(unmatched_openai)} OpenAI files without QWEN Concise matches")
    
    return matched_pairs


def calculate_bertscore_scores(
    matched_pairs: List[Dict], 
    model_type: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: Optional[str] = None
) -> List[Dict]:
    """Calculate BERTScore for all matched pairs."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nCalculating BERTScore using model: {model_type}")
    print(f"Device: {device}")
    print(f"Processing {len(matched_pairs)} pairs in batches of {batch_size}...")
    
    candidates = [pair['qwen_concise_answer'] for pair in matched_pairs]
    references = [pair['openai_answer'] for pair in matched_pairs]
    
    for i, (cand, ref) in enumerate(zip(candidates, references)):
        if not cand.strip():
            candidates[i] = " "
        if not ref.strip():
            references[i] = " "
    
    try:
        print("  Loading BERT model (this may take a moment on first run)...")
        P, R, F1 = score(
            candidates,
            references,
            lang='en',
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=True,
            rescale_with_baseline=True
        )
        
        print(f"  ✅ BERTScore calculation complete!")
        
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
    """Calculate aggregated statistics."""
    print("\nCalculating aggregated statistics...")
    
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
    
    overall_stats = {
        'precision': calc_stats(precision_scores),
        'recall': calc_stats(recall_scores),
        'f1': calc_stats(f1_scores),
    }
    
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
    
    per_compound_stats = {}
    compounds = defaultdict(list)
    for r in results:
        compounds[r['file_base']].append(r['bertscore_f1'])
    
    for compound, scores in compounds.items():
        per_compound_stats[compound] = {
            'mean_f1': statistics.mean(scores),
            'count': len(scores)
        }
    
    qwen_lengths = [r['qwen_concise_length'] for r in results]
    openai_lengths = [r['openai_length'] for r in results]
    qwen_tokens = [r['qwen_concise_tokens'] for r in results]
    openai_tokens = [r['openai_tokens'] for r in results]
    
    length_stats = {
        'qwen_concise': {
            'mean_chars': statistics.mean(qwen_lengths),
            'median_chars': statistics.median(qwen_lengths),
            'min_chars': min(qwen_lengths),
            'max_chars': max(qwen_lengths),
            'mean_tokens': statistics.mean(qwen_tokens),
            'median_tokens': statistics.median(qwen_tokens),
            'min_tokens': min(qwen_tokens),
            'max_tokens': max(qwen_tokens),
        },
        'openai': {
            'mean_chars': statistics.mean(openai_lengths),
            'median_chars': statistics.median(openai_lengths),
            'min_chars': min(openai_lengths),
            'max_chars': max(openai_lengths),
            'mean_tokens': statistics.mean(openai_tokens),
            'median_tokens': statistics.median(openai_tokens),
            'min_tokens': min(openai_tokens),
            'max_tokens': max(openai_tokens),
        },
        'length_ratio': {
            'mean_chars_ratio': statistics.mean([q/o if o > 0 else 0 for q, o in zip(qwen_lengths, openai_lengths)]),
            'mean_tokens_ratio': statistics.mean([q/o if o > 0 else 0 for q, o in zip(qwen_tokens, openai_tokens)]),
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
    """Load BLEU results for comparison with BERTScore."""
    bleu_csv = RESULTS_DIR / "per_question_bleu_concise.csv"
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
            return None
    except Exception as e:
        print(f"  ⚠️  Error loading BLEU results: {e}")
        return None


def add_bleu_comparison(results: List[Dict], stats: Dict) -> Dict:
    """Add BLEU comparison to statistics."""
    bleu_data = load_bleu_results_for_comparison()
    if not bleu_data:
        return stats
    
    print("\nAdding BLEU comparison...")
    
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
        if len(bertscore_f1_scores) > 1:
            if np is not None:
                correlation_f1_bleu4 = np.corrcoef(bertscore_f1_scores, bleu_4_scores)[0, 1]
                correlation_f1_bleu1 = np.corrcoef(bertscore_f1_scores, bleu_1_scores)[0, 1]
            else:
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
    """Save results to CSV and JSON files."""
    print("\nSaving results...")
    
    if pd is not None:
        df = pd.DataFrame(results)
        csv_path = RESULTS_DIR / "per_question_bertscore_concise.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Saved detailed results to: {csv_path}")
    else:
        json_path = RESULTS_DIR / "per_question_bertscore_concise.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved detailed results to: {json_path}")
    
    summary_path = RESULTS_DIR / "summary_bertscore_concise.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved summary metrics to: {summary_path}")


def print_summary(stats: Dict):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("BERTScore Summary: QWEN RAG Concise vs OpenAI Baseline")
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
    qwen_len = stats['answer_lengths']['qwen_concise']
    openai_len = stats['answer_lengths']['openai']
    ratio = stats['answer_lengths']['length_ratio']
    
    print(f"  QWEN Concise (candidate):")
    print(f"    Mean chars:   {qwen_len['mean_chars']:.1f}")
    print(f"    Median chars: {qwen_len['median_chars']:.1f}")
    print(f"    Mean tokens:  {qwen_len['mean_tokens']:.1f}")
    print(f"    Range chars:  [{qwen_len['min_chars']}, {qwen_len['max_chars']}]")
    print(f"  OpenAI (reference):")
    print(f"    Mean chars:   {openai_len['mean_chars']:.1f}")
    print(f"    Median chars: {openai_len['median_chars']:.1f}")
    print(f"    Mean tokens:  {openai_len['mean_tokens']:.1f}")
    print(f"    Range chars:  [{openai_len['min_chars']}, {openai_len['max_chars']}]")
    print(f"  Length Ratio (QWEN Concise / OpenAI):")
    print(f"    Mean chars ratio:  {ratio['mean_chars_ratio']:.2f}x")
    print(f"    Mean tokens ratio: {ratio['mean_tokens_ratio']:.2f}x")
    
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
    
    parser = argparse.ArgumentParser(description="Calculate BERTScore for QWEN RAG Concise vs OpenAI baseline")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"BERT model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for BERTScore calculation (default: 16)")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help="Device to use (default: auto)")
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("BERTScore Calculator: QWEN RAG Concise vs OpenAI Baseline")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    
    qwen_data = load_qwen_concise_answers(QWEN_CONCISE_DIR)
    openai_data = load_openai_baseline_answers(OPENAI_BASELINE_DIR)
    
    matched_pairs = match_qa_pairs(qwen_data, openai_data)
    
    if not matched_pairs:
        print("❌ No matched pairs found. Exiting.")
        return
    
    results = calculate_bertscore_scores(
        matched_pairs,
        model_type=args.model,
        batch_size=args.batch_size,
        device=device
    )
    
    if not results:
        print("❌ BERTScore calculation failed. Exiting.")
        return
    
    stats = aggregate_statistics(results)
    stats = add_bleu_comparison(results, stats)
    save_results(results, stats)
    print_summary(stats)
    
    print(f"\n✅ BERTScore calculation complete!")
    print(f"   Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

