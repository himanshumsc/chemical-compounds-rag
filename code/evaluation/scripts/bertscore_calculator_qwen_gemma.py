#!/usr/bin/env python3
"""
BERTScore Calculator: Qwen RAG Concise and Gemma RAG Concise vs OpenAI Baseline
Compares both Qwen and Gemma RAG Concise answers against OpenAI baseline answers
using semantic similarity via BERT embeddings
Uses:
- Qwen: dev/output/qwen_rag_concise
- Gemma: dev/output/gemma3_rag_concise
- OpenAI: dev/test/data/processed/qa_pairs_individual_components_comprehensive
"""

import json
import sys
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories
QWEN_DIR = Path("/home/himanshu/dev/output/qwen_rag_concise")
GEMMA_DIR = Path("/home/himanshu/dev/output/gemma3_rag_concise")
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


def load_model_answers(model_dir: Path, model_name: str) -> Dict[str, List[Dict]]:
    """
    Load all model answer files (Qwen or Gemma).
    
    Args:
        model_dir: Directory containing answer files
        model_name: Name of the model (for logging)
    
    Returns:
        Dictionary mapping file_base -> list of answers
    """
    model_data = {}
    model_files = sorted(model_dir.glob("*__answers.json"))
    
    logger.info(f"Loading {len(model_files)} {model_name} answer files...")
    
    for model_file in model_files:
        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_base = extract_file_base(model_file.name)
            model_data[file_base] = data.get('answers', [])
            
        except Exception as e:
            logger.error(f"Error loading {model_file.name}: {e}")
    
    logger.info(f"Loaded {len(model_data)} {model_name} files")
    return model_data


def load_openai_answers(openai_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load all OpenAI answer files.
    
    Returns:
        Dictionary mapping file_base -> list of qa_pairs
    """
    openai_data = {}
    openai_files = sorted(openai_dir.glob("*.json"))
    
    logger.info(f"Loading {len(openai_files)} OpenAI answer files...")
    
    for openai_file in openai_files:
        try:
            with open(openai_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_base = extract_file_base(openai_file.name)
            openai_data[file_base] = data.get('qa_pairs', [])
            
        except Exception as e:
            logger.error(f"Error loading {openai_file.name}: {e}")
    
    logger.info(f"Loaded {len(openai_data)} OpenAI files")
    return openai_data


def match_qa_pairs(qwen_data: Dict, gemma_data: Dict, openai_data: Dict) -> List[Dict]:
    """
    Match Qwen, Gemma, and OpenAI answers by file and question index.
    
    Returns:
        List of matched triplets with metadata
    """
    matched_pairs = []
    unmatched_qwen = []
    unmatched_gemma = []
    unmatched_openai = []
    
    logger.info("Matching QA pairs...")
    
    # Find all unique file bases
    all_files = set(qwen_data.keys()) | set(gemma_data.keys()) | set(openai_data.keys())
    
    for file_base in sorted(all_files):
        qwen_answers = qwen_data.get(file_base, [])
        gemma_answers = gemma_data.get(file_base, [])
        openai_qa_pairs = openai_data.get(file_base, [])
        
        if not qwen_answers:
            unmatched_qwen.append(file_base)
        if not gemma_answers:
            unmatched_gemma.append(file_base)
        if not openai_qa_pairs:
            unmatched_openai.append(file_base)
        
        # Only include if we have all three
        if not qwen_answers or not gemma_answers or not openai_qa_pairs:
            continue
        
        # Match by index (Q1=0, Q2=1, Q3=2, Q4=3)
        max_pairs = min(len(qwen_answers), len(gemma_answers), len(openai_qa_pairs))
        
        for idx in range(max_pairs):
            qwen_answer_obj = qwen_answers[idx]
            gemma_answer_obj = gemma_answers[idx]
            openai_qa_obj = openai_qa_pairs[idx]
            
            qwen_question = qwen_answer_obj.get('question', '').strip()
            qwen_answer = qwen_answer_obj.get('answer', '').strip()
            gemma_question = gemma_answer_obj.get('question', '').strip()
            gemma_answer = gemma_answer_obj.get('answer', '').strip()
            openai_question = openai_qa_obj.get('question', '').strip()
            openai_answer = openai_qa_obj.get('answer', '').strip()
            
            # Optional: Verify questions match (log warning if not)
            questions = [q for q in [qwen_question, gemma_question, openai_question] if q]
            if len(set(questions)) > 1:
                logger.warning(f"Questions don't match for {file_base} Q{idx+1}")
            
            matched_pairs.append({
                'file_base': file_base,
                'question_index': idx,
                'question': openai_question or qwen_question or gemma_question,
                'qwen_answer': qwen_answer,
                'gemma_answer': gemma_answer,
                'openai_answer': openai_answer,
                'qwen_length': len(qwen_answer),
                'gemma_length': len(gemma_answer),
                'openai_length': len(openai_answer),
            })
    
    logger.info(f"Matched {len(matched_pairs)} question-answer triplets")
    if unmatched_qwen:
        logger.warning(f"{len(unmatched_qwen)} Qwen files without matches")
    if unmatched_gemma:
        logger.warning(f"{len(unmatched_gemma)} Gemma files without matches")
    if unmatched_openai:
        logger.warning(f"{len(unmatched_openai)} OpenAI files without matches")
    
    return matched_pairs


def calculate_bertscore_scores(
    matched_pairs: List[Dict], 
    model_type: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Calculate BERTScore for all matched pairs (both Qwen and Gemma vs OpenAI).
    
    Args:
        matched_pairs: List of matched QA triplets
        model_type: BERT model to use
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu'), auto-detect if None
    
    Returns:
        List of dictionaries with BERTScore results added
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Calculating BERTScore using model: {model_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Processing {len(matched_pairs)} pairs in batches of {batch_size}...")
    
    # Extract candidates and references for Qwen
    qwen_candidates = [pair['qwen_answer'] for pair in matched_pairs]
    gemma_candidates = [pair['gemma_answer'] for pair in matched_pairs]
    references = [pair['openai_answer'] for pair in matched_pairs]
    
    # Handle empty answers
    for i, (qwen_cand, gemma_cand, ref) in enumerate(zip(qwen_candidates, gemma_candidates, references)):
        if not qwen_cand.strip():
            qwen_candidates[i] = " "  # BERTScore needs non-empty string
        if not gemma_cand.strip():
            gemma_candidates[i] = " "
        if not ref.strip():
            references[i] = " "
    
    try:
        # Calculate BERTScore for Qwen
        logger.info("Calculating BERTScore for Qwen vs OpenAI...")
        logger.info("Loading BERT model (this may take a moment on first run)...")
        P_qwen, R_qwen, F1_qwen = score(
            qwen_candidates,
            references,
            lang='en',
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=True,
            rescale_with_baseline=True  # Rescale scores to be more interpretable
        )
        logger.info("Qwen BERTScore calculation complete!")
        
        # Calculate BERTScore for Gemma
        logger.info("Calculating BERTScore for Gemma vs OpenAI...")
        P_gemma, R_gemma, F1_gemma = score(
            gemma_candidates,
            references,
            lang='en',
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=True,
            rescale_with_baseline=True
        )
        logger.info("Gemma BERTScore calculation complete!")
        
        # Convert to lists and combine with metadata
        results = []
        for i, pair in enumerate(matched_pairs):
            results.append({
                **pair,
                'qwen_bertscore_precision': float(P_qwen[i].cpu().item()),
                'qwen_bertscore_recall': float(R_qwen[i].cpu().item()),
                'qwen_bertscore_f1': float(F1_qwen[i].cpu().item()),
                'gemma_bertscore_precision': float(P_gemma[i].cpu().item()),
                'gemma_bertscore_recall': float(R_gemma[i].cpu().item()),
                'gemma_bertscore_f1': float(F1_gemma[i].cpu().item()),
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def aggregate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate aggregated statistics for both models.
    
    Returns:
        Dictionary with various aggregated metrics
    """
    logger.info("Calculating aggregated statistics...")
    
    def calc_stats(scores):
        return {
            'mean': statistics.mean(scores),
            'median': statistics.median(scores),
            'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'min': min(scores),
            'max': max(scores),
            'count': len(scores)
        }
    
    # Extract BERTScore metrics for both models
    qwen_precision = [r['qwen_bertscore_precision'] for r in results]
    qwen_recall = [r['qwen_bertscore_recall'] for r in results]
    qwen_f1 = [r['qwen_bertscore_f1'] for r in results]
    
    gemma_precision = [r['gemma_bertscore_precision'] for r in results]
    gemma_recall = [r['gemma_bertscore_recall'] for r in results]
    gemma_f1 = [r['gemma_bertscore_f1'] for r in results]
    
    # Overall statistics
    qwen_overall = {
        'precision': calc_stats(qwen_precision),
        'recall': calc_stats(qwen_recall),
        'f1': calc_stats(qwen_f1),
    }
    
    gemma_overall = {
        'precision': calc_stats(gemma_precision),
        'recall': calc_stats(gemma_recall),
        'f1': calc_stats(gemma_f1),
    }
    
    # Per-question-type statistics (Q1, Q2, Q3, Q4)
    qwen_per_question = {}
    gemma_per_question = {}
    
    for q_idx in range(4):
        q_results = [r for r in results if r['question_index'] == q_idx]
        if q_results:
            qwen_q_precision = [r['qwen_bertscore_precision'] for r in q_results]
            qwen_q_recall = [r['qwen_bertscore_recall'] for r in q_results]
            qwen_q_f1 = [r['qwen_bertscore_f1'] for r in q_results]
            
            gemma_q_precision = [r['gemma_bertscore_precision'] for r in q_results]
            gemma_q_recall = [r['gemma_bertscore_recall'] for r in q_results]
            gemma_q_f1 = [r['gemma_bertscore_f1'] for r in q_results]
            
            qwen_per_question[f'Q{q_idx+1}'] = {
                'count': len(q_results),
                'mean_precision': statistics.mean(qwen_q_precision),
                'mean_recall': statistics.mean(qwen_q_recall),
                'mean_f1': statistics.mean(qwen_q_f1),
                'median_f1': statistics.median(qwen_q_f1),
                'std_f1': statistics.stdev(qwen_q_f1) if len(qwen_q_f1) > 1 else 0.0,
            }
            
            gemma_per_question[f'Q{q_idx+1}'] = {
                'count': len(q_results),
                'mean_precision': statistics.mean(gemma_q_precision),
                'mean_recall': statistics.mean(gemma_q_recall),
                'mean_f1': statistics.mean(gemma_q_f1),
                'median_f1': statistics.median(gemma_q_f1),
                'std_f1': statistics.stdev(gemma_q_f1) if len(gemma_q_f1) > 1 else 0.0,
            }
    
    # Per-compound statistics
    qwen_per_compound = {}
    gemma_per_compound = {}
    
    qwen_compounds = defaultdict(list)
    gemma_compounds = defaultdict(list)
    
    for r in results:
        qwen_compounds[r['file_base']].append(r['qwen_bertscore_f1'])
        gemma_compounds[r['file_base']].append(r['gemma_bertscore_f1'])
    
    for compound, scores in qwen_compounds.items():
        qwen_per_compound[compound] = {
            'mean_f1': statistics.mean(scores),
            'count': len(scores)
        }
    
    for compound, scores in gemma_compounds.items():
        gemma_per_compound[compound] = {
            'mean_f1': statistics.mean(scores),
            'count': len(scores)
        }
    
    # Answer length statistics
    qwen_lengths = [r['qwen_length'] for r in results]
    gemma_lengths = [r['gemma_length'] for r in results]
    openai_lengths = [r['openai_length'] for r in results]
    
    length_stats = {
        'qwen': {
            'mean': statistics.mean(qwen_lengths),
            'median': statistics.median(qwen_lengths),
            'min': min(qwen_lengths),
            'max': max(qwen_lengths),
        },
        'gemma': {
            'mean': statistics.mean(gemma_lengths),
            'median': statistics.median(gemma_lengths),
            'min': min(gemma_lengths),
            'max': max(gemma_lengths),
        },
        'openai': {
            'mean': statistics.mean(openai_lengths),
            'median': statistics.median(openai_lengths),
            'min': min(openai_lengths),
            'max': max(openai_lengths),
        }
    }
    
    # Comparison statistics
    comparison = {
        'overall': {
            'qwen_vs_gemma_f1': {
                'qwen_mean': statistics.mean(qwen_f1),
                'gemma_mean': statistics.mean(gemma_f1),
                'difference': statistics.mean(qwen_f1) - statistics.mean(gemma_f1),
                'qwen_better': statistics.mean(qwen_f1) > statistics.mean(gemma_f1),
            },
            'qwen_vs_gemma_precision': {
                'qwen_mean': statistics.mean(qwen_precision),
                'gemma_mean': statistics.mean(gemma_precision),
                'difference': statistics.mean(qwen_precision) - statistics.mean(gemma_precision),
                'qwen_better': statistics.mean(qwen_precision) > statistics.mean(gemma_precision),
            },
            'qwen_vs_gemma_recall': {
                'qwen_mean': statistics.mean(qwen_recall),
                'gemma_mean': statistics.mean(gemma_recall),
                'difference': statistics.mean(qwen_recall) - statistics.mean(gemma_recall),
                'qwen_better': statistics.mean(qwen_recall) > statistics.mean(gemma_recall),
            },
        },
        'per_question_type': {}
    }
    
    for q_idx in range(4):
        q_results = [r for r in results if r['question_index'] == q_idx]
        if q_results:
            qwen_f1_mean = statistics.mean([r['qwen_bertscore_f1'] for r in q_results])
            gemma_f1_mean = statistics.mean([r['gemma_bertscore_f1'] for r in q_results])
            qwen_prec_mean = statistics.mean([r['qwen_bertscore_precision'] for r in q_results])
            gemma_prec_mean = statistics.mean([r['gemma_bertscore_precision'] for r in q_results])
            qwen_rec_mean = statistics.mean([r['qwen_bertscore_recall'] for r in q_results])
            gemma_rec_mean = statistics.mean([r['gemma_bertscore_recall'] for r in q_results])
            
            comparison['per_question_type'][f'Q{q_idx+1}'] = {
                'qwen_mean_f1': qwen_f1_mean,
                'gemma_mean_f1': gemma_f1_mean,
                'f1_difference': qwen_f1_mean - gemma_f1_mean,
                'f1_qwen_better': qwen_f1_mean > gemma_f1_mean,
                'qwen_mean_precision': qwen_prec_mean,
                'gemma_mean_precision': gemma_prec_mean,
                'precision_difference': qwen_prec_mean - gemma_prec_mean,
                'precision_qwen_better': qwen_prec_mean > gemma_prec_mean,
                'qwen_mean_recall': qwen_rec_mean,
                'gemma_mean_recall': gemma_rec_mean,
                'recall_difference': qwen_rec_mean - gemma_rec_mean,
                'recall_qwen_better': qwen_rec_mean > gemma_rec_mean,
            }
    
    return {
        'qwen': {
            'overall': qwen_overall,
            'per_question_type': qwen_per_question,
            'per_compound': qwen_per_compound,
        },
        'gemma': {
            'overall': gemma_overall,
            'per_question_type': gemma_per_question,
            'per_compound': gemma_per_compound,
        },
        'answer_lengths': length_stats,
        'comparison': comparison,
        'total_pairs': len(results)
    }


def save_results(results: List[Dict], stats: Dict):
    """
    Save results to CSV and JSON files.
    """
    logger.info("Saving results...")
    
    # Save detailed CSV
    if pd is not None:
        df = pd.DataFrame(results)
        csv_path = RESULTS_DIR / "per_question_bertscore_qwen_gemma.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to: {csv_path}")
    else:
        # Fallback: save as JSON
        json_path = RESULTS_DIR / "per_question_bertscore_qwen_gemma.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detailed results to: {json_path}")
    
    # Save summary statistics
    summary_path = RESULTS_DIR / "summary_bertscore_qwen_gemma.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary metrics to: {summary_path}")
    
    # Save comparison statistics
    comparison_path = RESULTS_DIR / "comparison_bertscore_qwen_gemma.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(stats['comparison'], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved comparison metrics to: {comparison_path}")


def print_summary(stats: Dict):
    """
    Print summary statistics to console.
    """
    print("\n" + "="*70)
    print("BERTScore Summary - Qwen RAG Concise vs Gemma RAG Concise")
    print("="*70)
    print("Qwen: RAG Concise (character limits: Q1=600, Q2=1000, Q3=1800, Q4=2000)")
    print("Gemma: RAG Concise (character limits: Q1=600, Q2=1000, Q3=1800, Q4=2000)")
    print("OpenAI: Baseline (comprehensive)")
    print("="*70)
    
    print(f"\nTotal Question-Answer Pairs: {stats['total_pairs']}")
    
    # Qwen overall
    print("\n" + "-"*70)
    print("QWEN Overall Corpus BERTScore:")
    print("-"*70)
    qwen_overall = stats['qwen']['overall']
    for metric_type in ['precision', 'recall', 'f1']:
        s = qwen_overall[metric_type]
        print(f"  {metric_type.upper()}:")
        print(f"    Mean:   {s['mean']:.4f}")
        print(f"    Median: {s['median']:.4f}")
        print(f"    Std:    {s['std']:.4f}")
        print(f"    Range:  [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Gemma overall
    print("\n" + "-"*70)
    print("GEMMA Overall Corpus BERTScore:")
    print("-"*70)
    gemma_overall = stats['gemma']['overall']
    for metric_type in ['precision', 'recall', 'f1']:
        s = gemma_overall[metric_type]
        print(f"  {metric_type.upper()}:")
        print(f"    Mean:   {s['mean']:.4f}")
        print(f"    Median: {s['median']:.4f}")
        print(f"    Std:    {s['std']:.4f}")
        print(f"    Range:  [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Comparison
    print("\n" + "-"*70)
    print("COMPARISON: Qwen vs Gemma")
    print("-"*70)
    comp_f1 = stats['comparison']['overall']['qwen_vs_gemma_f1']
    comp_prec = stats['comparison']['overall']['qwen_vs_gemma_precision']
    comp_rec = stats['comparison']['overall']['qwen_vs_gemma_recall']
    
    print(f"  F1 Score:")
    print(f"    Qwen Mean:  {comp_f1['qwen_mean']:.4f}")
    print(f"    Gemma Mean: {comp_f1['gemma_mean']:.4f}")
    print(f"    Difference: {comp_f1['difference']:.4f} ({'+' if comp_f1['difference'] > 0 else ''}{comp_f1['difference']*100:.2f}%)")
    print(f"    Winner:     {'Qwen' if comp_f1['qwen_better'] else 'Gemma'}")
    
    print(f"\n  Precision:")
    print(f"    Qwen Mean:  {comp_prec['qwen_mean']:.4f}")
    print(f"    Gemma Mean: {comp_prec['gemma_mean']:.4f}")
    print(f"    Difference: {comp_prec['difference']:.4f} ({'+' if comp_prec['difference'] > 0 else ''}{comp_prec['difference']*100:.2f}%)")
    print(f"    Winner:     {'Qwen' if comp_prec['qwen_better'] else 'Gemma'}")
    
    print(f"\n  Recall:")
    print(f"    Qwen Mean:  {comp_rec['qwen_mean']:.4f}")
    print(f"    Gemma Mean: {comp_rec['gemma_mean']:.4f}")
    print(f"    Difference: {comp_rec['difference']:.4f} ({'+' if comp_rec['difference'] > 0 else ''}{comp_rec['difference']*100:.2f}%)")
    print(f"    Winner:     {'Qwen' if comp_rec['qwen_better'] else 'Gemma'}")
    
    # Per-question comparison
    print("\n" + "-"*70)
    print("Per-Question-Type Comparison (F1):")
    print("-"*70)
    for q_type in ['Q1', 'Q2', 'Q3', 'Q4']:
        if q_type in stats['comparison']['per_question_type']:
            q_comp = stats['comparison']['per_question_type'][q_type]
            qwen_q = stats['qwen']['per_question_type'][q_type]
            gemma_q = stats['gemma']['per_question_type'][q_type]
            print(f"  {q_type}:")
            print(f"    Qwen:  {qwen_q['mean_f1']:.4f} (median: {qwen_q['median_f1']:.4f})")
            print(f"    Gemma: {gemma_q['mean_f1']:.4f} (median: {gemma_q['median_f1']:.4f})")
            print(f"    Diff:  {q_comp['f1_difference']:.4f} ({'+' if q_comp['f1_difference'] > 0 else ''}{q_comp['f1_difference']*100:.2f}%)")
            print(f"    Winner: {'Qwen' if q_comp['f1_qwen_better'] else 'Gemma'}")
    
    # Answer length statistics
    print("\n" + "-"*70)
    print("Answer Length Statistics:")
    print("-"*70)
    lengths = stats['answer_lengths']
    for model in ['qwen', 'gemma', 'openai']:
        model_len = lengths[model]
        print(f"  {model.upper()}:")
        print(f"    Mean:   {model_len['mean']:.1f} chars")
        print(f"    Median: {model_len['median']:.1f} chars")
        print(f"    Range:  [{model_len['min']}, {model_len['max']}] chars")
    
    print("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate BERTScore for Qwen and Gemma RAG Concise vs OpenAI answers")
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
    
    logger.info("="*70)
    logger.info("BERTScore Calculator: Qwen RAG Concise vs Gemma RAG Concise")
    logger.info("="*70)
    logger.info(f"Qwen Directory: {QWEN_DIR}")
    logger.info(f"Gemma Directory: {GEMMA_DIR}")
    logger.info(f"OpenAI Directory: {OPENAI_DIR}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*70)
    
    # Load data
    qwen_data = load_model_answers(QWEN_DIR, "Qwen")
    gemma_data = load_model_answers(GEMMA_DIR, "Gemma")
    openai_data = load_openai_answers(OPENAI_DIR)
    
    # Match pairs
    matched_pairs = match_qa_pairs(qwen_data, gemma_data, openai_data)
    
    if not matched_pairs:
        logger.error("No matched pairs found. Exiting.")
        return
    
    # Calculate BERTScore
    results = calculate_bertscore_scores(
        matched_pairs,
        model_type=args.model,
        batch_size=args.batch_size,
        device=device
    )
    
    if not results:
        logger.error("BERTScore calculation failed. Exiting.")
        return
    
    # Aggregate statistics
    stats = aggregate_statistics(results)
    
    # Save results
    save_results(results, stats)
    
    # Print summary
    print_summary(stats)
    
    logger.info("BERTScore calculation complete!")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("Files: per_question_bertscore_qwen_gemma.csv, summary_bertscore_qwen_gemma.json, comparison_bertscore_qwen_gemma.json")


if __name__ == "__main__":
    main()

