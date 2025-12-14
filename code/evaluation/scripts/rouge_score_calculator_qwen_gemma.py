#!/usr/bin/env python3
"""
ROUGE Score Calculator: Qwen RAG Concise and Gemma RAG Concise vs OpenAI Baseline
Compares both Qwen and Gemma RAG Concise answers against OpenAI baseline answers
using ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics
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
    from rouge_score import rouge_scorer
except ImportError:
    print("Error: rouge-score not installed. Install with: pip install rouge-score")
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

# ROUGE scorer configuration
ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']


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


def calculate_rouge_scores(candidate: str, reference: str, scorer: rouge_scorer.RougeScorer) -> Dict[str, Dict[str, float]]:
    """
    Calculate ROUGE scores for a single answer pair.
    
    Args:
        candidate: Model-generated answer
        reference: OpenAI-generated answer (baseline)
        scorer: ROUGE scorer instance
    
    Returns:
        Dictionary with ROUGE scores (precision, recall, fmeasure) for each type
    """
    # Handle empty cases
    if not candidate or not candidate.strip():
        candidate = " "
    if not reference or not reference.strip():
        reference = " "
    
    scores = scorer.score(reference, candidate)
    
    result = {}
    for rouge_type in ROUGE_TYPES:
        if rouge_type in scores:
            score = scores[rouge_type]
            result[rouge_type] = {
                'precision': score.precision,
                'recall': score.recall,
                'fmeasure': score.fmeasure
            }
        else:
            result[rouge_type] = {
                'precision': 0.0,
                'recall': 0.0,
                'fmeasure': 0.0
            }
    
    return result


def calculate_all_rouge_scores(matched_pairs: List[Dict]) -> List[Dict]:
    """
    Calculate ROUGE scores for all matched pairs (both Qwen and Gemma vs OpenAI).
    
    Returns:
        List of dictionaries with ROUGE scores and metadata
    """
    logger.info("Calculating ROUGE scores...")
    logger.info("Initializing ROUGE scorer...")
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    
    results = []
    
    for i, pair in enumerate(matched_pairs):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(matched_pairs)} pairs...")
        
        # Calculate ROUGE for Qwen vs OpenAI
        qwen_rouge = calculate_rouge_scores(
            pair['qwen_answer'],
            pair['openai_answer'],
            scorer
        )
        
        # Calculate ROUGE for Gemma vs OpenAI
        gemma_rouge = calculate_rouge_scores(
            pair['gemma_answer'],
            pair['openai_answer'],
            scorer
        )
        
        result = {
            **pair,
        }
        
        # Add Qwen ROUGE scores
        for rouge_type in ROUGE_TYPES:
            result[f'qwen_{rouge_type}_precision'] = qwen_rouge[rouge_type]['precision']
            result[f'qwen_{rouge_type}_recall'] = qwen_rouge[rouge_type]['recall']
            result[f'qwen_{rouge_type}_fmeasure'] = qwen_rouge[rouge_type]['fmeasure']
        
        # Add Gemma ROUGE scores
        for rouge_type in ROUGE_TYPES:
            result[f'gemma_{rouge_type}_precision'] = gemma_rouge[rouge_type]['precision']
            result[f'gemma_{rouge_type}_recall'] = gemma_rouge[rouge_type]['recall']
            result[f'gemma_{rouge_type}_fmeasure'] = gemma_rouge[rouge_type]['fmeasure']
        
        results.append(result)
    
    logger.info(f"Completed ROUGE calculation for {len(results)} pairs")
    return results


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
    
    # Extract ROUGE scores for both models
    qwen_stats = {}
    gemma_stats = {}
    
    for rouge_type in ROUGE_TYPES:
        qwen_precision = [r[f'qwen_{rouge_type}_precision'] for r in results]
        qwen_recall = [r[f'qwen_{rouge_type}_recall'] for r in results]
        qwen_fmeasure = [r[f'qwen_{rouge_type}_fmeasure'] for r in results]
        
        gemma_precision = [r[f'gemma_{rouge_type}_precision'] for r in results]
        gemma_recall = [r[f'gemma_{rouge_type}_recall'] for r in results]
        gemma_fmeasure = [r[f'gemma_{rouge_type}_fmeasure'] for r in results]
        
        qwen_stats[rouge_type] = {
            'precision': calc_stats(qwen_precision),
            'recall': calc_stats(qwen_recall),
            'fmeasure': calc_stats(qwen_fmeasure),
        }
        
        gemma_stats[rouge_type] = {
            'precision': calc_stats(gemma_precision),
            'recall': calc_stats(gemma_recall),
            'fmeasure': calc_stats(gemma_fmeasure),
        }
    
    # Per-question-type statistics (Q1, Q2, Q3, Q4) - using F-measure
    qwen_per_question = {}
    gemma_per_question = {}
    
    for q_idx in range(4):
        q_results = [r for r in results if r['question_index'] == q_idx]
        if q_results:
            qwen_q_rouge1 = [r['qwen_rouge1_fmeasure'] for r in q_results]
            qwen_q_rouge2 = [r['qwen_rouge2_fmeasure'] for r in q_results]
            qwen_q_rougeL = [r['qwen_rougeL_fmeasure'] for r in q_results]
            qwen_q_rougeLsum = [r['qwen_rougeLsum_fmeasure'] for r in q_results]
            
            gemma_q_rouge1 = [r['gemma_rouge1_fmeasure'] for r in q_results]
            gemma_q_rouge2 = [r['gemma_rouge2_fmeasure'] for r in q_results]
            gemma_q_rougeL = [r['gemma_rougeL_fmeasure'] for r in q_results]
            gemma_q_rougeLsum = [r['gemma_rougeLsum_fmeasure'] for r in q_results]
            
            qwen_per_question[f'Q{q_idx+1}'] = {
                'count': len(q_results),
                'mean_rouge1_fmeasure': statistics.mean(qwen_q_rouge1),
                'mean_rouge2_fmeasure': statistics.mean(qwen_q_rouge2),
                'mean_rougeL_fmeasure': statistics.mean(qwen_q_rougeL),
                'mean_rougeLsum_fmeasure': statistics.mean(qwen_q_rougeLsum),
                'median_rouge1_fmeasure': statistics.median(qwen_q_rouge1),
                'median_rouge2_fmeasure': statistics.median(qwen_q_rouge2),
                'median_rougeL_fmeasure': statistics.median(qwen_q_rougeL),
                'median_rougeLsum_fmeasure': statistics.median(qwen_q_rougeLsum),
            }
            
            gemma_per_question[f'Q{q_idx+1}'] = {
                'count': len(q_results),
                'mean_rouge1_fmeasure': statistics.mean(gemma_q_rouge1),
                'mean_rouge2_fmeasure': statistics.mean(gemma_q_rouge2),
                'mean_rougeL_fmeasure': statistics.mean(gemma_q_rougeL),
                'mean_rougeLsum_fmeasure': statistics.mean(gemma_q_rougeLsum),
                'median_rouge1_fmeasure': statistics.median(gemma_q_rouge1),
                'median_rouge2_fmeasure': statistics.median(gemma_q_rouge2),
                'median_rougeL_fmeasure': statistics.median(gemma_q_rougeL),
                'median_rougeLsum_fmeasure': statistics.median(gemma_q_rougeLsum),
            }
    
    # Per-compound statistics
    qwen_per_compound = {}
    gemma_per_compound = {}
    
    qwen_compounds = defaultdict(list)
    gemma_compounds = defaultdict(list)
    
    for r in results:
        qwen_compounds[r['file_base']].append(r['qwen_rouge1_fmeasure'])
        gemma_compounds[r['file_base']].append(r['gemma_rouge1_fmeasure'])
    
    for compound, scores in qwen_compounds.items():
        qwen_per_compound[compound] = {
            'mean_rouge1_fmeasure': statistics.mean(scores),
            'count': len(scores)
        }
    
    for compound, scores in gemma_compounds.items():
        gemma_per_compound[compound] = {
            'mean_rouge1_fmeasure': statistics.mean(scores),
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
    
    # Comparison statistics (using F-measure for comparison)
    comparison = {
        'overall': {},
        'per_question_type': {}
    }
    
    for rouge_type in ROUGE_TYPES:
        qwen_fmeasure = [r[f'qwen_{rouge_type}_fmeasure'] for r in results]
        gemma_fmeasure = [r[f'gemma_{rouge_type}_fmeasure'] for r in results]
        
        comparison['overall'][f'qwen_vs_gemma_{rouge_type}'] = {
            'qwen_mean': statistics.mean(qwen_fmeasure),
            'gemma_mean': statistics.mean(gemma_fmeasure),
            'difference': statistics.mean(qwen_fmeasure) - statistics.mean(gemma_fmeasure),
            'qwen_better': statistics.mean(qwen_fmeasure) > statistics.mean(gemma_fmeasure),
        }
    
    for q_idx in range(4):
        q_results = [r for r in results if r['question_index'] == q_idx]
        if q_results:
            qwen_rouge1 = statistics.mean([r['qwen_rouge1_fmeasure'] for r in q_results])
            gemma_rouge1 = statistics.mean([r['gemma_rouge1_fmeasure'] for r in q_results])
            qwen_rouge2 = statistics.mean([r['qwen_rouge2_fmeasure'] for r in q_results])
            gemma_rouge2 = statistics.mean([r['gemma_rouge2_fmeasure'] for r in q_results])
            qwen_rougeL = statistics.mean([r['qwen_rougeL_fmeasure'] for r in q_results])
            gemma_rougeL = statistics.mean([r['gemma_rougeL_fmeasure'] for r in q_results])
            
            comparison['per_question_type'][f'Q{q_idx+1}'] = {
                'qwen_mean_rouge1_fmeasure': qwen_rouge1,
                'gemma_mean_rouge1_fmeasure': gemma_rouge1,
                'rouge1_difference': qwen_rouge1 - gemma_rouge1,
                'rouge1_qwen_better': qwen_rouge1 > gemma_rouge1,
                'qwen_mean_rouge2_fmeasure': qwen_rouge2,
                'gemma_mean_rouge2_fmeasure': gemma_rouge2,
                'rouge2_difference': qwen_rouge2 - gemma_rouge2,
                'rouge2_qwen_better': qwen_rouge2 > gemma_rouge2,
                'qwen_mean_rougeL_fmeasure': qwen_rougeL,
                'gemma_mean_rougeL_fmeasure': gemma_rougeL,
                'rougeL_difference': qwen_rougeL - gemma_rougeL,
                'rougeL_qwen_better': qwen_rougeL > gemma_rougeL,
            }
    
    return {
        'qwen': {
            'overall': qwen_stats,
            'per_question_type': qwen_per_question,
            'per_compound': qwen_per_compound,
        },
        'gemma': {
            'overall': gemma_stats,
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
        csv_path = RESULTS_DIR / "per_question_rouge_qwen_gemma.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to: {csv_path}")
    else:
        # Fallback: save as JSON
        json_path = RESULTS_DIR / "per_question_rouge_qwen_gemma.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detailed results to: {json_path}")
    
    # Save summary statistics
    summary_path = RESULTS_DIR / "summary_rouge_qwen_gemma.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary metrics to: {summary_path}")
    
    # Save comparison statistics
    comparison_path = RESULTS_DIR / "comparison_rouge_qwen_gemma.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(stats['comparison'], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved comparison metrics to: {comparison_path}")


def print_summary(stats: Dict):
    """
    Print summary statistics to console.
    """
    print("\n" + "="*70)
    print("ROUGE Score Summary - Qwen RAG Concise vs Gemma RAG Concise")
    print("="*70)
    print("Qwen: RAG Concise (character limits: Q1=600, Q2=1000, Q3=1800, Q4=2000)")
    print("Gemma: RAG Concise (character limits: Q1=600, Q2=1000, Q3=1800, Q4=2000)")
    print("OpenAI: Baseline (comprehensive)")
    print("="*70)
    
    print(f"\nTotal Question-Answer Pairs: {stats['total_pairs']}")
    
    # Qwen overall
    print("\n" + "-"*70)
    print("QWEN Overall Corpus ROUGE Scores (F-measure):")
    print("-"*70)
    qwen_overall = stats['qwen']['overall']
    for rouge_type in ROUGE_TYPES:
        s = qwen_overall[rouge_type]['fmeasure']
        print(f"  {rouge_type.upper()}:")
        print(f"    Mean:   {s['mean']:.4f}")
        print(f"    Median: {s['median']:.4f}")
        print(f"    Std:    {s['std']:.4f}")
        print(f"    Range:  [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Gemma overall
    print("\n" + "-"*70)
    print("GEMMA Overall Corpus ROUGE Scores (F-measure):")
    print("-"*70)
    gemma_overall = stats['gemma']['overall']
    for rouge_type in ROUGE_TYPES:
        s = gemma_overall[rouge_type]['fmeasure']
        print(f"  {rouge_type.upper()}:")
        print(f"    Mean:   {s['mean']:.4f}")
        print(f"    Median: {s['median']:.4f}")
        print(f"    Std:    {s['std']:.4f}")
        print(f"    Range:  [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Comparison
    print("\n" + "-"*70)
    print("COMPARISON: Qwen vs Gemma (F-measure)")
    print("-"*70)
    for rouge_type in ROUGE_TYPES:
        comp = stats['comparison']['overall'][f'qwen_vs_gemma_{rouge_type}']
        print(f"  {rouge_type.upper()}:")
        print(f"    Qwen Mean:  {comp['qwen_mean']:.4f}")
        print(f"    Gemma Mean: {comp['gemma_mean']:.4f}")
        print(f"    Difference: {comp['difference']:.4f} ({'+' if comp['difference'] > 0 else ''}{comp['difference']*100:.2f}%)")
        print(f"    Winner:     {'Qwen' if comp['qwen_better'] else 'Gemma'}")
    
    # Per-question comparison
    print("\n" + "-"*70)
    print("Per-Question-Type Comparison (ROUGE-1 F-measure):")
    print("-"*70)
    for q_type in ['Q1', 'Q2', 'Q3', 'Q4']:
        if q_type in stats['comparison']['per_question_type']:
            q_comp = stats['comparison']['per_question_type'][q_type]
            qwen_q = stats['qwen']['per_question_type'][q_type]
            gemma_q = stats['gemma']['per_question_type'][q_type]
            print(f"  {q_type}:")
            print(f"    Qwen:  {qwen_q['mean_rouge1_fmeasure']:.4f} (median: {qwen_q['median_rouge1_fmeasure']:.4f})")
            print(f"    Gemma: {gemma_q['mean_rouge1_fmeasure']:.4f} (median: {gemma_q['median_rouge1_fmeasure']:.4f})")
            print(f"    Diff:  {q_comp['rouge1_difference']:.4f} ({'+' if q_comp['rouge1_difference'] > 0 else ''}{q_comp['rouge1_difference']*100:.2f}%)")
            print(f"    Winner: {'Qwen' if q_comp['rouge1_qwen_better'] else 'Gemma'}")
    
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
    """Main execution function"""
    logger.info("="*70)
    logger.info("ROUGE Score Calculator: Qwen RAG Concise vs Gemma RAG Concise")
    logger.info("="*70)
    logger.info(f"Qwen Directory: {QWEN_DIR}")
    logger.info(f"Gemma Directory: {GEMMA_DIR}")
    logger.info(f"OpenAI Directory: {OPENAI_DIR}")
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
    
    # Calculate ROUGE scores
    results = calculate_all_rouge_scores(matched_pairs)
    
    # Aggregate statistics
    stats = aggregate_statistics(results)
    
    # Save results
    save_results(results, stats)
    
    # Print summary
    print_summary(stats)
    
    logger.info("ROUGE score calculation complete!")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("Files: per_question_rouge_qwen_gemma.csv, summary_rouge_qwen_gemma.json, comparison_rouge_qwen_gemma.json")


if __name__ == "__main__":
    main()

