#!/usr/bin/env python3
"""
BLEU Score Calculator: Qwen RAG Concise and Gemma RAG Concise vs OpenAI Baseline
Compares both Qwen and Gemma RAG Concise answers against OpenAI baseline answers
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
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
except ImportError:
    print("Error: nltk not installed. Install with: pip install nltk")
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

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)


def tokenize_answer(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize answer text for BLEU calculation.
    
    Args:
        text: Answer text to tokenize
        lowercase: Whether to lowercase tokens
    
    Returns:
        List of tokens
    """
    if not text or not text.strip():
        return []
    
    # Lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Tokenize using NLTK word tokenizer
    tokens = word_tokenize(text)
    
    # Remove empty tokens
    tokens = [t for t in tokens if t.strip()]
    
    return tokens


def calculate_sentence_bleu(candidate: str, reference: str) -> Dict[str, float]:
    """
    Calculate BLEU scores for a single answer pair.
    
    Args:
        candidate: Model-generated answer
        reference: OpenAI-generated answer (baseline)
    
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    candidate_tokens = tokenize_answer(candidate)
    reference_tokens = tokenize_answer(reference)
    
    # Handle empty cases
    if not candidate_tokens or not reference_tokens:
        return {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0
        }
    
    # Use smoothing to handle zero matches
    smoothing = SmoothingFunction().method1
    
    try:
        # BLEU-1 (unigram precision)
        bleu_1 = sentence_bleu(
            [reference_tokens], 
            candidate_tokens, 
            weights=(1, 0, 0, 0), 
            smoothing_function=smoothing
        )
        
        # BLEU-2 (bigram precision)
        bleu_2 = sentence_bleu(
            [reference_tokens], 
            candidate_tokens, 
            weights=(0.5, 0.5, 0, 0), 
            smoothing_function=smoothing
        )
        
        # BLEU-3 (trigram precision)
        bleu_3 = sentence_bleu(
            [reference_tokens], 
            candidate_tokens, 
            weights=(0.33, 0.33, 0.33, 0), 
            smoothing_function=smoothing
        )
        
        # BLEU-4 (4-gram precision, standard BLEU)
        bleu_4 = sentence_bleu(
            [reference_tokens], 
            candidate_tokens, 
            smoothing_function=smoothing
        )
        
        return {
            'bleu_1': float(bleu_1),
            'bleu_2': float(bleu_2),
            'bleu_3': float(bleu_3),
            'bleu_4': float(bleu_4)
        }
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0
        }


def extract_file_base(filename: str) -> str:
    """
    Extract base filename for matching.
    
    Examples:
        "37_Carbon_Dioxide__answers.json" -> "37_Carbon_Dioxide"
        "37_Carbon_Dioxide.json" -> "37_Carbon_Dioxide"
    """
    # Remove __answers suffix if present
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


def calculate_all_bleu_scores(matched_pairs: List[Dict]) -> List[Dict]:
    """
    Calculate BLEU scores for all matched pairs (both Qwen and Gemma vs OpenAI).
    
    Returns:
        List of dictionaries with BLEU scores and metadata
    """
    logger.info("Calculating BLEU scores...")
    results = []
    
    for i, pair in enumerate(matched_pairs):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(matched_pairs)} pairs...")
        
        # Calculate BLEU for Qwen vs OpenAI
        qwen_bleu = calculate_sentence_bleu(
            pair['qwen_answer'],
            pair['openai_answer']
        )
        
        # Calculate BLEU for Gemma vs OpenAI
        gemma_bleu = calculate_sentence_bleu(
            pair['gemma_answer'],
            pair['openai_answer']
        )
        
        result = {
            **pair,
            'qwen_bleu_1': qwen_bleu['bleu_1'],
            'qwen_bleu_2': qwen_bleu['bleu_2'],
            'qwen_bleu_3': qwen_bleu['bleu_3'],
            'qwen_bleu_4': qwen_bleu['bleu_4'],
            'gemma_bleu_1': gemma_bleu['bleu_1'],
            'gemma_bleu_2': gemma_bleu['bleu_2'],
            'gemma_bleu_3': gemma_bleu['bleu_3'],
            'gemma_bleu_4': gemma_bleu['bleu_4'],
        }
        results.append(result)
    
    logger.info(f"Completed BLEU calculation for {len(results)} pairs")
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
    
    # Extract BLEU scores for both models
    qwen_bleu_1 = [r['qwen_bleu_1'] for r in results]
    qwen_bleu_2 = [r['qwen_bleu_2'] for r in results]
    qwen_bleu_3 = [r['qwen_bleu_3'] for r in results]
    qwen_bleu_4 = [r['qwen_bleu_4'] for r in results]
    
    gemma_bleu_1 = [r['gemma_bleu_1'] for r in results]
    gemma_bleu_2 = [r['gemma_bleu_2'] for r in results]
    gemma_bleu_3 = [r['gemma_bleu_3'] for r in results]
    gemma_bleu_4 = [r['gemma_bleu_4'] for r in results]
    
    # Overall statistics
    qwen_overall = {
        'bleu_1': calc_stats(qwen_bleu_1),
        'bleu_2': calc_stats(qwen_bleu_2),
        'bleu_3': calc_stats(qwen_bleu_3),
        'bleu_4': calc_stats(qwen_bleu_4),
    }
    
    gemma_overall = {
        'bleu_1': calc_stats(gemma_bleu_1),
        'bleu_2': calc_stats(gemma_bleu_2),
        'bleu_3': calc_stats(gemma_bleu_3),
        'bleu_4': calc_stats(gemma_bleu_4),
    }
    
    # Per-question-type statistics (Q1, Q2, Q3, Q4)
    qwen_per_question = {}
    gemma_per_question = {}
    
    for q_idx in range(4):
        q_results = [r for r in results if r['question_index'] == q_idx]
        if q_results:
            qwen_q_bleu_4 = [r['qwen_bleu_4'] for r in q_results]
            gemma_q_bleu_4 = [r['gemma_bleu_4'] for r in q_results]
            
            qwen_per_question[f'Q{q_idx+1}'] = {
                'count': len(q_results),
                'mean_bleu_4': statistics.mean(qwen_q_bleu_4),
                'median_bleu_4': statistics.median(qwen_q_bleu_4),
                'std_bleu_4': statistics.stdev(qwen_q_bleu_4) if len(qwen_q_bleu_4) > 1 else 0.0,
            }
            
            gemma_per_question[f'Q{q_idx+1}'] = {
                'count': len(q_results),
                'mean_bleu_4': statistics.mean(gemma_q_bleu_4),
                'median_bleu_4': statistics.median(gemma_q_bleu_4),
                'std_bleu_4': statistics.stdev(gemma_q_bleu_4) if len(gemma_q_bleu_4) > 1 else 0.0,
            }
    
    # Per-compound statistics
    qwen_per_compound = {}
    gemma_per_compound = {}
    
    qwen_compounds = defaultdict(list)
    gemma_compounds = defaultdict(list)
    
    for r in results:
        qwen_compounds[r['file_base']].append(r['qwen_bleu_4'])
        gemma_compounds[r['file_base']].append(r['gemma_bleu_4'])
    
    for compound, scores in qwen_compounds.items():
        qwen_per_compound[compound] = {
            'mean_bleu_4': statistics.mean(scores),
            'count': len(scores)
        }
    
    for compound, scores in gemma_compounds.items():
        gemma_per_compound[compound] = {
            'mean_bleu_4': statistics.mean(scores),
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
            'qwen_vs_gemma_bleu_4': {
                'qwen_mean': statistics.mean(qwen_bleu_4),
                'gemma_mean': statistics.mean(gemma_bleu_4),
                'difference': statistics.mean(qwen_bleu_4) - statistics.mean(gemma_bleu_4),
                'qwen_better': statistics.mean(qwen_bleu_4) > statistics.mean(gemma_bleu_4),
            }
        },
        'per_question_type': {}
    }
    
    for q_idx in range(4):
        q_results = [r for r in results if r['question_index'] == q_idx]
        if q_results:
            qwen_mean = statistics.mean([r['qwen_bleu_4'] for r in q_results])
            gemma_mean = statistics.mean([r['gemma_bleu_4'] for r in q_results])
            comparison['per_question_type'][f'Q{q_idx+1}'] = {
                'qwen_mean_bleu_4': qwen_mean,
                'gemma_mean_bleu_4': gemma_mean,
                'difference': qwen_mean - gemma_mean,
                'qwen_better': qwen_mean > gemma_mean,
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
        csv_path = RESULTS_DIR / "per_question_bleu_qwen_gemma.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to: {csv_path}")
    else:
        # Fallback: save as JSON
        json_path = RESULTS_DIR / "per_question_bleu_qwen_gemma.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detailed results to: {json_path}")
    
    # Save summary statistics
    summary_path = RESULTS_DIR / "summary_metrics_qwen_gemma.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary metrics to: {summary_path}")
    
    # Save comparison statistics
    comparison_path = RESULTS_DIR / "comparison_bleu_qwen_gemma.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(stats['comparison'], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved comparison metrics to: {comparison_path}")


def print_summary(stats: Dict):
    """
    Print summary statistics to console.
    """
    print("\n" + "="*70)
    print("BLEU Score Summary - Qwen RAG Concise vs Gemma RAG Concise")
    print("="*70)
    print("Qwen: RAG Concise (character limits: Q1=600, Q2=1000, Q3=1800, Q4=2000)")
    print("Gemma: RAG Concise (character limits: Q1=600, Q2=1000, Q3=1800, Q4=2000)")
    print("OpenAI: Baseline (comprehensive)")
    print("="*70)
    
    print(f"\nTotal Question-Answer Pairs: {stats['total_pairs']}")
    
    # Qwen overall
    print("\n" + "-"*70)
    print("QWEN Overall Corpus BLEU Scores:")
    print("-"*70)
    qwen_overall = stats['qwen']['overall']
    for bleu_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
        s = qwen_overall[bleu_type]
        print(f"  {bleu_type.upper()}:")
        print(f"    Mean:   {s['mean']:.4f}")
        print(f"    Median: {s['median']:.4f}")
        print(f"    Std:    {s['std']:.4f}")
        print(f"    Range:  [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Gemma overall
    print("\n" + "-"*70)
    print("GEMMA Overall Corpus BLEU Scores:")
    print("-"*70)
    gemma_overall = stats['gemma']['overall']
    for bleu_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
        s = gemma_overall[bleu_type]
        print(f"  {bleu_type.upper()}:")
        print(f"    Mean:   {s['mean']:.4f}")
        print(f"    Median: {s['median']:.4f}")
        print(f"    Std:    {s['std']:.4f}")
        print(f"    Range:  [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Comparison
    print("\n" + "-"*70)
    print("COMPARISON: Qwen vs Gemma (BLEU-4)")
    print("-"*70)
    comp = stats['comparison']['overall']['qwen_vs_gemma_bleu_4']
    print(f"  Qwen Mean BLEU-4:  {comp['qwen_mean']:.4f}")
    print(f"  Gemma Mean BLEU-4: {comp['gemma_mean']:.4f}")
    print(f"  Difference:        {comp['difference']:.4f} ({'+' if comp['difference'] > 0 else ''}{comp['difference']*100:.2f}%)")
    print(f"  Winner:            {'Qwen' if comp['qwen_better'] else 'Gemma'}")
    
    # Per-question comparison
    print("\n" + "-"*70)
    print("Per-Question-Type Comparison (BLEU-4):")
    print("-"*70)
    for q_type in ['Q1', 'Q2', 'Q3', 'Q4']:
        if q_type in stats['comparison']['per_question_type']:
            q_comp = stats['comparison']['per_question_type'][q_type]
            qwen_q = stats['qwen']['per_question_type'][q_type]
            gemma_q = stats['gemma']['per_question_type'][q_type]
            print(f"  {q_type}:")
            print(f"    Qwen:  {qwen_q['mean_bleu_4']:.4f} (median: {qwen_q['median_bleu_4']:.4f})")
            print(f"    Gemma: {gemma_q['mean_bleu_4']:.4f} (median: {gemma_q['median_bleu_4']:.4f})")
            print(f"    Diff:  {q_comp['difference']:.4f} ({'+' if q_comp['difference'] > 0 else ''}{q_comp['difference']*100:.2f}%)")
            print(f"    Winner: {'Qwen' if q_comp['qwen_better'] else 'Gemma'}")
    
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
    logger.info("BLEU Score Calculator: Qwen RAG Concise vs Gemma RAG Concise")
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
    
    # Calculate BLEU scores
    results = calculate_all_bleu_scores(matched_pairs)
    
    # Aggregate statistics
    stats = aggregate_statistics(results)
    
    # Save results
    save_results(results, stats)
    
    # Print summary
    print_summary(stats)
    
    logger.info("BLEU score calculation complete!")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("Files: per_question_bleu_qwen_gemma.csv, summary_metrics_qwen_gemma.json, comparison_bleu_qwen_gemma.json")


if __name__ == "__main__":
    main()

