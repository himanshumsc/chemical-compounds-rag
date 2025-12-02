#!/usr/bin/env python3
"""
BLEU Score Calculator: QWEN RAG vs OpenAI Baseline
Compares QWEN RAG-generated answers (candidate) against OpenAI-generated answers (reference)
"""

import json
import re
import sys
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

# Directories
QWEN_RAG_DIR = Path("/home/himanshu/dev/output/qwen_rag")
OPENAI_BASELINE_DIR = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive")
RESULTS_DIR = Path(__file__).parent.parent / "results"
VIS_DIR = Path(__file__).parent.parent / "visualizations"

# Create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
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
        candidate: QWEN RAG-generated answer
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
        "1_13-Butadiene__answers.json" -> "1_13-Butadiene"
        "1_13-Butadiene.json" -> "1_13-Butadiene"
    """
    # Remove __answers suffix if present
    base = filename.replace("__answers.json", "").replace(".json", "")
    return base


def load_qwen_rag_answers(qwen_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load all QWEN RAG answer files.
    
    Returns:
        Dictionary mapping file_base -> list of answers
    """
    qwen_data = {}
    qwen_files = sorted(qwen_dir.glob("*__answers.json"))
    
    print(f"Loading {len(qwen_files)} QWEN RAG answer files...")
    
    for qwen_file in qwen_files:
        try:
            with open(qwen_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_base = extract_file_base(qwen_file.name)
            qwen_data[file_base] = data.get('answers', [])
            
        except Exception as e:
            print(f"Error loading {qwen_file.name}: {e}")
    
    print(f"Loaded {len(qwen_data)} QWEN RAG files")
    return qwen_data


def load_openai_baseline_answers(openai_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load all OpenAI baseline answer files.
    
    Returns:
        Dictionary mapping file_base -> list of qa_pairs
    """
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
    """
    Match QWEN RAG and OpenAI baseline answers by file and question index.
    
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
                print(f"     QWEN RAG: {qwen_question[:50]}...")
                print(f"     OpenAI: {openai_question[:50]}...")
            
            matched_pairs.append({
                'file_base': file_base,
                'question_index': idx,
                'question': openai_question or qwen_question,
                'qwen_rag_answer': qwen_answer,
                'openai_answer': openai_answer,
                'qwen_rag_length': len(qwen_answer),
                'openai_length': len(openai_answer),
                'qwen_rag_tokens': len(qwen_answer.split()),
                'openai_tokens': len(openai_answer.split()),
            })
    
    print(f"\nMatched {len(matched_pairs)} question-answer pairs")
    if unmatched_qwen:
        print(f"  ⚠️  {len(unmatched_qwen)} QWEN RAG files without OpenAI matches")
    if unmatched_openai:
        print(f"  ⚠️  {len(unmatched_openai)} OpenAI files without QWEN RAG matches")
    
    return matched_pairs


def calculate_all_bleu_scores(matched_pairs: List[Dict]) -> List[Dict]:
    """
    Calculate BLEU scores for all matched pairs.
    
    Returns:
        List of dictionaries with BLEU scores and metadata
    """
    print("\nCalculating BLEU scores...")
    results = []
    
    for i, pair in enumerate(matched_pairs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(matched_pairs)} pairs...")
        
        bleu_scores = calculate_sentence_bleu(
            pair['qwen_rag_answer'],
            pair['openai_answer']
        )
        
        result = {
            **pair,
            **bleu_scores
        }
        results.append(result)
    
    print(f"Completed BLEU calculation for {len(results)} pairs")
    return results


def aggregate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate aggregated statistics.
    
    Returns:
        Dictionary with various aggregated metrics
    """
    print("\nCalculating aggregated statistics...")
    
    # Extract BLEU scores
    bleu_1_scores = [r['bleu_1'] for r in results]
    bleu_2_scores = [r['bleu_2'] for r in results]
    bleu_3_scores = [r['bleu_3'] for r in results]
    bleu_4_scores = [r['bleu_4'] for r in results]
    
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
        'bleu_1': calc_stats(bleu_1_scores),
        'bleu_2': calc_stats(bleu_2_scores),
        'bleu_3': calc_stats(bleu_3_scores),
        'bleu_4': calc_stats(bleu_4_scores),
    }
    
    # Per-question-type statistics (Q1, Q2, Q3, Q4)
    per_question_stats = {}
    for q_idx in range(4):
        q_results = [r for r in results if r['question_index'] == q_idx]
        if q_results:
            q_bleu_4 = [r['bleu_4'] for r in q_results]
            per_question_stats[f'Q{q_idx+1}'] = {
                'count': len(q_results),
                'mean_bleu_4': statistics.mean(q_bleu_4),
                'median_bleu_4': statistics.median(q_bleu_4),
                'std_bleu_4': statistics.stdev(q_bleu_4) if len(q_bleu_4) > 1 else 0.0,
            }
    
    # Per-compound statistics
    per_compound_stats = {}
    compounds = defaultdict(list)
    for r in results:
        compounds[r['file_base']].append(r['bleu_4'])
    
    for compound, scores in compounds.items():
        per_compound_stats[compound] = {
            'mean_bleu_4': statistics.mean(scores),
            'count': len(scores)
        }
    
    # Answer length statistics (characters and tokens)
    qwen_lengths = [r['qwen_rag_length'] for r in results]
    openai_lengths = [r['openai_length'] for r in results]
    qwen_tokens = [r['qwen_rag_tokens'] for r in results]
    openai_tokens = [r['openai_tokens'] for r in results]
    
    length_stats = {
        'qwen_rag': {
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


def save_results(results: List[Dict], stats: Dict):
    """
    Save results to CSV and JSON files.
    """
    print("\nSaving results...")
    
    # Save detailed CSV
    if pd is not None:
        df = pd.DataFrame(results)
        csv_path = RESULTS_DIR / "per_question_bleu_rag.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Saved detailed results to: {csv_path}")
    else:
        # Fallback: save as JSON
        json_path = RESULTS_DIR / "per_question_bleu_rag.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved detailed results to: {json_path}")
    
    # Save summary statistics
    summary_path = RESULTS_DIR / "summary_metrics_rag.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved summary metrics to: {summary_path}")


def print_summary(stats: Dict):
    """
    Print summary statistics to console.
    """
    print("\n" + "="*70)
    print("BLEU Score Summary: QWEN RAG vs OpenAI Baseline")
    print("="*70)
    
    overall = stats['overall']
    print(f"\nTotal Question-Answer Pairs: {stats['total_pairs']}")
    
    print("\nOverall Corpus BLEU Scores:")
    for bleu_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
        s = overall[bleu_type]
        print(f"  {bleu_type.upper()}:")
        print(f"    Mean:   {s['mean']:.4f}")
        print(f"    Median: {s['median']:.4f}")
        print(f"    Std:    {s['std']:.4f}")
        print(f"    Range:  [{s['min']:.4f}, {s['max']:.4f}]")
    
    print("\nPer-Question-Type BLEU-4 Scores:")
    for q_type, q_stats in stats['per_question_type'].items():
        print(f"  {q_type}:")
        print(f"    Mean:   {q_stats['mean_bleu_4']:.4f}")
        print(f"    Median: {q_stats['median_bleu_4']:.4f}")
        print(f"    Count:  {q_stats['count']}")
    
    print("\nAnswer Length Statistics:")
    qwen_len = stats['answer_lengths']['qwen_rag']
    openai_len = stats['answer_lengths']['openai']
    ratio = stats['answer_lengths']['length_ratio']
    
    print(f"  QWEN RAG (candidate):")
    print(f"    Mean chars:   {qwen_len['mean_chars']:.1f}")
    print(f"    Median chars: {qwen_len['median_chars']:.1f}")
    print(f"    Mean tokens:  {qwen_len['mean_tokens']:.1f}")
    print(f"    Range chars:  [{qwen_len['min_chars']}, {qwen_len['max_chars']}]")
    print(f"  OpenAI (reference):")
    print(f"    Mean chars:   {openai_len['mean_chars']:.1f}")
    print(f"    Median chars: {openai_len['median_chars']:.1f}")
    print(f"    Mean tokens:  {openai_len['mean_tokens']:.1f}")
    print(f"    Range chars:  [{openai_len['min_chars']}, {openai_len['max_chars']}]")
    print(f"  Length Ratio (QWEN RAG / OpenAI):")
    print(f"    Mean chars ratio:  {ratio['mean_chars_ratio']:.2f}x")
    print(f"    Mean tokens ratio: {ratio['mean_tokens_ratio']:.2f}x")
    
    print("\n" + "="*70)


def main():
    """Main execution function"""
    print("="*70)
    print("BLEU Score Calculator: QWEN RAG vs OpenAI Baseline")
    print("="*70)
    
    # Load data
    qwen_data = load_qwen_rag_answers(QWEN_RAG_DIR)
    openai_data = load_openai_baseline_answers(OPENAI_BASELINE_DIR)
    
    # Match pairs
    matched_pairs = match_qa_pairs(qwen_data, openai_data)
    
    if not matched_pairs:
        print("❌ No matched pairs found. Exiting.")
        return
    
    # Calculate BLEU scores
    results = calculate_all_bleu_scores(matched_pairs)
    
    # Aggregate statistics
    stats = aggregate_statistics(results)
    
    # Save results
    save_results(results, stats)
    
    # Print summary
    print_summary(stats)
    
    print(f"\n✅ BLEU score calculation complete!")
    print(f"   Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

