#!/usr/bin/env python3
"""
Analyze the comprehensive QA update output:
- Token usage statistics
- Answer length statistics
- Comparison with token limit
- Failed files analysis
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

def count_tokens_approx(text):
    """
    Approximate token count.
    OpenAI uses tiktoken, but for approximation:
    - 1 token ≈ 0.75 words
    - 1 token ≈ 4 characters
    Using word-based approximation as it's more accurate for English text.
    """
    if not text:
        return 0
    words = len(text.split())
    # Conservative estimate: 1.3 tokens per word
    return int(words * 1.3)

def analyze_qa_files(qa_dir):
    """Analyze all QA files in the directory."""
    qa_dir = Path(qa_dir)
    files = sorted(list(qa_dir.glob('*.json')))
    
    stats = {
        'total_files': len(files),
        'files_with_qa': 0,
        'all_answers': [],
        'q2_answers': [],
        'q3_answers': [],
        'q4_answers': [],
        'updated_files': 0,
        'files_by_compound': defaultdict(list),
    }
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Check if file was updated
            if data.get('updated_by') == 'comprehensive_text_generator':
                stats['updated_files'] += 1
                
            qa_pairs = data.get('qa_pairs', [])
            if len(qa_pairs) >= 4:
                stats['files_with_qa'] += 1
                compound_name = data.get('compound_name', 'Unknown')
                
                # Analyze Q2, Q3, Q4 (indices 1, 2, 3)
                for idx, qa in enumerate(qa_pairs[1:4], 2):
                    answer = qa.get('answer', '')
                    if answer:
                        length = len(answer)
                        tokens = count_tokens_approx(answer)
                        
                        answer_data = {
                            'file': f.name,
                            'compound': compound_name,
                            'question_num': idx,
                            'length': length,
                            'tokens': tokens,
                            'words': len(answer.split()),
                        }
                        
                        stats['all_answers'].append(answer_data)
                        
                        if idx == 2:
                            stats['q2_answers'].append(answer_data)
                        elif idx == 3:
                            stats['q3_answers'].append(answer_data)
                        elif idx == 4:
                            stats['q4_answers'].append(answer_data)
                            
                        stats['files_by_compound'][compound_name].append(answer_data)
                        
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            continue
    
    return stats

def print_statistics(stats):
    """Print detailed statistics."""
    print("="*70)
    print("COMPREHENSIVE QA UPDATE - OUTPUT ANALYSIS")
    print("="*70)
    
    print(f"\n📁 File Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Files with QA pairs: {stats['files_with_qa']}")
    print(f"  Files updated: {stats['updated_files']}")
    
    if not stats['all_answers']:
        print("\nNo answers found to analyze.")
        return
    
    print(f"\n📊 Overall Answer Statistics (Q2, Q3, Q4):")
    all_lengths = [a['length'] for a in stats['all_answers']]
    all_tokens = [a['tokens'] for a in stats['all_answers']]
    all_words = [a['words'] for a in stats['all_answers']]
    
    print(f"  Total answers analyzed: {len(stats['all_answers'])}")
    print(f"\n  Character Length:")
    print(f"    Average: {statistics.mean(all_lengths):.0f} chars")
    print(f"    Median: {statistics.median(all_lengths):.0f} chars")
    print(f"    Min: {min(all_lengths)} chars")
    print(f"    Max: {max(all_lengths)} chars")
    print(f"    Std Dev: {statistics.stdev(all_lengths):.0f} chars")
    
    print(f"\n  Token Count (approximate):")
    print(f"    Average: {statistics.mean(all_tokens):.0f} tokens")
    print(f"    Median: {statistics.median(all_tokens):.0f} tokens")
    print(f"    Min: {min(all_tokens)} tokens")
    print(f"    Max: {max(all_tokens)} tokens")
    print(f"    Std Dev: {statistics.stdev(all_tokens):.0f} tokens")
    
    print(f"\n  Word Count:")
    print(f"    Average: {statistics.mean(all_words):.0f} words")
    print(f"    Median: {statistics.median(all_words):.0f} words")
    print(f"    Min: {min(all_words)} words")
    print(f"    Max: {max(all_words)} words")
    
    # Token limit analysis
    max_token_limit = 500
    over_limit = [a for a in stats['all_answers'] if a['tokens'] > max_token_limit]
    near_limit = [a for a in stats['all_answers'] if 400 <= a['tokens'] <= max_token_limit]
    
    print(f"\n  Token Limit Analysis (limit: {max_token_limit}):")
    print(f"    Under limit: {len(stats['all_answers']) - len(over_limit)} ({100*(len(stats['all_answers']) - len(over_limit))/len(stats['all_answers']):.1f}%)")
    print(f"    Near limit (400-500): {len(near_limit)} ({100*len(near_limit)/len(stats['all_answers']):.1f}%)")
    print(f"    Over limit: {len(over_limit)} ({100*len(over_limit)/len(stats['all_answers']):.1f}%)")
    
    if over_limit:
        print(f"\n    ⚠️  Answers over limit:")
        for a in sorted(over_limit, key=lambda x: x['tokens'], reverse=True)[:5]:
            print(f"      {a['compound']} Q{a['question_num']}: {a['tokens']} tokens ({a['length']} chars)")
    
    # Per-question statistics
    for q_num, q_answers in [(2, stats['q2_answers']), (3, stats['q3_answers']), (4, stats['q4_answers'])]:
        if q_answers:
            print(f"\n📝 Question {q_num} Statistics:")
            q_lengths = [a['length'] for a in q_answers]
            q_tokens = [a['tokens'] for a in q_answers]
            q_words = [a['words'] for a in q_answers]
            
            print(f"  Total answers: {len(q_answers)}")
            print(f"  Average: {statistics.mean(q_lengths):.0f} chars, {statistics.mean(q_tokens):.0f} tokens, {statistics.mean(q_words):.0f} words")
            print(f"  Median: {statistics.median(q_lengths):.0f} chars, {statistics.median(q_tokens):.0f} tokens, {statistics.median(q_words):.0f} words")
            print(f"  Range: {min(q_lengths)}-{max(q_lengths)} chars, {min(q_tokens)}-{max(q_tokens)} tokens")
            
            # Find longest answer
            longest = max(q_answers, key=lambda x: x['tokens'])
            print(f"  Longest: {longest['compound']} - {longest['tokens']} tokens ({longest['length']} chars)")
    
    # Top compounds by answer length
    print(f"\n📈 Top 10 Compounds by Average Answer Length:")
    compound_avgs = {}
    for compound, answers in stats['files_by_compound'].items():
        if answers:
            avg_tokens = statistics.mean([a['tokens'] for a in answers])
            compound_avgs[compound] = avg_tokens
    
    for compound, avg_tokens in sorted(compound_avgs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {compound}: {avg_tokens:.0f} tokens avg")

def main():
    qa_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive")
    
    if not qa_dir.exists():
        print(f"Error: Directory not found: {qa_dir}")
        return
    
    print("Analyzing comprehensive QA files...")
    stats = analyze_qa_files(qa_dir)
    print_statistics(stats)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()

