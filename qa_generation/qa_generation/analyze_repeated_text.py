#!/usr/bin/env python3
"""
Analyze repeated text patterns in comprehensive_text of failed files.
Check for:
- Duplicate sentences/paragraphs
- Repeated phrases
- High repetition ratios
"""

import json
import re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

def find_repeated_sentences(text):
    """Find repeated sentences in text."""
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short fragments
    
    sentence_counts = Counter(sentences)
    repeated = {sent: count for sent, count in sentence_counts.items() if count > 1}
    return repeated

def find_repeated_phrases(text, min_length=50):
    """Find repeated phrases of minimum length."""
    words = text.split()
    phrases = {}
    
    # Check for repeated sequences of words
    for i in range(len(words) - min_length // 5):  # Approximate word count
        for length in range(10, min(len(words) - i, 50)):  # Phrase length 10-50 words
            phrase = ' '.join(words[i:i+length])
            if len(phrase) >= min_length:
                if phrase not in phrases:
                    phrases[phrase] = 0
                phrases[phrase] += 1
    
    # Return phrases that appear more than once
    repeated = {phrase: count for phrase, count in phrases.items() if count > 1}
    return repeated

def calculate_repetition_ratio(text):
    """Calculate ratio of unique to total content."""
    # Split into chunks (sentences or paragraphs)
    chunks = re.split(r'[.!?]\n+|[.!?]\s+', text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
    
    if not chunks:
        return 0, 0, 0
    
    unique_chunks = set(chunks)
    total_chunks = len(chunks)
    unique_count = len(unique_chunks)
    
    repetition_ratio = 1 - (unique_count / total_chunks) if total_chunks > 0 else 0
    
    return repetition_ratio, unique_count, total_chunks

def find_duplicate_paragraphs(text):
    """Find duplicate paragraphs."""
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    para_counts = Counter(paragraphs)
    duplicates = {para: count for para, count in para_counts.items() if count > 1}
    return duplicates

def analyze_file(compound_file):
    """Analyze a single compound file for repeated text."""
    try:
        with open(compound_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        comprehensive_text = data.get('comprehensive_text', '')
        if not comprehensive_text:
            return None
        
        compound_name = data.get('name', 'Unknown')
        text_length = len(comprehensive_text)
        
        # Analyze repetition
        repeated_sentences = find_repeated_sentences(comprehensive_text)
        repeated_phrases = find_repeated_phrases(comprehensive_text, min_length=30)
        duplicate_paragraphs = find_duplicate_paragraphs(comprehensive_text)
        repetition_ratio, unique_chunks, total_chunks = calculate_repetition_ratio(comprehensive_text)
        
        # Count occurrences of common repeated patterns
        # Check for timeline references (common pattern)
        timeline_pattern = r'Page \d+:|^\d{4} •'
        timeline_matches = len(re.findall(timeline_pattern, comprehensive_text, re.MULTILINE))
        
        # Check for repeated compound names
        clean_name = compound_name.replace('.', '').strip()
        name_occurrences = len(re.findall(re.escape(clean_name), comprehensive_text, re.IGNORECASE))
        
        return {
            'compound_name': compound_name,
            'text_length': text_length,
            'repeated_sentences_count': len(repeated_sentences),
            'repeated_sentences_total': sum(repeated_sentences.values()),
            'repeated_phrases_count': len(repeated_phrases),
            'duplicate_paragraphs_count': len(duplicate_paragraphs),
            'repetition_ratio': repetition_ratio,
            'unique_chunks': unique_chunks,
            'total_chunks': total_chunks,
            'timeline_references': timeline_matches,
            'name_occurrences': name_occurrences,
            'repeated_sentences': dict(list(repeated_sentences.items())[:5]),  # Top 5
            'duplicate_paragraphs': dict(list(duplicate_paragraphs.items())[:3]),  # Top 3
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    failed_files = [
        '113_Petroleum.json', '140_Riboflavin.json', '151_Sodium_Cyclamate.json',
        '154_Sodium_Hypochlorite.json', '156_Sodium_Phosphate.json', '172_Toluene.json',
        '22_Benzene.json', '32_Calcium_Oxide.json', '40_Cellulose.json', '44_Chlorophyll.json'
    ]
    
    # Get all 17 failed files from log
    qa_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive")
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    
    print("="*70)
    print("ANALYZING REPEATED TEXT IN FAILED FILES")
    print("="*70)
    
    results = []
    
    # Process all files in QA directory to find failed ones
    qa_files = list(qa_dir.glob('*.json'))
    for qa_file in qa_files:
        try:
            with open(qa_file, 'r') as f:
                qa_data = json.load(f)
            
            # Check if this file was NOT updated (failed)
            if qa_data.get('updated_by') != 'comprehensive_text_generator':
                compound_id = qa_data.get('compound_id')
                source_file = qa_data.get('source_file', '')
                
                # Find compound file
                compound_file = None
                if compound_id:
                    pattern = f'compound_{compound_id:03d}_*.json'
                    matches = list(compounds_dir.glob(pattern))
                    if matches:
                        compound_file = matches[0]
                
                if not compound_file and source_file:
                    compound_file = compounds_dir / source_file
                
                if compound_file and compound_file.exists():
                    analysis = analyze_file(compound_file)
                    if analysis and 'error' not in analysis:
                        analysis['qa_file'] = qa_file.name
                        results.append(analysis)
        except:
            continue
    
    # Print results
    print(f"\nAnalyzed {len(results)} failed files\n")
    
    # Summary statistics
    if results:
        avg_repetition = sum(r['repetition_ratio'] for r in results) / len(results)
        avg_text_length = sum(r['text_length'] for r in results) / len(results)
        avg_timeline_refs = sum(r['timeline_references'] for r in results) / len(results)
        
        print("Summary Statistics:")
        print(f"  Average text length: {avg_text_length:.0f} chars")
        print(f"  Average repetition ratio: {avg_repetition:.2%}")
        print(f"  Average timeline references: {avg_timeline_refs:.0f}")
        print()
    
    # Detailed analysis per file
    print("Detailed Analysis per File:")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: x['repetition_ratio'], reverse=True):
        print(f"\n{r['compound_name']} ({r['qa_file']}):")
        print(f"  Text length: {r['text_length']:,} chars")
        print(f"  Repetition ratio: {r['repetition_ratio']:.2%}")
        print(f"  Unique chunks: {r['unique_chunks']}/{r['total_chunks']}")
        print(f"  Repeated sentences: {r['repeated_sentences_count']} ({r['repeated_sentences_total']} total occurrences)")
        print(f"  Repeated phrases: {r['repeated_phrases_count']}")
        print(f"  Duplicate paragraphs: {r['duplicate_paragraphs_count']}")
        print(f"  Timeline references: {r['timeline_references']}")
        print(f"  Compound name occurrences: {r['name_occurrences']}")
        
        if r['repeated_sentences']:
            print(f"  Top repeated sentences:")
            for sent, count in list(r['repeated_sentences'].items())[:3]:
                print(f"    [{count}x] {sent[:80]}...")
        
        if r['duplicate_paragraphs']:
            print(f"  Duplicate paragraphs:")
            for para, count in list(r['duplicate_paragraphs'].items())[:2]:
                print(f"    [{count}x] {para[:100]}...")
    
    # Compare with successful files (sample)
    print("\n" + "="*70)
    print("COMPARISON: Sample of Successful Files")
    print("="*70)
    
    successful_sample = []
    for qa_file in qa_files[:20]:  # Sample 20 files
        try:
            with open(qa_file, 'r') as f:
                qa_data = json.load(f)
            
            if qa_data.get('updated_by') == 'comprehensive_text_generator':
                compound_id = qa_data.get('compound_id')
                if compound_id:
                    pattern = f'compound_{compound_id:03d}_*.json'
                    matches = list(compounds_dir.glob(pattern))
                    if matches:
                        analysis = analyze_file(matches[0])
                        if analysis and 'error' not in analysis:
                            successful_sample.append(analysis)
        except:
            continue
    
    if successful_sample:
        avg_success_rep = sum(r['repetition_ratio'] for r in successful_sample) / len(successful_sample)
        avg_success_length = sum(r['text_length'] for r in successful_sample) / len(successful_sample)
        
        print(f"\nSuccessful files (sample of {len(successful_sample)}):")
        print(f"  Average text length: {avg_success_length:.0f} chars")
        print(f"  Average repetition ratio: {avg_success_rep:.2%}")
        print()
        print(f"Failed files ({len(results)}):")
        print(f"  Average text length: {avg_text_length:.0f} chars")
        print(f"  Average repetition ratio: {avg_repetition:.2%}")
        print()
        print(f"Difference:")
        print(f"  Length: {avg_text_length - avg_success_length:+.0f} chars")
        print(f"  Repetition: {avg_repetition - avg_success_rep:+.2%}")

if __name__ == "__main__":
    main()

