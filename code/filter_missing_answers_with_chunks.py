#!/usr/bin/env python3
"""
Filter QA sets with missing information answers and extract ChromaDB chunks.

This script:
1. Filters JSON files from gemma3_rag_concise that contain answers indicating missing info
2. Copies matching files to gemma3_rag_concise_missing_ans
3. Extracts the ChromaDB chunks that were used for each answer
4. Adds chunks to the JSON structure
"""

import sys
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Patch SQLite for ChromaDB compatibility
sys.modules['sqlite3'] = __import__('pysqlite3')

from chromadb_search import ChromaDBSearchEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter patterns (case-insensitive, partial match)
MISSING_INFO_PATTERNS = [
    r"text does not contain information",
    r"information is not present in the provided text",
    r"provided documents do not contain",
    r"not mentioned in the provided context",
    r"not found in provided",
    r"not available in the provided",
    r"not present in the provided",
    r"not mentioned in provided",
    r"not found in provided sources",
    r"not available in provided context",
    r"cannot find.*in.*provided",
    r"no information.*provided",
    r"information.*not.*available.*provided",
]

def compile_patterns() -> List[re.Pattern]:
    """Compile regex patterns for case-insensitive matching."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in MISSING_INFO_PATTERNS]

def answer_contains_missing_info(answer_text: str, patterns: List[re.Pattern]) -> bool:
    """Check if answer text matches any missing info pattern."""
    if not answer_text:
        return False
    for pattern in patterns:
        if pattern.search(answer_text):
            return True
    return False

def build_context_from_chunks(chunks: List[Dict]) -> str:
    """Build context string from retrieved chunks (same as multimodal_qa_runner_vllm.py)."""
    if not chunks:
        return ""
    context_parts = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get('text', '')
        chunk_score = chunk.get('score', 0.0)
        context_parts.append(
            f"[Source {i+1} (Relevance: {chunk_score:.3f})]\n{chunk_text}"
        )
    return "\n\n".join(context_parts)

def load_qa_file(qa_path: Path) -> Optional[Dict]:
    """Load original QA file JSON."""
    try:
        if not qa_path.exists():
            logger.warning(f"QA file not found: {qa_path}")
            return None
        with open(qa_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load QA file {qa_path}: {e}")
        return None

def extract_chunks_for_answer(
    answer_data: Dict,
    search_engine: ChromaDBSearchEngine,
    image_path: Optional[str] = None,
    n_results: int = 5
) -> tuple[List[Dict], str]:
    """
    Extract ChromaDB chunks for an answer.
    
    Returns:
        (chunks_list, formatted_context_string)
    """
    chunks = []
    formatted_context = ""
    
    search_type = answer_data.get('search_type', 'text')
    question = answer_data.get('question', '')
    
    try:
        if search_type == 'image' and image_path:
            # Image similarity search for Q1
            if Path(image_path).exists():
                chunks = search_engine.image_similarity_search(image_path, n_results=n_results)
                logger.info(f"  Q1: Found {len(chunks)} chunks via image search")
            else:
                logger.warning(f"  Q1: Image path not found: {image_path}")
        else:
            # Text search for Q2-Q4
            if question:
                chunks = search_engine.text_search(question, n_results=n_results)
                logger.info(f"  Q{answer_data.get('question_idx', '?')}: Found {len(chunks)} chunks via text search")
            else:
                logger.warning(f"  No question text available")
        
        # Format context
        formatted_context = build_context_from_chunks(chunks)
        
    except Exception as e:
        logger.error(f"  Failed to extract chunks: {e}")
    
    return chunks, formatted_context

def process_json_file(
    json_path: Path,
    output_dir: Path,
    search_engine: ChromaDBSearchEngine,
    patterns: List[re.Pattern],
    qa_base_dir: Optional[Path] = None
) -> bool:
    """
    Process a single JSON file: check for missing info, copy if matches, extract chunks.
    
    Returns:
        True if file was copied (matched filter), False otherwise
    """
    try:
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check all answers for missing info patterns (including Q1)
        answers = data.get('answers', [])
        has_missing_info = False
        filtered_answers_indices = []  # Track which answers were filtered
        
        for idx, answer in enumerate(answers):
            answer_text = answer.get('answer', '')
            if answer_contains_missing_info(answer_text, patterns):
                has_missing_info = True
                filtered_answers_indices.append(idx)
                logger.info(f"  Found missing info in Q{idx+1}: {answer.get('question', '')[:50]}...")
        
        if not has_missing_info:
            return False
        
        # File matches filter - copy to output directory
        output_path = output_dir / json_path.name
        logger.info(f"Copying {json_path.name} to {output_dir}")
        shutil.copy2(json_path, output_path)
        
        # Load original QA file to get image_path
        image_path = None
        source_file = data.get('source_file', '')
        source_input_dir = data.get('source_input_dir', '')
        
        if source_file and source_input_dir:
            qa_file_path = Path(source_input_dir) / source_file
            qa_data = load_qa_file(qa_file_path)
            if qa_data:
                image_path = qa_data.get('image_path', '')
                logger.info(f"  Loaded image_path: {image_path}")
        elif qa_base_dir and source_file:
            # Fallback: try qa_base_dir
            qa_file_path = qa_base_dir / source_file
            qa_data = load_qa_file(qa_file_path)
            if qa_data:
                image_path = qa_data.get('image_path', '')
                logger.info(f"  Loaded image_path from fallback: {image_path}")
        
        # Mark filtered answers with flag
        for idx in filtered_answers_indices:
            answers[idx]['filtered_as_missing_info'] = True
            logger.info(f"  Marked Q{idx+1} as filtered (missing info)")
        
        # Extract chunks for each answer (skip Q1 - image-based questions)
        logger.info(f"  Extracting chunks for Q2-Q4 (skipping Q1)...")
        for idx, answer in enumerate(answers):
            question_idx = idx + 1
            
            # Skip Q1 (index 0) - don't extract chunks for image-based questions
            if idx == 0:
                answer['rag_chunks'] = []
                answer['rag_context_formatted'] = ""
                answer['chunks_skipped'] = "Q1 (image-based question)"
                continue
            
            answer['question_idx'] = question_idx  # Add for logging
            
            chunks, formatted_context = extract_chunks_for_answer(
                answer,
                search_engine,
                image_path=None,  # Q2-Q4 are text-based, no image needed
                n_results=5
            )
            
            # Add chunks to answer
            answer['rag_chunks'] = chunks
            answer['rag_context_formatted'] = formatted_context
        
        # Add file-level and answer-level metadata
        data['filtered_by_missing_info'] = True
        data['filtered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data['filtered_answers_count'] = len(filtered_answers_indices)
        data['filtered_answers_indices'] = [idx + 1 for idx in filtered_answers_indices]  # 1-indexed for readability
        data['rag_chunks_extracted'] = True
        data['chunks_extracted_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save updated JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✓ Saved with chunks: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter missing answer JSONs and extract ChromaDB chunks"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/home/himanshu/dev/output/gemma3_rag_concise',
        help='Input directory with JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/himanshu/dev/output/gemma3_rag_concise_missing_ans',
        help='Output directory for filtered JSONs with chunks'
    )
    parser.add_argument(
        '--chromadb-path',
        type=str,
        default='/home/himanshu/dev/data/chromadb',
        help='Path to ChromaDB database'
    )
    parser.add_argument(
        '--qa-base-dir',
        type=str,
        default='/home/himanshu/dev/test/data/processed/qa_pairs_individual_components',
        help='Base directory for original QA files (fallback if source_input_dir not found)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for CLIP model (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    chromadb_path = Path(args.chromadb_path)
    qa_base_dir = Path(args.qa_base_dir) if args.qa_base_dir else None
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize ChromaDB search engine
    logger.info(f"Initializing ChromaDB search engine from {chromadb_path}...")
    try:
        search_engine = ChromaDBSearchEngine(str(chromadb_path), device=args.device)
        logger.info("ChromaDB search engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return 1
    
    # Compile patterns
    patterns = compile_patterns()
    logger.info(f"Compiled {len(patterns)} filter patterns")
    
    # Find all JSON files
    json_files = list(input_dir.glob('*.json'))
    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Process files
    matched_count = 0
    error_count = 0
    
    for json_file in json_files:
        logger.info(f"Processing: {json_file.name}")
        try:
            if process_json_file(json_file, output_dir, search_engine, patterns, qa_base_dir):
                matched_count += 1
        except Exception as e:
            logger.error(f"Failed to process {json_file.name}: {e}")
            error_count += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Summary:")
    logger.info(f"  Total files scanned: {len(json_files)}")
    logger.info(f"  Files matched filter: {matched_count}")
    logger.info(f"  Files with errors: {error_count}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

