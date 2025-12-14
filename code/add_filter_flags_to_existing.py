#!/usr/bin/env python3
"""
Add filter flags to existing filtered JSON files.
This retroactively adds flags to files that were filtered before the flag feature was added.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter patterns (same as filter_missing_answers_with_chunks.py)
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

def process_json_file(json_path: Path, patterns: List[re.Pattern]) -> bool:
    """Process a single JSON file to add filter flags. Returns True if modified."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip if already has filter flags
        if data.get('filtered_by_missing_info', False):
            logger.info(f"  {json_path.name} already has filter flags, skipping")
            return False
        
        answers = data.get('answers', [])
        filtered_answers_indices = []
        
        # Check each answer for missing info patterns
        for idx, answer in enumerate(answers):
            answer_text = answer.get('answer', '')
            if answer_contains_missing_info(answer_text, patterns):
                filtered_answers_indices.append(idx)
                answer['filtered_as_missing_info'] = True
                logger.info(f"  Marked Q{idx+1} as filtered (missing info)")
        
        if not filtered_answers_indices:
            logger.warning(f"  {json_path.name} has no matching answers, but is in filtered directory")
            # Still add file-level flag since it's in the filtered directory
            data['filtered_by_missing_info'] = True
            data['filtered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data['filtered_answers_count'] = 0
            data['filtered_answers_indices'] = []
        else:
            # Add file-level metadata
            data['filtered_by_missing_info'] = True
            data['filtered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data['filtered_answers_count'] = len(filtered_answers_indices)
            data['filtered_answers_indices'] = [idx + 1 for idx in filtered_answers_indices]  # 1-indexed
        
        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✓ Added flags to {json_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    input_dir = Path("/home/himanshu/dev/output/gemma3_rag_concise_missing_ans")
    
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)
    
    json_files = sorted(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        sys.exit(0)
    
    patterns = compile_patterns()
    
    print(f"Processing {len(json_files)} JSON files to add filter flags...")
    
    modified_count = 0
    for json_path in json_files:
        print(f"Processing: {json_path.name}")
        if process_json_file(json_path, patterns):
            modified_count += 1
    
    print(f"\nCompleted: {modified_count}/{len(json_files)} files modified")

if __name__ == "__main__":
    main()

