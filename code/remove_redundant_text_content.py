#!/usr/bin/env python3
"""
Remove redundant 'text_content' field from rag_chunks in filtered JSON files.
The 'text' field at the chunk level and 'text_content' in metadata are identical.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def remove_text_content_from_chunks(data: Dict[str, Any]) -> bool:
    """Remove 'text_content' from metadata in rag_chunks. Returns True if any changes made."""
    modified = False
    
    answers = data.get('answers', [])
    for answer in answers:
        chunks = answer.get('rag_chunks', [])
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if 'text_content' in metadata:
                # Verify they're the same (optional check)
                chunk_text = chunk.get('text', '')
                metadata_text_content = metadata.get('text_content', '')
                
                if chunk_text != metadata_text_content:
                    print(f"  WARNING: text and text_content differ in chunk {chunk.get('id', 'unknown')}")
                    print(f"    text length: {len(chunk_text)}")
                    print(f"    text_content length: {len(metadata_text_content)}")
                
                del metadata['text_content']
                modified = True
    
    return modified

def process_json_file(json_path: Path) -> bool:
    """Process a single JSON file. Returns True if modified."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = remove_text_content_from_chunks(data)
        
        if modified:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        import traceback
        traceback.print_exc()
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
    
    print(f"Processing {len(json_files)} JSON files...")
    
    modified_count = 0
    for json_path in json_files:
        print(f"Processing: {json_path.name}")
        if process_json_file(json_path):
            modified_count += 1
            print(f"  ✓ Removed text_content from {json_path.name}")
        else:
            print(f"  - No changes needed for {json_path.name}")
    
    print(f"\nCompleted: {modified_count}/{len(json_files)} files modified")

if __name__ == "__main__":
    main()

