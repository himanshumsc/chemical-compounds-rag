#!/usr/bin/env python3
"""
Process all 178 compounds to generate QA pairs.
This script will run the QA generation for all compounds in the dataset.
"""

import sys
from pathlib import Path

# Add the current directory to Python path to import our module
sys.path.append(str(Path(__file__).parent))

from generate_qa_pairs import QAGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Process all 178 compounds to generate QA pairs."""
    # Paths
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    output_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs")
    
    # Check if compounds directory exists
    if not compounds_dir.exists():
        logger.error(f"Compounds directory not found: {compounds_dir}")
        return
    
    # Initialize QA Generator
    try:
        qa_generator = QAGenerator()
        logger.info("QA Generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize QA Generator: {e}")
        return
    
    # Process ALL compounds (remove max_files limit)
    logger.info("Starting QA generation for ALL compounds...")
    results = qa_generator.process_all_compounds(compounds_dir, output_dir, max_files=None)
    
    # Print results
    print("\n" + "="*60)
    print("QA GENERATION RESULTS - ALL COMPOUNDS")
    print("="*60)
    print(f"Total files processed: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Average time per compound: {results['duration']/results['total_files']:.2f} seconds")
    
    if results['failed_files']:
        print(f"\nFailed files:")
        for file in results['failed_files']:
            print(f"  - {file}")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*60)
    
    # Calculate estimated cost (rough estimate)
    # GPT-4o pricing: ~$0.005 per 1K input tokens, ~$0.015 per 1K output tokens
    # Average input: ~2.7K chars ≈ 2.7K tokens, Average output: ~1.2K tokens (4 QA pairs)
    estimated_input_tokens = results['successful'] * 2.7  # 2.7K tokens per compound
    estimated_output_tokens = results['successful'] * 1.2  # ~1.2K tokens per compound (4 QA pairs)
    
    # DALL-E 2 pricing: $0.02 per 512x512 image
    image_cost = results['successful'] * 0.02  # $0.02 per 512x512 image
    
    input_cost = (estimated_input_tokens / 1000) * 0.005
    output_cost = (estimated_output_tokens / 1000) * 0.015
    total_cost = input_cost + output_cost + image_cost
    
    print(f"\n💰 ESTIMATED COST:")
    print(f"QA Input tokens: ~{estimated_input_tokens:.0f}K (${input_cost:.3f})")
    print(f"QA Output tokens: ~{estimated_output_tokens:.0f}K (${output_cost:.3f})")
    print(f"Images (512x512): {results['successful']} × $0.02 = ${image_cost:.3f}")
    print(f"Total estimated cost: ${total_cost:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
