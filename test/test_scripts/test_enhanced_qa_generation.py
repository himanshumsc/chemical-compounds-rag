#!/usr/bin/env python3
"""
Test enhanced QA generation with image identification question.
"""

import sys
from pathlib import Path
import json
import webbrowser
import time

# Add the current directory to Python path to import our module
sys.path.append(str(Path(__file__).parent))

from generate_qa_pairs import QAGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_qa_generation():
    """Test enhanced QA generation with image identification."""
    # Paths
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    output_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs_enhanced")
    
    # Test with Benzene (has clear molecular structure)
    test_compound = "compound_018_Benzene.json"
    
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
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    compound_path = compounds_dir / test_compound
    
    if not compound_path.exists():
        logger.warning(f"Compound file not found: {test_compound}")
        return
    
    logger.info(f"Testing enhanced QA generation with: {test_compound}")
    
    try:
        # Load compound data
        with open(compound_path, 'r', encoding='utf-8') as f:
            compound_data = json.load(f)
        
        compound_name = compound_data.get('name', 'Unknown')
        main_entry_content = compound_data.get('main_entry_content', '')
        
        if not main_entry_content:
            logger.warning(f"No main_entry_content found for {compound_name}")
            return
        
        # Generate QA pairs (now includes image identification)
        qa_pairs = qa_generator.generate_qa_pairs(main_entry_content, compound_name)
        
        if not qa_pairs:
            logger.warning(f"No QA pairs generated for {compound_name}")
            return
        
        # Generate image
        image_url = qa_generator.generate_compound_image(compound_name, main_entry_content)
        
        if not image_url:
            logger.warning(f"No image generated for {compound_name}")
            return
        
        # Create output data structure
        output_data = {
            "compound_id": compound_data.get('compound_id'),
            "compound_name": compound_name,
            "source_file": compound_path.name,
            "main_entry_length": compound_data.get('main_entry_length', 0),
            "qa_pairs_count": len(qa_pairs),
            "image_url": image_url,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qa_pairs": qa_pairs,
            "note": "First QA pair is for image identification, followed by 3 additional educational QA pairs"
        }
        
        # Save QA pairs to file
        output_filename = f"qa_enhanced_{compound_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.json"
        output_file_path = output_dir / output_filename
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved enhanced QA pairs for {compound_name} to {output_filename}")
        
        # Display the QA pairs
        print("\n" + "="*60)
        print("ENHANCED QA PAIRS GENERATED")
        print("="*60)
        print(f"Compound: {compound_name}")
        print(f"Image URL: {image_url}")
        print(f"Total QA pairs: {len(qa_pairs)}")
        print("\nQA PAIRS:")
        
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\n{i}. {qa['question']}")
            print(f"   Answer: {qa['answer']}")
            print(f"   Difficulty: {qa['difficulty_level']} | Category: {qa['topic_category']}")
        
        print("\n" + "="*60)
        
        # Open image in browser for inspection
        try:
            webbrowser.open(image_url)
            logger.info(f"Opened image in browser for inspection")
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")
        
    except Exception as e:
        logger.error(f"Error processing {test_compound}: {e}")

if __name__ == "__main__":
    test_enhanced_qa_generation()
