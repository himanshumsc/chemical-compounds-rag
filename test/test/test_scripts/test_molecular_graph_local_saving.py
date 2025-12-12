#!/usr/bin/env python3
"""
Test molecular graph diagram with local image saving for Caffeine.
"""

import sys
from pathlib import Path
import json
import time

# Add the current directory to Python path to import our module
sys.path.append(str(Path(__file__).parent))

from generate_qa_pairs import QAGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_molecular_graph_with_local_saving():
    """Test molecular graph diagram with local image saving for Caffeine."""
    # Paths
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    output_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs_local_images")
    
    # Test with Caffeine (complex molecule)
    test_compound = "compound_026_Caffeine.json"
    
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
    
    logger.info(f"Testing molecular graph diagram with local saving for: {test_compound}")
    
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
        
        # Generate image and save locally
        local_image_path = qa_generator.generate_compound_image(compound_name, main_entry_content, output_dir)
        
        if not local_image_path:
            logger.warning(f"No image generated for {compound_name}")
            return
        
        # Create output data structure
        output_data = {
            "compound_id": compound_data.get('compound_id'),
            "compound_name": compound_name,
            "source_file": compound_path.name,
            "main_entry_length": compound_data.get('main_entry_length', 0),
            "qa_pairs_count": len(qa_pairs),
            "image_path": local_image_path,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qa_pairs": qa_pairs,
            "note": "First QA pair is for image identification, followed by 3 additional educational QA pairs. Image is saved locally."
        }
        
        # Save QA pairs to file
        output_filename = f"qa_local_{compound_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.json"
        output_file_path = output_dir / output_filename
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved QA pairs with local image for {compound_name} to {output_filename}")
        
        # Display the results
        print("\n" + "="*60)
        print("MOLECULAR GRAPH DIAGRAM WITH LOCAL SAVING")
        print("="*60)
        print(f"Compound: {compound_name}")
        print(f"Local Image Path: {local_image_path}")
        print(f"Full Image Path: {output_dir / local_image_path}")
        print(f"Total QA pairs: {len(qa_pairs)}")
        
        # Check if image file exists
        full_image_path = output_dir / local_image_path
        if full_image_path.exists():
            print(f"✅ Image file exists: {full_image_path}")
            print(f"   File size: {full_image_path.stat().st_size} bytes")
        else:
            print(f"❌ Image file not found: {full_image_path}")
        
        print("\nQA PAIRS:")
        
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\n{i}. {qa['question']}")
            print(f"   Answer: {qa['answer']}")
            print(f"   Difficulty: {qa['difficulty_level']} | Category: {qa['topic_category']}")
        
        print("\n" + "="*60)
        print("✅ SUCCESS: Molecular graph diagram generated and saved locally!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🖼️  Images directory: {output_dir / 'images'}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error processing {test_compound}: {e}")

if __name__ == "__main__":
    test_molecular_graph_with_local_saving()
