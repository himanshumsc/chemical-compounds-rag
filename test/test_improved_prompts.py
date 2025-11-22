#!/usr/bin/env python3
"""
Test improved image prompts with one compound to check quality.
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

def test_improved_prompts():
    """Test improved image prompts with one compound."""
    # Paths
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    
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
    
    compound_path = compounds_dir / test_compound
    
    if not compound_path.exists():
        logger.warning(f"Compound file not found: {test_compound}")
        return
    
    logger.info(f"Testing improved prompts with: {test_compound}")
    
    try:
        # Load compound data
        with open(compound_path, 'r', encoding='utf-8') as f:
            compound_data = json.load(f)
        
        compound_name = compound_data.get('name', 'Unknown')
        main_entry_content = compound_data.get('main_entry_content', '')
        
        if not main_entry_content:
            logger.warning(f"No main_entry_content found for {compound_name}")
            return
        
        # Generate improved image prompt
        image_prompt = qa_generator.generate_image_prompt(compound_name, main_entry_content)
        logger.info(f"IMPROVED Image prompt for {compound_name}:")
        logger.info(f"  {image_prompt}")
        
        # Generate image
        image_url = qa_generator.generate_compound_image(compound_name, main_entry_content)
        
        if image_url:
            logger.info(f"✅ Successfully generated improved image for {compound_name}")
            logger.info(f"   Image URL: {image_url}")
            
            # Open image in browser for inspection
            try:
                webbrowser.open(image_url)
                logger.info(f"   Opened improved image in browser for inspection")
            except Exception as e:
                logger.warning(f"   Could not open browser: {e}")
            
        else:
            logger.error(f"❌ Failed to generate improved image for {compound_name}")
        
    except Exception as e:
        logger.error(f"Error processing {test_compound}: {e}")

if __name__ == "__main__":
    test_improved_prompts()
