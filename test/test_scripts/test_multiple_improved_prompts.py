#!/usr/bin/env python3
"""
Test improved image prompts with multiple compounds to check quality.
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

def test_multiple_improved_prompts():
    """Test improved image prompts with multiple compounds."""
    # Paths
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    
    # Test with different types of compounds
    test_compounds = [
        "compound_001_Acetic acid.json",      # CH3COOH
        "compound_018_Benzene.json",          # C6H6
        "compound_034_Carbon Dioxide.json",   # CO2
        "compound_146_Sodium Chloride.json",  # NaCl
        "compound_173_Water.json"             # H2O
    ]
    
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
    
    logger.info(f"Testing improved prompts with {len(test_compounds)} compounds...")
    
    for i, test_compound in enumerate(test_compounds, 1):
        compound_path = compounds_dir / test_compound
        
        if not compound_path.exists():
            logger.warning(f"Compound file not found: {test_compound}")
            continue
        
        logger.info(f"Testing {i}/{len(test_compounds)}: {test_compound}")
        
        try:
            # Load compound data
            with open(compound_path, 'r', encoding='utf-8') as f:
                compound_data = json.load(f)
            
            compound_name = compound_data.get('name', 'Unknown')
            main_entry_content = compound_data.get('main_entry_content', '')
            
            if not main_entry_content:
                logger.warning(f"No main_entry_content found for {compound_name}")
                continue
            
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
            
            # Add delay between requests
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Error processing {test_compound}: {e}")
    
    print("\n" + "="*60)
    print("IMPROVED PROMPT TEST COMPLETED")
    print("="*60)
    print("Please review the generated images in your browser.")
    print("The new prompts should show:")
    print("1. Simple 2D ball-and-stick molecular models")
    print("2. Chemical formulas included in the prompt")
    print("3. Minimalist scientific diagrams")
    print("4. Clean white backgrounds")
    print("5. 256x256 pixel size")
    print("\nIf these look good, we can proceed with all 178 compounds!")

if __name__ == "__main__":
    test_multiple_improved_prompts()
