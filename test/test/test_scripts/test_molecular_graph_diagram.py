#!/usr/bin/env python3
"""
Test molecular graph diagram prompt with Caffeine.
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

def test_molecular_graph_diagram():
    """Test molecular graph diagram prompt with Caffeine."""
    # Paths
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    
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
    
    compound_path = compounds_dir / test_compound
    
    if not compound_path.exists():
        logger.warning(f"Compound file not found: {test_compound}")
        return
    
    logger.info(f"Testing molecular graph diagram prompt with: {test_compound}")
    
    try:
        # Load compound data
        with open(compound_path, 'r', encoding='utf-8') as f:
            compound_data = json.load(f)
        
        compound_name = compound_data.get('name', 'Unknown')
        main_entry_content = compound_data.get('main_entry_content', '')
        
        if not main_entry_content:
            logger.warning(f"No main_entry_content found for {compound_name}")
            return
        
        # Generate molecular graph diagram prompt
        image_prompt = qa_generator.generate_image_prompt(compound_name, main_entry_content)
        logger.info(f"MOLECULAR GRAPH DIAGRAM prompt for {compound_name}:")
        logger.info(f"  {image_prompt}")
        
        # Generate image
        image_url = qa_generator.generate_compound_image(compound_name, main_entry_content)
        
        if image_url:
            logger.info(f"✅ Successfully generated molecular graph diagram for {compound_name}")
            logger.info(f"   Image URL: {image_url}")
            
            # Open image in browser for inspection
            try:
                webbrowser.open(image_url)
                logger.info(f"   Opened molecular graph diagram in browser for inspection")
            except Exception as e:
                logger.warning(f"   Could not open browser: {e}")
            
        else:
            logger.error(f"❌ Failed to generate molecular graph diagram for {compound_name}")
        
        # Also test with a simpler molecule for comparison
        logger.info(f"\nTesting with Water for comparison...")
        water_compound = "compound_173_Water.json"
        water_path = compounds_dir / water_compound
        
        if water_path.exists():
            with open(water_path, 'r', encoding='utf-8') as f:
                water_data = json.load(f)
            
            water_name = water_data.get('name', 'Unknown')
            water_content = water_data.get('main_entry_content', '')
            
            water_prompt = qa_generator.generate_image_prompt(water_name, water_content)
            logger.info(f"MOLECULAR GRAPH DIAGRAM prompt for {water_name}:")
            logger.info(f"  {water_prompt}")
            
            water_image_url = qa_generator.generate_compound_image(water_name, water_content)
            
            if water_image_url:
                logger.info(f"✅ Successfully generated molecular graph diagram for {water_name}")
                logger.info(f"   Image URL: {water_image_url}")
                
                try:
                    webbrowser.open(water_image_url)
                    logger.info(f"   Opened Water molecular graph diagram in browser for comparison")
                except Exception as e:
                    logger.warning(f"   Could not open browser: {e}")
        
    except Exception as e:
        logger.error(f"Error processing {test_compound}: {e}")
    
    print("\n" + "="*60)
    print("MOLECULAR GRAPH DIAGRAM TEST COMPLETED")
    print("="*60)
    print("Please review the generated images in your browser.")
    print("The molecular graph diagrams should show:")
    print("1. Graph-based molecular structure representation")
    print("2. Nodes (atoms) and edges (bonds) clearly visible")
    print("3. Complete molecular structures fitting within 512x512")
    print("4. Clean, minimalist scientific diagrams")
    print("5. Better representation for complex molecules like Caffeine")
    print("\nCompare with previous ball-and-stick models to see the difference!")

if __name__ == "__main__":
    test_molecular_graph_diagram()
