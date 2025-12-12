#!/usr/bin/env python3
"""
Test script to check image generation quality before processing all compounds.
This will test DALL-E 2 image generation with a few sample compounds.
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

def test_image_generation():
    """Test image generation with sample compounds."""
    # Paths
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    output_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs_test")
    
    # Test compounds (different types)
    test_compounds = [
        "compound_001_Acetic acid.json",      # Liquid organic acid
        "compound_018_Benzene.json",          # Aromatic hydrocarbon
        "compound_034_Carbon Dioxide.json",   # Gas
        "compound_146_Sodium Chloride.json",  # Ionic solid
        "compound_173_Water.json"             # Simple molecule
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
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Testing image generation with {len(test_compounds)} compounds...")
    
    results = []
    
    for i, compound_file in enumerate(test_compounds, 1):
        compound_path = compounds_dir / compound_file
        
        if not compound_path.exists():
            logger.warning(f"Compound file not found: {compound_file}")
            continue
        
        logger.info(f"Testing {i}/{len(test_compounds)}: {compound_file}")
        
        try:
            # Load compound data
            with open(compound_path, 'r', encoding='utf-8') as f:
                compound_data = json.load(f)
            
            compound_name = compound_data.get('name', 'Unknown')
            main_entry_content = compound_data.get('main_entry_content', '')
            
            if not main_entry_content:
                logger.warning(f"No main_entry_content found for {compound_name}")
                continue
            
            # Generate image prompt (just to see what it looks like)
            image_prompt = qa_generator.generate_image_prompt(compound_name, main_entry_content)
            logger.info(f"Image prompt for {compound_name}:")
            logger.info(f"  {image_prompt}")
            
            # Generate image
            image_url = qa_generator.generate_compound_image(compound_name, main_entry_content)
            
            if image_url:
                logger.info(f"✅ Successfully generated image for {compound_name}")
                logger.info(f"   Image URL: {image_url}")
                
                # Store result
                results.append({
                    "compound_name": compound_name,
                    "compound_file": compound_file,
                    "image_url": image_url,
                    "image_prompt": image_prompt,
                    "success": True
                })
                
                # Open image in browser for manual inspection
                try:
                    webbrowser.open(image_url)
                    logger.info(f"   Opened image in browser for inspection")
                except Exception as e:
                    logger.warning(f"   Could not open browser: {e}")
                
            else:
                logger.error(f"❌ Failed to generate image for {compound_name}")
                results.append({
                    "compound_name": compound_name,
                    "compound_file": compound_file,
                    "image_url": "",
                    "image_prompt": image_prompt,
                    "success": False
                })
            
            # Add delay between requests
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error processing {compound_file}: {e}")
            results.append({
                "compound_name": compound_name if 'compound_name' in locals() else 'Unknown',
                "compound_file": compound_file,
                "image_url": "",
                "image_prompt": "",
                "success": False,
                "error": str(e)
            })
    
    # Save test results
    test_results_file = output_dir / "image_generation_test_results.json"
    with open(test_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("IMAGE GENERATION TEST RESULTS")
    print("="*60)
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f"Total compounds tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\n✅ SUCCESSFUL IMAGE GENERATIONS:")
        for result in results:
            if result.get('success', False):
                print(f"  - {result['compound_name']}")
                print(f"    URL: {result['image_url']}")
    
    if failed > 0:
        print(f"\n❌ FAILED IMAGE GENERATIONS:")
        for result in results:
            if not result.get('success', False):
                print(f"  - {result['compound_name']}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
    
    print(f"\nTest results saved to: {test_results_file}")
    print("="*60)
    
    # Quality assessment prompt
    print(f"\n🔍 QUALITY ASSESSMENT:")
    print("Please review the generated images in your browser and assess:")
    print("1. Are the images scientifically accurate?")
    print("2. Do they show appropriate molecular structures?")
    print("3. Are they educational and clear?")
    print("4. Do they match the compound properties (liquid/solid/gas)?")
    print("5. Overall quality rating (1-10)?")
    print("\nBased on your assessment, we can proceed with all 178 compounds or adjust the prompts.")

if __name__ == "__main__":
    test_image_generation()
