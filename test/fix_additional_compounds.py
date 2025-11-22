#!/usr/bin/env python3
"""
Fix Additional Compounds - Correct IDs and Names
Fix the compound IDs and names for the 3 additional compounds found
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_additional_compounds():
    """Fix the compound IDs and names for additional compounds"""
    
    output_dir = Path("/home/himanshu/dev/test/data/processed")
    
    # Define the correct compound information
    compounds_to_fix = [
        {
            "old_file": "compound_175_Liquid.json",
            "new_id": 175,
            "correct_name": "2,2'-Dichlorodiethyl Sulfide (Mustard Gas)"
        },
        {
            "old_file": "compound_175_Solid.json", 
            "new_id": 176,
            "correct_name": "2-(4-Isobutylphenyl)propionic Acid (Ibuprofen)"
        },
        {
            "old_file": "compound_175_Methyltrinitroben.json",
            "new_id": 177,
            "correct_name": "2,4,6-Trinitrotoluene (TNT)"
        }
    ]
    
    logger.info("Fixing additional compound IDs and names...")
    
    for compound_info in compounds_to_fix:
        old_file_path = output_dir / "individual_compounds" / compound_info["old_file"]
        
        if old_file_path.exists():
            # Load the existing file
            with open(old_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update the data
            data["compound_id"] = compound_info["new_id"]
            data["name"] = compound_info["correct_name"]
            data["metadata"]["compound_index"] = compound_info["new_id"]
            data["metadata"]["total_compounds"] = 177
            
            # Create new filename
            clean_name = compound_info["correct_name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            new_filename = f"compound_{compound_info['new_id']:03d}_{clean_name}.json"
            new_file_path = output_dir / "individual_compounds" / new_filename
            
            # Save with new filename
            with open(new_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Remove old file
            old_file_path.unlink()
            
            logger.info(f"Fixed: {compound_info['old_file']} -> {new_filename}")
            logger.info(f"  ID: {compound_info['new_id']}")
            logger.info(f"  Name: {compound_info['correct_name']}")
        else:
            logger.warning(f"File not found: {compound_info['old_file']}")
    
    # Update the compound names list
    update_compound_list(output_dir)
    
    logger.info("✅ Additional compounds fixed successfully!")

def update_compound_list(output_dir: Path):
    """Update the compound names list with corrected information"""
    try:
        compound_list_path = output_dir / "compound_names_list.json"
        
        # Load existing list
        with open(compound_list_path, 'r', encoding='utf-8') as f:
            compounds = json.load(f)
        
        # Remove the last 3 entries (they have wrong IDs)
        compounds = compounds[:-3]
        
        # Add the corrected entries
        corrected_compounds = [
            {
                "compound_id": 175,
                "name": "2,2'-Dichlorodiethyl Sulfide (Mustard Gas)",
                "start_page": 5,
                "end_page": 5,
                "total_pages": 1
            },
            {
                "compound_id": 176,
                "name": "2-(4-Isobutylphenyl)propionic Acid (Ibuprofen)",
                "start_page": 9,
                "end_page": 9,
                "total_pages": 1
            },
            {
                "compound_id": 177,
                "name": "2,4,6-Trinitrotoluene (TNT)",
                "start_page": 15,
                "end_page": 15,
                "total_pages": 1
            }
        ]
        
        compounds.extend(corrected_compounds)
        
        # Save updated list
        with open(compound_list_path, 'w', encoding='utf-8') as f:
            json.dump(compounds, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated compound list: {len(compounds)} total compounds")
        
    except Exception as e:
        logger.error(f"Failed to update compound list: {e}")

if __name__ == "__main__":
    fix_additional_compounds()
