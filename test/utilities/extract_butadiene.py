#!/usr/bin/env python3
"""
Extract 1,3-Butadiene from Arabic pages 1-3
"""

import fitz  # PyMuPDF
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_butadiene():
    """Extract 1,3-Butadiene from Arabic pages 1-3"""
    
    pdf_path = "/home/himanshu/dev/test/data/raw/chemical-compounds.pdf"
    output_dir = Path("/home/himanshu/dev/test/data/processed")
    
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"PDF opened with {len(doc)} pages")
        
        # Build page mapping
        page_mapping = {}
        for pdf_page_num in range(len(doc)):
            page = doc[pdf_page_num]
            text = page.get_text()
            
            # Extract Arabic page number from footer
            lines = text.split('\n')
            for line in reversed(lines[-5:]):  # Check last 5 lines
                line = line.strip()
                if line.isdigit() and 1 <= int(line) <= 1000:
                    arabic_page_num = int(line)
                    page_mapping[arabic_page_num] = pdf_page_num + 1
                    break
        
        # Extract content from Arabic pages 1-3
        all_content = []
        pdf_pages = []
        
        for arabic_page in [1, 2, 3]:
            pdf_page = page_mapping.get(arabic_page)
            if pdf_page:
                logger.info(f"Extracting Arabic page {arabic_page} (PDF page {pdf_page})")
                page = doc[pdf_page - 1]  # Convert to 0-based
                text = page.get_text()
                all_content.append(text)
                pdf_pages.append(pdf_page)
        
        # Combine all content
        combined_content = '\n'.join(all_content)
        
        # Create compound data
        compound_data = {
            "compound_id": 178,  # Next available ID
            "name": "1,3-Butadiene",
            "arabic_start_page": 1,
            "arabic_end_page": 3,
            "pdf_start_page": pdf_pages[0],
            "total_pages": 3,
            "main_entry_content": combined_content,
            "main_entry_length": len(combined_content),
            "total_references": 1,
            "reference_types_found": ["main_entry"],
            "comprehensive_text": f"=== MAIN ENTRY (Arabic pages 1-3, PDF pages {pdf_pages[0]}-{pdf_pages[-1]}) ===\n\n{combined_content}",
            "comprehensive_text_length": len(combined_content) + 100,
            "metadata": {
                "source": str(pdf_path),
                "extraction_method": "arabic_pages_1_3_additional",
                "arabic_pages": "1-3",
                "pdf_pages": f"{pdf_pages[0]}-{pdf_pages[-1]}",
                "compound_index": 178,
                "total_compounds": 178,
                "extraction_timestamp": str(Path().cwd())
            },
            "references_breakdown": {
                "main_entry": {
                    "arabic_page": 1,
                    "pdf_page": pdf_pages[0],
                    "content": combined_content,
                    "text_length": len(combined_content)
                },
                "other_references": []
            }
        }
        
        # Save individual compound JSON
        filename = "compound_178_13-Butadiene.json"
        filepath = output_dir / "individual_compounds" / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save individual compound JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(compound_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved: {filename}")
        
        # Update compound list
        compound_list_path = output_dir / "compound_names_list.json"
        
        # Load existing list
        with open(compound_list_path, 'r', encoding='utf-8') as f:
            compounds = json.load(f)
        
        # Add new compound
        new_entry = {
            "compound_id": 178,
            "name": "1,3-Butadiene",
            "start_page": 1,
            "end_page": 3,
            "total_pages": 3
        }
        compounds.append(new_entry)
        
        # Save updated list
        with open(compound_list_path, 'w', encoding='utf-8') as f:
            json.dump(compounds, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated compound list: {len(compounds)} total compounds")
        
        # Show key information
        logger.info(f"\n✅ SUCCESS! 1,3-Butadiene extracted:")
        logger.info(f"  Formula: CH2=CHCH=CH2")
        logger.info(f"  Arabic pages: 1-3")
        logger.info(f"  PDF pages: {pdf_pages[0]}-{pdf_pages[-1]}")
        logger.info(f"  Content length: {len(combined_content)} characters")
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    extract_butadiene()
