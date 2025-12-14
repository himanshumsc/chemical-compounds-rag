#!/usr/bin/env python3
"""
Additional Compound Extractor for Arabic Pages 1-18
Extract compounds from the front matter pages (Arabic 1-18) and add to existing dataset
"""

import fitz  # PyMuPDF
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdditionalCompoundExtractor:
    """Extract compounds from Arabic pages 1-18 and add to existing dataset"""
    
    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.doc = None
        self.page_mapping = {}  # Arabic page number -> PDF page number
        
    def open_pdf(self):
        """Open the PDF"""
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"PDF opened with {len(self.doc)} pages")
            return True
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            return False
    
    def build_page_mapping(self):
        """Build mapping from Arabic page numbers to PDF page numbers"""
        logger.info("Building page mapping from Arabic numerals in footers...")
        
        for pdf_page_num in range(len(self.doc)):
            page = self.doc[pdf_page_num]
            text = page.get_text()
            
            # Extract Arabic page number from footer
            arabic_page_num = self.extract_footer_page_number(text)
            
            if arabic_page_num:
                self.page_mapping[arabic_page_num] = pdf_page_num + 1  # Convert to 1-based
        
        logger.info(f"Built mapping for {len(self.page_mapping)} pages")
        logger.info(f"Arabic page range: {min(self.page_mapping.keys())} - {max(self.page_mapping.keys())}")
        
        return self.page_mapping
    
    def extract_footer_page_number(self, text: str) -> int:
        """Extract Arabic page number from footer"""
        lines = text.split('\n')
        
        # Look for Arabic numerals in the footer (usually at bottom)
        for line in reversed(lines[-5:]):  # Check last 5 lines
            line = line.strip()
            # Look for standalone numbers (Arabic page numbers)
            if line.isdigit() and 1 <= int(line) <= 1000:  # Reasonable page range
                return int(line)
        
        # Also check for page numbers with context
        for line in lines:
            # Look for patterns like "Page 99" or just "99" at end of line
            if re.search(r'\b(\d{1,3})\b', line):
                match = re.search(r'\b(\d{1,3})\b', line)
                if match:
                    page_num = int(match.group(1))
                    if 1 <= page_num <= 1000:  # Reasonable range
                        return page_num
        
        return None
    
    def get_pdf_page_for_arabic(self, arabic_page_num: int) -> int:
        """Get PDF page number for given Arabic page number"""
        return self.page_mapping.get(arabic_page_num)
    
    def extract_compounds_from_front_matter(self):
        """Extract compounds from Arabic pages 1-18"""
        logger.info("Extracting compounds from Arabic pages 1-18...")
        
        compounds = []
        
        for arabic_page in range(1, 19):  # Arabic pages 1-18
            pdf_page = self.get_pdf_page_for_arabic(arabic_page)
            
            if not pdf_page:
                logger.warning(f"Could not find PDF page for Arabic page {arabic_page}")
                continue
            
            logger.info(f"Processing Arabic page {arabic_page} (PDF page {pdf_page})")
            
            page = self.doc[pdf_page - 1]  # Convert to 0-based
            text = page.get_text()
            
            # Look for compound entries in this page
            compound_entries = self.find_compound_entries(text, arabic_page, pdf_page)
            
            if compound_entries:
                compounds.extend(compound_entries)
                logger.info(f"Found {len(compound_entries)} compound entries on Arabic page {arabic_page}")
        
        logger.info(f"Total compounds found in front matter: {len(compounds)}")
        return compounds
    
    def find_compound_entries(self, text: str, arabic_page: int, pdf_page: int):
        """Find compound entries in the text"""
        compounds = []
        
        # Look for patterns that indicate compound entries
        # Check for "KEY FACTS", "OVERVIEW", "FORMULA:" patterns
        if any(indicator in text for indicator in ['KEY FACTS', 'OVERVIEW', 'FORMULA:', 'OTHER NAMES:']):
            # This looks like a compound entry
            compound_name = self.extract_compound_name(text)
            
            if compound_name:
                compound_data = {
                    "compound_id": 175 + len(compounds),  # Start from 175 (after existing 174)
                    "name": compound_name,
                    "arabic_start_page": arabic_page,
                    "arabic_end_page": arabic_page,  # Single page for now
                    "pdf_start_page": pdf_page,
                    "total_pages": 1,
                    "main_entry_content": text,
                    "main_entry_length": len(text),
                    "total_references": 1,
                    "reference_types_found": ["main_entry"],
                    "comprehensive_text": f"=== MAIN ENTRY (Arabic page {arabic_page}, PDF page {pdf_page}) ===\n\n{text}",
                    "comprehensive_text_length": len(text) + 100,  # Approximate
                    "metadata": {
                        "source": str(self.pdf_path),
                        "extraction_method": "front_matter_additional",
                        "arabic_pages": f"{arabic_page}",
                        "pdf_pages": f"{pdf_page}",
                        "compound_index": 175 + len(compounds),
                        "total_compounds": 174 + len(compounds) + 1,
                        "extraction_timestamp": str(Path().cwd())
                    },
                    "references_breakdown": {
                        "main_entry": {
                            "arabic_page": arabic_page,
                            "pdf_page": pdf_page,
                            "content": text,
                            "text_length": len(text)
                        },
                        "other_references": []
                    }
                }
                
                compounds.append(compound_data)
                logger.info(f"Found compound: {compound_name}")
        
        return compounds
    
    def extract_compound_name(self, text: str) -> str:
        """Extract compound name from text"""
        lines = text.split('\n')
        
        # Look for compound name patterns
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and very short lines
            if len(line) < 3:
                continue
            
            # Look for lines that might be compound names
            # Skip common headers and footers
            skip_patterns = [
                'CHEMICAL COMPOUNDS', 'KEY FACTS', 'OVERVIEW', 'FORMULA:', 
                'ELEMENTS:', 'COMPOUND TYPE:', 'STATE:', 'MOLECULAR WEIGHT:',
                'MELTING POINT:', 'BOILING POINT:', 'SOLUBILITY:', 'OTHER NAMES:',
                'COMMON USES', 'HOW IT IS MADE', 'FOR FURTHER INFORMATION'
            ]
            
            if any(pattern in line.upper() for pattern in skip_patterns):
                continue
            
            # Look for potential compound names (capitalized words, chemical names)
            if (line.isupper() or line.istitle()) and len(line) > 3 and len(line) < 100:
                # Additional checks for chemical compound names
                if any(char.isdigit() for char in line):  # Contains numbers (like "C6H6")
                    continue
                if line.count(' ') > 5:  # Too many words
                    continue
                
                return line
        
        return None
    
    def load_existing_compounds(self):
        """Load existing compound list"""
        try:
            compound_list_path = self.output_dir / "compound_names_list.json"
            with open(compound_list_path, 'r', encoding='utf-8') as f:
                existing_compounds = json.load(f)
            
            logger.info(f"Loaded {len(existing_compounds)} existing compounds")
            return existing_compounds
            
        except Exception as e:
            logger.error(f"Failed to load existing compounds: {e}")
            return []
    
    def save_additional_compounds(self, additional_compounds: List[Dict]):
        """Save additional compounds as individual JSON files"""
        logger.info(f"Saving {len(additional_compounds)} additional compounds...")
        
        for compound in additional_compounds:
            # Clean the compound name for filename
            clean_name = self.clean_compound_name(compound['name'])
            compound_id = compound['compound_id']
            
            # Create filename
            filename = f"compound_{compound_id:03d}_{clean_name}.json"
            filepath = self.output_dir / "individual_compounds" / filename
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save individual compound JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(compound, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved: {filename}")
    
    def clean_compound_name(self, name: str) -> str:
        """Clean compound name for filename"""
        # Remove dots and page numbers from the end
        name = re.sub(r'\s*\.{2,}\s*\d+$', '', name)
        # Replace spaces and non-alphanumeric characters with underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Replace multiple underscores with a single one
        name = re.sub(r'_{2,}', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Truncate if too long
        if len(name) > 50:
            name = name[:50]
        return name
    
    def update_compound_list(self, additional_compounds: List[Dict]):
        """Update the compound names list with additional compounds"""
        try:
            # Load existing list
            existing_compounds = self.load_existing_compounds()
            
            # Add new compounds
            for compound in additional_compounds:
                new_entry = {
                    "compound_id": compound['compound_id'],
                    "name": compound['name'],
                    "start_page": compound['arabic_start_page'],
                    "end_page": compound['arabic_end_page'],
                    "total_pages": compound['total_pages']
                }
                existing_compounds.append(new_entry)
            
            # Save updated list
            compound_list_path = self.output_dir / "compound_names_list.json"
            with open(compound_list_path, 'w', encoding='utf-8') as f:
                json.dump(existing_compounds, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated compound list with {len(additional_compounds)} new compounds")
            logger.info(f"Total compounds now: {len(existing_compounds)}")
            
        except Exception as e:
            logger.error(f"Failed to update compound list: {e}")
    
    def extract_additional_compounds(self):
        """Main extraction function"""
        logger.info(f"\n{'='*80}")
        logger.info("EXTRACTING ADDITIONAL COMPOUNDS FROM ARABIC PAGES 1-18")
        logger.info(f"{'='*80}")
        
        # Build page mapping
        if not self.build_page_mapping():
            return False
        
        # Extract compounds from front matter
        additional_compounds = self.extract_compounds_from_front_matter()
        
        if not additional_compounds:
            logger.info("No additional compounds found in Arabic pages 1-18")
            return False
        
        # Save additional compounds
        self.save_additional_compounds(additional_compounds)
        
        # Update compound list
        self.update_compound_list(additional_compounds)
        
        logger.info(f"\n{'='*80}")
        logger.info("ADDITIONAL EXTRACTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Additional compounds found: {len(additional_compounds)}")
        
        for compound in additional_compounds:
            logger.info(f"  - {compound['name']} (Arabic page {compound['arabic_start_page']})")
        
        return True

def main():
    """Main function"""
    pdf_path = "/home/himanshu/dev/test/data/raw/chemical-compounds.pdf"
    output_dir = "/home/himanshu/dev/test/data/processed"
    
    extractor = AdditionalCompoundExtractor(pdf_path, output_dir)
    
    if not extractor.open_pdf():
        return
    
    success = extractor.extract_additional_compounds()
    
    extractor.doc.close()
    
    if success:
        logger.info(f"\n🎉 SUCCESS! Additional compounds extracted and added!")
        logger.info(f"📁 Results saved to: {output_dir}/individual_compounds/")
    else:
        logger.info(f"\n❌ No additional compounds found!")

if __name__ == "__main__":
    main()
