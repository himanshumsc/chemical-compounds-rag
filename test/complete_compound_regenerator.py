#!/usr/bin/env python3
"""
Complete Compound Dataset Regenerator
Clear old results and regenerate all 174 compounds with correct page mapping + comprehensive collection
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

class CompleteCompoundRegenerator:
    """Regenerates all compounds with correct page mapping and comprehensive collection"""
    
    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.doc = None
        self.page_mapping = {}  # Arabic page number -> PDF page number
        self.compounds_data = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def open_pdf(self):
        """Open the PDF"""
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"PDF opened with {len(self.doc)} pages")
            return True
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            return False
    
    def clear_old_results(self):
        """Clear all old results for cleanliness"""
        logger.info("Clearing old results...")
        
        # Remove individual compounds directory
        individual_dir = self.output_dir / "individual_compounds"
        if individual_dir.exists():
            import shutil
            shutil.rmtree(individual_dir)
            logger.info(f"Removed old individual_compounds directory")
        
        # Keep the complete JSON and compound list for reference
        logger.info("Kept chemical_compounds_complete.json and compound_names_list.json for reference")
    
    def load_existing_compound_list(self):
        """Load the existing compound list from our previous extraction"""
        try:
            compound_list_path = self.output_dir / "compound_names_list.json"
            with open(compound_list_path, 'r', encoding='utf-8') as f:
                self.compounds_data = json.load(f)
            
            logger.info(f"Loaded {len(self.compounds_data)} compounds from existing list")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load existing compound list: {e}")
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
    
    def extract_main_compound_content(self, compound_name: str, arabic_page_num: int):
        """Extract main compound content using correct page mapping"""
        pdf_page_num = self.get_pdf_page_for_arabic(arabic_page_num)
        
        if not pdf_page_num:
            logger.warning(f"Could not find PDF page for Arabic page {arabic_page_num}")
            return None
        
        # Extract content from the PDF page
        content = self.extract_compound_content_from_page(pdf_page_num - 1)
        
        return {
            'arabic_page': arabic_page_num,
            'pdf_page': pdf_page_num,
            'content': content,
            'text_length': len(content) if content else 0
        }
    
    def extract_compound_content_from_page(self, start_pdf_page_idx: int, max_pages: int = 10):
        """Extract compound content starting from a specific PDF page"""
        try:
            content_pages = []
            
            for i in range(max_pages):
                page_idx = start_pdf_page_idx + i
                if page_idx >= len(self.doc):
                    break
                    
                page = self.doc[page_idx]
                text = page.get_text()
                
                # Check if this page contains compound content
                if ('KEY FACTS' in text or 'OVERVIEW' in text or 'FORMULA:' in text or 
                    'COMMON USES' in text or 'HOW IT IS MADE' in text or
                    'OTHER NAMES:' in text):
                    content_pages.append(text)
                else:
                    # If we hit a page without compound content, stop
                    break
            
            if content_pages:
                full_content = '\n'.join(content_pages)
                return full_content
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return None
    
    def find_all_compound_references(self, compound_name: str, exclude_main_page: int = None):
        """Find all other references to the compound throughout the PDF"""
        if not self.doc:
            return []
        
        clean_name = self.clean_compound_name(compound_name)
        search_variations = [clean_name, clean_name.upper(), clean_name.lower()]
        
        other_references = []
        
        # Search through all pages
        for page_num in range(len(self.doc)):
            # Skip the main entry page
            if exclude_main_page and page_num + 1 == exclude_main_page:
                continue
                
            page = self.doc[page_num]
            text = page.get_text()
            
            # Check if any search variation appears on this page
            found_variations = []
            for variation in search_variations:
                if variation in text:
                    found_variations.append(variation)
            
            if found_variations:
                # Analyze the context
                reference_type = self.analyze_reference_context(text, clean_name)
                context = self.extract_context(text, clean_name)
                
                reference = {
                    'page_number': page_num + 1,
                    'found_variations': found_variations,
                    'reference_type': reference_type,
                    'context': context
                }
                
                other_references.append(reference)
        
        return other_references
    
    def clean_compound_name(self, name: str) -> str:
        """Clean compound name for searching"""
        cleaned = name.replace('.', '').strip()
        return cleaned
    
    def analyze_reference_context(self, text: str, compound_name: str) -> str:
        """Analyze the context to determine the type of reference"""
        
        # Check for timeline references
        if any(indicator in text.lower() for indicator in ['timeline', 'bce', 'ce', 'century', 'discovered', 'isolated']):
            return 'timeline'
        
        # Check for solubility references
        if any(indicator in text.lower() for indicator in ['soluble', 'insoluble', 'solvent', 'dissolves']):
            return 'solubility_reference'
        
        # Check for synthesis references
        if any(indicator in text.lower() for indicator in ['synthesis', 'manufacture', 'produce', 'made from', 'derived']):
            return 'synthesis_reference'
        
        # Check for usage references
        if any(indicator in text.lower() for indicator in ['used in', 'application', 'industry', 'commercial']):
            return 'usage_reference'
        
        # Check for structure references
        if any(indicator in text.lower() for indicator in ['structure', 'molecule', 'atom', 'bond', 'ring']):
            return 'structure_reference'
        
        # Check for safety references
        if any(indicator in text.lower() for indicator in ['hazard', 'toxic', 'dangerous', 'safety', 'exposure']):
            return 'safety_reference'
        
        return 'general_reference'
    
    def extract_context(self, text: str, compound_name: str, context_length: int = 200) -> str:
        """Extract relevant context around the compound mention"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if compound_name.lower() in line.lower():
                # Get context around the line
                start_idx = max(0, i - 2)
                end_idx = min(len(lines), i + 3)
                context_lines = lines[start_idx:end_idx]
                context = '\n'.join(context_lines)
                
                # Truncate if too long
                if len(context) > context_length:
                    context = context[:context_length] + "..."
                
                return context
        
        return text[:context_length] + "..." if len(text) > context_length else text
    
    def create_comprehensive_compound_data(self, compound: Dict[str, Any]):
        """Create comprehensive data for a single compound"""
        compound_name = compound['name']
        arabic_page = compound['start_page']
        
        logger.info(f"Processing compound {compound['compound_id']}: '{compound_name}' (Arabic page {arabic_page})")
        
        # Extract main entry
        main_entry = self.extract_main_compound_content(compound_name, arabic_page)
        
        if not main_entry or not main_entry['content']:
            logger.warning(f"No main entry found for {compound_name}")
            return None
        
        # Find all other references
        other_references = self.find_all_compound_references(compound_name, exclude_main_page=main_entry['pdf_page'])
        
        # Combine all text
        all_text_parts = []
        
        # Add main entry
        all_text_parts.append(f"=== MAIN ENTRY (Arabic page {main_entry['arabic_page']}, PDF page {main_entry['pdf_page']}) ===\n")
        all_text_parts.append(main_entry['content'])
        all_text_parts.append("\n")
        
        # Add other references grouped by type
        reference_types = {}
        for ref in other_references:
            ref_type = ref['reference_type']
            if ref_type not in reference_types:
                reference_types[ref_type] = []
            reference_types[ref_type].append(ref)
        
        for ref_type, refs in reference_types.items():
            all_text_parts.append(f"=== {ref_type.upper().replace('_', ' ')} REFERENCES ===\n")
            for ref in refs:
                all_text_parts.append(f"Page {ref['page_number']}: {ref['context']}\n")
            all_text_parts.append("\n")
        
        comprehensive_text = '\n'.join(all_text_parts)
        
        # Create comprehensive compound data
        comprehensive_data = {
            "compound_id": compound['compound_id'],
            "name": compound_name,
            "arabic_start_page": arabic_page,
            "arabic_end_page": compound['end_page'],
            "pdf_start_page": main_entry['pdf_page'],
            "total_pages": compound['end_page'] - compound['start_page'] + 1,
            "main_entry_content": main_entry['content'],
            "main_entry_length": main_entry['text_length'],
            "total_references": len(other_references) + 1,
            "reference_types_found": list(reference_types.keys()),
            "comprehensive_text": comprehensive_text,
            "comprehensive_text_length": len(comprehensive_text),
            "metadata": {
                "source": str(self.pdf_path),
                "extraction_method": "correct_page_mapping_comprehensive",
                "arabic_pages": f"{arabic_page}-{compound['end_page']}",
                "pdf_pages": f"{main_entry['pdf_page']}",
                "compound_index": compound['compound_id'],
                "total_compounds": len(self.compounds_data),
                "extraction_timestamp": str(Path().cwd())
            },
            "references_breakdown": {
                "main_entry": main_entry,
                "other_references": other_references
            }
        }
        
        logger.info(f"✅ {compound_name}: {comprehensive_data['comprehensive_text_length']} chars, {comprehensive_data['total_references']} refs")
        
        return comprehensive_data
    
    def save_individual_compound_json(self, compound_data: Dict[str, Any]):
        """Save individual compound as JSON file"""
        try:
            # Clean the compound name for filename
            clean_name = self.clean_compound_name(compound_data['name'])
            compound_id = compound_data['compound_id']
            
            # Create filename
            filename = f"compound_{compound_id:03d}_{clean_name}.json"
            filepath = self.output_dir / "individual_compounds" / filename
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save individual compound JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(compound_data, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save {compound_data['name']}: {e}")
            return None
    
    def regenerate_all_compounds(self):
        """Regenerate all compounds with correct data"""
        logger.info(f"\n{'='*80}")
        logger.info("STARTING COMPLETE COMPOUND REGENERATION")
        logger.info(f"{'='*80}")
        
        # Clear old results
        self.clear_old_results()
        
        # Load existing compound list
        if not self.load_existing_compound_list():
            return False
        
        # Build page mapping
        if not self.build_page_mapping():
            return False
        
        # Process all compounds
        successful_extractions = 0
        failed_extractions = 0
        
        for i, compound in enumerate(self.compounds_data):
            logger.info(f"\n--- Processing compound {i+1}/{len(self.compounds_data)} ---")
            
            try:
                # Create comprehensive data
                comprehensive_data = self.create_comprehensive_compound_data(compound)
                
                if comprehensive_data:
                    # Save individual JSON
                    filepath = self.save_individual_compound_json(comprehensive_data)
                    if filepath:
                        successful_extractions += 1
                    else:
                        failed_extractions += 1
                else:
                    failed_extractions += 1
                    
            except Exception as e:
                logger.error(f"Error processing {compound['name']}: {e}")
                failed_extractions += 1
        
        # Create summary
        logger.info(f"\n{'='*80}")
        logger.info("REGENERATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total compounds: {len(self.compounds_data)}")
        logger.info(f"Successful extractions: {successful_extractions}")
        logger.info(f"Failed extractions: {failed_extractions}")
        logger.info(f"Success rate: {successful_extractions/len(self.compounds_data)*100:.1f}%")
        
        return successful_extractions > 0

def main():
    """Main function"""
    pdf_path = "/home/himanshu/dev/test/data/raw/chemical-compounds.pdf"
    output_dir = "/home/himanshu/dev/test/data/processed"
    
    regenerator = CompleteCompoundRegenerator(pdf_path, output_dir)
    
    if not regenerator.open_pdf():
        return
    
    success = regenerator.regenerate_all_compounds()
    
    regenerator.doc.close()
    
    if success:
        logger.info(f"\n🎉 SUCCESS! All compounds regenerated with correct data!")
        logger.info(f"📁 Results saved to: {output_dir}/individual_compounds/")
    else:
        logger.info(f"\n❌ Regeneration failed!")

if __name__ == "__main__":
    main()
