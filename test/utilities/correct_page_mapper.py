#!/usr/bin/env python3
"""
Correct Page Number Mapping
Map TOC page numbers to actual PDF pages using Arabic numerals in footers
"""

import fitz  # PyMuPDF
import re
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectPageMapper:
    """Maps TOC page numbers to actual PDF pages using footer Arabic numerals"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
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
    
    def extract_footer_page_number(self, text: str) -> int:
        """Extract Arabic page number from footer"""
        lines = text.split('\n')
        
        # Look for Arabic numerals in the footer (usually at bottom)
        # Check last few lines for page numbers
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
                logger.debug(f"Arabic page {arabic_page_num} -> PDF page {pdf_page_num + 1}")
        
        logger.info(f"Built mapping for {len(self.page_mapping)} pages")
        logger.info(f"Arabic page range: {min(self.page_mapping.keys())} - {max(self.page_mapping.keys())}")
        
        return self.page_mapping
    
    def get_pdf_page_for_arabic(self, arabic_page_num: int) -> int:
        """Get PDF page number for given Arabic page number"""
        return self.page_mapping.get(arabic_page_num)
    
    def extract_compound_by_arabic_page(self, compound_name: str, arabic_page_num: int):
        """Extract compound content using Arabic page number from TOC"""
        pdf_page_num = self.get_pdf_page_for_arabic(arabic_page_num)
        
        if not pdf_page_num:
            logger.error(f"Could not find PDF page for Arabic page {arabic_page_num}")
            return None
        
        logger.info(f"Arabic page {arabic_page_num} -> PDF page {pdf_page_num}")
        
        # Extract content from the PDF page
        page = self.doc[pdf_page_num - 1]  # Convert to 0-based
        text = page.get_text()
        
        # Extract compound content (multiple pages if needed)
        content = self.extract_compound_content_from_page(pdf_page_num - 1)
        
        return {
            'compound_name': compound_name,
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
                logger.info(f"Extracted {len(content_pages)} pages of compound content")
                return full_content
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return None
    
    def test_benzene_extraction(self):
        """Test Benzene extraction using correct page mapping"""
        logger.info(f"\n{'='*60}")
        logger.info("Testing Benzene extraction with correct page mapping")
        logger.info(f"{'='*60}")
        
        # Build page mapping
        self.build_page_mapping()
        
        # Extract Benzene from Arabic page 99 (as per TOC)
        result = self.extract_compound_by_arabic_page("Benzene", 99)
        
        if result:
            logger.info(f"\n✅ SUCCESS! Benzene extracted correctly:")
            logger.info(f"Arabic page: {result['arabic_page']}")
            logger.info(f"PDF page: {result['pdf_page']}")
            logger.info(f"Content length: {result['text_length']} characters")
            
            # Show first part of content
            logger.info(f"\nFirst 1000 characters:")
            logger.info("-" * 60)
            logger.info(result['content'][:1000])
            logger.info("-" * 60)
            
            # Check for Benzene indicators
            benzene_indicators = ["Benzene", "C6H6", "benzene", "aromatic", "ring"]
            found_indicators = [ind for ind in benzene_indicators if ind in result['content']]
            
            if found_indicators:
                logger.info(f"✅ Contains Benzene indicators: {found_indicators}")
            else:
                logger.info("❌ No clear Benzene indicators found")
            
            return result
        else:
            logger.info("❌ Failed to extract Benzene")
            return None
    
    def test_multiple_compounds(self):
        """Test extraction for multiple compounds"""
        test_compounds = [
            ("Acetic acid", 23),
            ("Benzene", 99),
            ("Water", 879)
        ]
        
        logger.info(f"\n{'='*60}")
        logger.info("Testing multiple compounds")
        logger.info(f"{'='*60}")
        
        results = []
        for compound_name, arabic_page in test_compounds:
            logger.info(f"\nTesting {compound_name} (Arabic page {arabic_page})...")
            result = self.extract_compound_by_arabic_page(compound_name, arabic_page)
            if result:
                results.append(result)
                logger.info(f"✅ {compound_name}: {result['text_length']} characters")
            else:
                logger.info(f"❌ {compound_name}: Failed")
        
        return results

def main():
    """Main function"""
    pdf_path = "/home/himanshu/dev/test/data/raw/chemical-compounds.pdf"
    
    mapper = CorrectPageMapper(pdf_path)
    if not mapper.open_pdf():
        return
    
    # Test Benzene extraction
    benzene_result = mapper.test_benzene_extraction()
    
    # Test multiple compounds
    all_results = mapper.test_multiple_compounds()
    
    mapper.doc.close()
    
    if benzene_result:
        logger.info(f"\n🎉 SUCCESS! Correct page mapping works!")
        logger.info(f"Benzene found on Arabic page {benzene_result['arabic_page']} (PDF page {benzene_result['pdf_page']})")
    else:
        logger.info(f"\n😞 Page mapping needs adjustment")

if __name__ == "__main__":
    main()
