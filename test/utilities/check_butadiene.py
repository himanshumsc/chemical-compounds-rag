#!/usr/bin/env python3
"""
Check for 1,3-Butadiene on Arabic pages 1-3
"""

import fitz  # PyMuPDF
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_butadiene_pages():
    """Check Arabic pages 1-3 for 1,3-Butadiene"""
    
    pdf_path = "/home/himanshu/dev/test/data/raw/chemical-compounds.pdf"
    
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
        
        logger.info(f"Built mapping for {len(page_mapping)} pages")
        
        # Check Arabic pages 1-3
        for arabic_page in [1, 2, 3]:
            pdf_page = page_mapping.get(arabic_page)
            
            if pdf_page:
                logger.info(f"\n=== Arabic page {arabic_page} (PDF page {pdf_page}) ===")
                page = doc[pdf_page - 1]  # Convert to 0-based
                text = page.get_text()
                
                # Check for 1,3-Butadiene
                if "1,3-Butadiene" in text or "butadiene" in text.lower():
                    logger.info(f"✅ FOUND 1,3-Butadiene on Arabic page {arabic_page}!")
                    
                    # Show context around the mention
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        if "1,3-Butadiene" in line or "butadiene" in line.lower():
                            logger.info(f"Line {i+1}: {line}")
                            # Show surrounding context
                            start = max(0, i-3)
                            end = min(len(lines), i+4)
                            logger.info("Context:")
                            for j in range(start, end):
                                marker = ">>> " if j == i else "    "
                                logger.info(f"{marker}{j+1}: {lines[j]}")
                            break
                else:
                    logger.info(f"❌ No 1,3-Butadiene found on Arabic page {arabic_page}")
                    
                # Show first few lines to understand content
                logger.info(f"First 10 lines of Arabic page {arabic_page}:")
                lines = text.split('\n')
                for i, line in enumerate(lines[:10]):
                    if line.strip():
                        logger.info(f"  {i+1}: {line}")
            else:
                logger.info(f"❌ Could not find PDF page for Arabic page {arabic_page}")
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    check_butadiene_pages()
