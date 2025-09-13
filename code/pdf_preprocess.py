#!/usr/bin/env python3
"""
PDF preprocessing for chemical-compounds.pdf

- Renders each page to PNG at ~200 DPI (via pdf2image) into data/processed/pdf_extracted_images
- Extracts native text and block info via PyMuPDF (fitz)
- Writes one JSONL record per page: {page_range, text, tables, images}

Usage:
  source /home/himanshu/dev/code/.venv/bin/activate
  python /home/himanshu/dev/code/pdf_preprocess.py \
    --input /home/himanshu/dev/data/raw/chemical-compounds.pdf \
    --outdir /home/himanshu/dev/data/processed
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import fitz  # PyMuPDF
    from pdf2image import convert_from_path
except ImportError:
    print("Required packages not installed:")
    print("pip install PyMuPDF pdf2image")
    exit(1)


def extract_page_content(page_doc, page_num: int) -> Dict:
    """Extract text and metadata from a PDF page"""
    page = page_doc[page_num]
    
    # Extract text
    text = page.get_text()
    
    # Extract images info
    images = []
    for img_index, img in enumerate(page.get_images()):
        images.append({
            "index": img_index,
            "xref": img[0],
            "bbox": img[1:5] if len(img) > 4 else None
        })
    
    # Extract tables (basic detection)
    tables = []
    # Note: More sophisticated table detection could be added here
    
    return {
        "page_number": page_num + 1,  # 1-indexed
        "text": text,
        "images": images,
        "tables": tables,
        "page_size": page.rect
    }


def process_pdf(input_path: str, output_dir: str):
    """Process PDF file and create image chunks + JSONL data"""
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Create output directories
    images_dir = output_dir / "pdf_extracted_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing PDF: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Images directory: {images_dir}")
    
    # Open PDF document
    doc = fitz.open(input_path)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")
    
    # Process each page
    jsonl_records = []
    
    for page_num in range(total_pages):
        print(f"Processing page {page_num + 1}/{total_pages}")
        
        # Extract content
        page_content = extract_page_content(doc, page_num)
        
        # Convert page to image
        try:
            # Convert single page to image
            images = convert_from_path(
                str(input_path), 
                first_page=page_num + 1, 
                last_page=page_num + 1,
                dpi=200
            )
            
            if images:
                image = images[0]
                image_path = images_dir / f"page_{page_num + 1}.png"
                image.save(image_path, "PNG")
                print(f"  Saved image: {image_path}")
            else:
                print(f"  Warning: No image generated for page {page_num + 1}")
                
        except Exception as e:
            print(f"  Error converting page {page_num + 1} to image: {e}")
            continue
        
        # Create JSONL record
        record = {
            "page_range": f"{page_num + 1}-{page_num + 1}",
            "text": page_content["text"],
            "images": page_content["images"],
            "tables": page_content["tables"],
            "page_size": {
                "width": page_content["page_size"].width,
                "height": page_content["page_size"].height
            },
            "image_path": str(image_path)
        }
        
        jsonl_records.append(record)
    
    # Save JSONL data
    jsonl_path = output_dir / "chemical-compounds_chunks.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete!")
    print(f"Images saved to: {images_dir}")
    print(f"JSONL data saved to: {jsonl_path}")
    print(f"Total records: {len(jsonl_records)}")
    
    doc.close()


def main():
    parser = argparse.ArgumentParser(description="Preprocess PDF into image chunks and JSONL data")
    parser.add_argument("--input", required=True, help="Input PDF file path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file does not exist: {args.input}")
        return 1
    
    try:
        process_pdf(args.input, args.outdir)
        return 0
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
