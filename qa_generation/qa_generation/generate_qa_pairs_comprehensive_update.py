#!/usr/bin/env python3
"""
Update QA pairs in qa_pairs_individual_components_comprehensive by regenerating
answers for questions 2, 3, 4 using comprehensive_text instead of main_entry_content.
Question 1 (image-based) is preserved unchanged.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import openai
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert chemistry educator providing comprehensive, accurate answers to questions about chemical compounds.

REQUIREMENTS:
1. Answer the question based ONLY on the provided Comprehensive Compound Information
2. Use all relevant information from the comprehensive text, including timeline references and cross-references
3. Provide detailed, educational answers that demonstrate deep understanding
4. Include relevant chemical formulas, properties, historical context, and applications when available
5. If information is not in the provided text, do not make assumptions

BOUNDARY CONDITION:
All answers must be based ONLY on the provided Comprehensive Compound Information text. Do not use outside knowledge or assumptions beyond that text.

OUTPUT FORMAT:
Return ONLY the answer text, no additional explanations or formatting."""


class ComprehensiveQAUpdater:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY.")
        self.client = openai.OpenAI(api_key=api_key)

    def find_compound_file(self, qa_data: Dict[str, Any], compounds_dir: Path) -> Optional[Path]:
        """Find the corresponding compound file by compound_id or source_file."""
        compound_id = qa_data.get('compound_id')
        source_file = qa_data.get('source_file', '')
        
        # Try to find by compound_id first
        if compound_id:
            pattern = f"compound_{compound_id:03d}_*.json"
            matches = list(compounds_dir.glob(pattern))
            if matches:
                return matches[0]
        
        # Fall back to source_file name
        if source_file:
            compound_path = compounds_dir / source_file
            if compound_path.exists():
                return compound_path
        
        return None

    def regenerate_answer(self, question: str, compound_name: str, comprehensive_text: str) -> str:
        """Regenerate answer for a question using comprehensive_text."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\nComprehensive Compound Information for {compound_name}:\n{comprehensive_text}\n\nAnswer the question based on the comprehensive information provided above."}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            logger.error(f"Error regenerating answer: {e}")
            raise

    def update_qa_file(self, qa_file_path: Path, compounds_dir: Path) -> bool:
        """Update a single QA file by regenerating answers for questions 2, 3, 4."""
        try:
            # Load existing QA file
            with open(qa_file_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            compound_name = qa_data.get('compound_name', 'Unknown')
            logger.info(f"Processing {qa_file_path.name} - {compound_name}")
            
            # Find corresponding compound file
            compound_file = self.find_compound_file(qa_data, compounds_dir)
            if not compound_file:
                logger.warning(f"Could not find compound file for {qa_file_path.name}")
                return False
            
            # Load compound data
            with open(compound_file, 'r', encoding='utf-8') as f:
                compound_data = json.load(f)
            
            comprehensive_text = compound_data.get('comprehensive_text', '')
            comprehensive_text_length = compound_data.get('comprehensive_text_length', 0)
            
            if not comprehensive_text:
                logger.warning(f"No comprehensive_text found for {compound_name}")
                return False
            
            # Get existing QA pairs
            qa_pairs = qa_data.get('qa_pairs', [])
            if len(qa_pairs) < 4:
                logger.warning(f"Expected at least 4 QA pairs, found {len(qa_pairs)} for {compound_name}")
                return False
            
            # Keep question 1 (index 0) unchanged
            # Regenerate answers for questions 2, 3, 4 (indices 1, 2, 3)
            updated_count = 0
            for i in [1, 2, 3]:
                if i < len(qa_pairs):
                    question = qa_pairs[i].get('question', '')
                    if question:
                        logger.info(f"  Regenerating answer for Q{i+1}: {question[:60]}...")
                        new_answer = self.regenerate_answer(question, compound_name, comprehensive_text)
                        qa_pairs[i]['answer'] = new_answer
                        # Add update metadata
                        qa_pairs[i]['updated_by'] = 'comprehensive_text_generator'
                        qa_pairs[i]['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                        updated_count += 1
                        time.sleep(0.3)  # Rate limiting
            
            # Update metadata
            qa_data['comprehensive_text_length'] = comprehensive_text_length
            qa_data['main_entry_length'] = compound_data.get('main_entry_length', 0)  # Keep original
            qa_data['qa_pairs'] = qa_pairs
            qa_data['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            qa_data['updated_by'] = 'comprehensive_text_generator'
            qa_data['note'] = "Questions 2, 3, 4 answers regenerated using comprehensive_text. Question 1 (image-based) preserved."
            
            # Save updated file
            with open(qa_file_path, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  Updated {updated_count} answers for {compound_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {qa_file_path.name}: {e}")
            return False

    def process_all(self, qa_dir: Path, compounds_dir: Path, max_files: int | None = None) -> Dict[str, Any]:
        """Process all QA files in the directory."""
        if not qa_dir.exists():
            logger.error(f"QA directory not found: {qa_dir}")
            return {}
        
        if not compounds_dir.exists():
            logger.error(f"Compounds directory not found: {compounds_dir}")
            return {}
        
        files = sorted(list(qa_dir.glob('*.json')))
        if max_files:
            files = files[:max_files]
        
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "failed_files": [],
            "start_time": time.time()
        }
        
        for i, qa_file in enumerate(files, 1):
            logger.info(f"Processing {i}/{len(files)}: {qa_file.name}")
            try:
                ok = self.update_qa_file(qa_file, compounds_dir)
                if ok:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["failed_files"].append(qa_file.name)
            except Exception as e:
                logger.error(f"Failed {qa_file.name}: {e}")
                results["failed"] += 1
                results["failed_files"].append(qa_file.name)
            
            # Rate limiting between files
            if i < len(files):
                time.sleep(0.5)
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        return results


def main():
    qa_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components_comprehensive")
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    
    if not qa_dir.exists():
        logger.error(f"QA directory not found: {qa_dir}")
        return
    
    if not compounds_dir.exists():
        logger.error(f"Compounds directory not found: {compounds_dir}")
        return
    
    try:
        updater = ComprehensiveQAUpdater()
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}")
        return
    
    logger.info("Updating QA pairs with comprehensive_text answers...")
    logger.info(f"QA Directory: {qa_dir}")
    logger.info(f"Compounds Directory: {compounds_dir}")
    
    # Process all files
    results = updater.process_all(qa_dir, compounds_dir, max_files=None)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE QA UPDATE RESULTS")
    print("="*60)
    print(f"Total files processed: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Duration: {results['duration']:.2f} seconds ({results['duration']/60:.2f} minutes)")
    if results['failed_files']:
        print(f"\nFailed files ({len(results['failed_files'])}):")
        for f in results['failed_files'][:10]:  # Show first 10
            print(f"  - {f}")
        if len(results['failed_files']) > 10:
            print(f"  ... and {len(results['failed_files']) - 10} more")
    print(f"\nOutput directory: {qa_dir}")


if __name__ == "__main__":
    main()

