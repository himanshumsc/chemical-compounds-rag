#!/usr/bin/env python3
"""
Update QA pairs in qa_pairs_individual_components_comprehensive by regenerating
answers for questions 2, 3, 4 using comprehensive_text in a SINGLE API call.
Question 1 (image-based) is preserved unchanged.

OPTIMIZATION: Sends comprehensive_text once and asks all 3 questions together,
reducing token usage by ~65% and API calls from 3 to 1 per file.
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
1. Answer each question based ONLY on the provided Comprehensive Compound Information
2. Use all relevant information from the comprehensive text, including timeline references and cross-references
3. Provide detailed, educational answers that demonstrate deep understanding
4. Include relevant chemical formulas, properties, historical context, and applications when available
5. If information is not in the provided text, do not make assumptions

BOUNDARY CONDITION:
All answers must be based ONLY on the provided Comprehensive Compound Information text. Do not use outside knowledge or assumptions beyond that text.

OUTPUT FORMAT:
Return a valid JSON object with exactly 3 answers, one for each question, in this format:
{
  "answer_1": "Answer to question 1",
  "answer_2": "Answer to question 2",
  "answer_3": "Answer to question 3"
}

Return ONLY the JSON object, no additional text or explanations."""


class ComprehensiveQABatchUpdater:
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

    def regenerate_answers_batch(self, questions: List[str], compound_name: str, comprehensive_text: str) -> List[str]:
        """Regenerate answers for all 3 questions in a single API call."""
        try:
            # Format all 3 questions
            questions_text = "\n\n".join([f"Question {i+1}: {q}" for i, q in enumerate(questions)])
            
            user_content = f"""Please answer the following 3 questions about {compound_name} based on the comprehensive information provided below.

{questions_text}

Comprehensive Compound Information for {compound_name}:
{comprehensive_text}

Provide answers to all 3 questions based on the comprehensive information above."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=1500,  # 3 answers × 500 tokens each
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            answers_json = json.loads(content)
            
            # Extract answers in order
            answers = [
                answers_json.get('answer_1', ''),
                answers_json.get('answer_2', ''),
                answers_json.get('answer_3', '')
            ]
            
            # Validate we got 3 answers
            if len(answers) != 3 or not all(answers):
                raise ValueError(f"Expected 3 answers, got {len([a for a in answers if a])}")
            
            return answers
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response content: {content[:500]}")
            raise
        except Exception as e:
            logger.error(f"Error regenerating answers: {e}")
            raise

    def update_qa_file(self, qa_file_path: Path, compounds_dir: Path) -> bool:
        """Update a single QA file by regenerating answers for questions 2, 3, 4 in batch."""
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
            # Get questions 2, 3, 4 (indices 1, 2, 3)
            questions = []
            for i in [1, 2, 3]:
                if i < len(qa_pairs):
                    question = qa_pairs[i].get('question', '')
                    if question:
                        questions.append(question)
            
            if len(questions) != 3:
                logger.warning(f"Expected 3 questions, found {len(questions)} for {compound_name}")
                return False
            
            # Regenerate all 3 answers in a single API call
            logger.info(f"  Regenerating answers for Q2, Q3, Q4 in batch...")
            try:
                answers = self.regenerate_answers_batch(questions, compound_name, comprehensive_text)
                
                # Update answers
                for i, answer in enumerate(answers):
                    qa_idx = i + 1  # Q2, Q3, Q4 (indices 1, 2, 3)
                    if qa_idx < len(qa_pairs):
                        qa_pairs[qa_idx]['answer'] = answer
                        qa_pairs[qa_idx]['updated_by'] = 'comprehensive_text_generator_batch'
                        qa_pairs[qa_idx]['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                logger.info(f"  Successfully updated 3 answers for {compound_name}")
                
            except Exception as e:
                logger.error(f"Error in batch regeneration: {e}")
                return False
            
            # Update metadata
            qa_data['comprehensive_text_length'] = comprehensive_text_length
            qa_data['main_entry_length'] = compound_data.get('main_entry_length', 0)  # Keep original
            qa_data['qa_pairs'] = qa_pairs
            qa_data['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            qa_data['updated_by'] = 'comprehensive_text_generator_batch'
            qa_data['note'] = "Questions 2, 3, 4 answers regenerated using comprehensive_text in batch (single API call). Question 1 (image-based) preserved."
            
            # Save updated file
            with open(qa_file_path, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {qa_file_path.name}: {e}")
            return False

    def process_all(self, qa_dir: Path, compounds_dir: Path, max_files: int | None = None, specific_files: List[str] | None = None) -> Dict[str, Any]:
        """Process QA files in the directory.
        
        Args:
            qa_dir: Directory containing QA files
            compounds_dir: Directory containing compound files
            max_files: Maximum number of files to process (None = all)
            specific_files: List of specific filenames to process (None = all files)
        """
        if not qa_dir.exists():
            logger.error(f"QA directory not found: {qa_dir}")
            return {}
        
        if not compounds_dir.exists():
            logger.error(f"Compounds directory not found: {compounds_dir}")
            return {}
        
        if specific_files:
            # Process only specified files
            files = [qa_dir / f for f in specific_files if (qa_dir / f).exists()]
            files = sorted(files)
            logger.info(f"Processing {len(files)} specific files: {specific_files}")
        else:
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
            
            # Rate limiting between files (longer delay for batch approach)
            if i < len(files):
                time.sleep(1.0)  # 1 second delay between files
        
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
        updater = ComprehensiveQABatchUpdater()
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}")
        return
    
    # List of failed files to retry (17 files that hit rate limits)
    failed_files = [
        '113_Petroleum.json',
        '140_Riboflavin.json',
        '151_Sodium_Cyclamate.json',
        '154_Sodium_Hypochlorite.json',
        '156_Sodium_Phosphate.json',
        '172_Toluene.json',
        '22_Benzene.json',
        '32_Calcium_Oxide.json',
        '40_Cellulose.json',
        '44_Chlorophyll.json',
        '5_Acetic_acid.json',
        '60_Ethyl_Alcohol.json',
        '68_Gamma-123456-Hexachlorocyclohexane.json',
        '7_Acetylsalicylic_Acid.json',
        '73_Hydrogen_Chloride.json',
        '84_Luminol.json',
        '86_Magnesium_Hydroxide.json'
    ]
    
    logger.info("Updating QA pairs with comprehensive_text answers (BATCH MODE)...")
    logger.info(f"QA Directory: {qa_dir}")
    logger.info(f"Compounds Directory: {compounds_dir}")
    logger.info("OPTIMIZATION: Sending comprehensive_text once and asking all 3 questions together")
    logger.info(f"Processing {len(failed_files)} failed files only")
    
    results = updater.process_all(qa_dir, compounds_dir, max_files=None, specific_files=failed_files)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE QA UPDATE RESULTS (BATCH MODE)")
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
    print("\nToken Savings: ~65% reduction (comprehensive_text sent once instead of 3 times)")


if __name__ == "__main__":
    main()

