#!/usr/bin/env python3
"""
Update QA pairs in qa_pairs_individual_components_comprehensive by regenerating
answers for questions 2, 3, 4 using comprehensive_text with character limits.
Question 1 (image-based) is preserved unchanged.

Character Limits (matching Qwen/Gemma):
- Q2: 1,000 characters
- Q3: 1,800 characters
- Q4: 2,000 characters
"""

import json
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import openai
    from openai import RateLimitError, APIError
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Character limits matching Qwen/Gemma
CHAR_LIMIT_Q2 = 1000
CHAR_LIMIT_Q3 = 1800
CHAR_LIMIT_Q4 = 2000

# Map question index to character limit
QUESTION_LIMITS = {
    1: CHAR_LIMIT_Q2,  # Q2 (index 1)
    2: CHAR_LIMIT_Q3,  # Q3 (index 2)
    3: CHAR_LIMIT_Q4,  # Q4 (index 3)
}

SYSTEM_PROMPT = """You are a helpful assistant."""


def truncate_answer(text: str, max_chars: int) -> Tuple[str, bool]:
    """
    Truncate answer at sentence or word boundary.
    Adds '...' if truncated.
    
    Returns:
        (truncated_text, was_truncated)
    """
    if len(text) <= max_chars:
        return text, False
    
    # Try to truncate at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_exclamation = truncated.rfind('!')
    last_question = truncated.rfind('?')
    
    last_sentence_end = max(last_period, last_exclamation, last_question)
    
    if last_sentence_end > max_chars * 0.7:  # At least 70% of limit
        truncated = truncated[:last_sentence_end + 1]
        return truncated + " ...", True
    
    # Fallback: truncate at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]
        return truncated + " ...", True
    
    # Last resort: hard truncate
    return truncated + " ...", True


class ComprehensiveQAUpdaterWithLimits:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY.")
        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limit_retries = 0
        self.rate_limit_delay = 60  # 1 minute

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

    def regenerate_answer(self, question: str, compound_name: str, comprehensive_text: str, char_limit: int, max_retries: int = 3) -> str:
        """Regenerate answer for a question using comprehensive_text with character limit."""
        # Use similar prompt structure to Qwen/Gemma, but with explicit instruction to use ONLY comprehensive_text
        user_prompt = f"""Based on the following chemical compounds database information, please answer the user's question.

CONTEXT:
{comprehensive_text}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
- Your answer MUST be based ONLY on the Comprehensive Compound Information provided above
- Do not use any outside knowledge or assumptions beyond the provided text
- Your answer MUST be brief, concise, and to the point
- Maximum length: {char_limit} characters (strict limit)
- Focus ONLY on the most essential and relevant information from the provided context
- Avoid unnecessary elaboration or repetition
- Be direct and factual
- If you exceed {char_limit} characters, your answer will be truncated

Generate a concise answer that fits within {char_limit} characters, using ONLY the information from the Comprehensive Compound Information provided above:"""
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500,  # Keep same as before, truncation handles character limit
                )
                answer = response.choices[0].message.content.strip()
                
                # Truncate if exceeds limit
                if len(answer) > char_limit:
                    answer, truncated = truncate_answer(answer, char_limit)
                    if truncated:
                        logger.warning(f"Answer truncated from {len(response.choices[0].message.content)} to {len(answer)} chars")
                
                return answer
                
            except RateLimitError as e:
                self.rate_limit_retries += 1
                if attempt < max_retries - 1:
                    wait_time = self.rate_limit_delay
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit error after {max_retries} attempts: {e}")
                    raise
                    
            except APIError as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    self.rate_limit_retries += 1
                    if attempt < max_retries - 1:
                        wait_time = self.rate_limit_delay
                        logger.warning(f"Rate limit detected. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit error after {max_retries} attempts: {e}")
                        raise
                else:
                    logger.error(f"API error: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"Error regenerating answer: {e}")
                raise

    def update_qa_file(self, qa_file_path: Path, compounds_dir: Path) -> bool:
        """Update a single QA file by regenerating answers for questions 2, 3, 4 with character limits."""
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
            
            # Keep question 1 (index 0) unchanged - COPY IT
            # Regenerate answers for questions 2, 3, 4 (indices 1, 2, 3)
            updated_count = 0
            for i in [1, 2, 3]:
                if i < len(qa_pairs):
                    question = qa_pairs[i].get('question', '')
                    if question:
                        char_limit = QUESTION_LIMITS[i]
                        logger.info(f"  Regenerating answer for Q{i+1} (limit: {char_limit} chars): {question[:60]}...")
                        new_answer = self.regenerate_answer(question, compound_name, comprehensive_text, char_limit)
                        qa_pairs[i]['answer'] = new_answer
                        # Add update metadata
                        qa_pairs[i]['updated_by'] = 'comprehensive_text_generator_with_limits'
                        qa_pairs[i]['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                        qa_pairs[i]['char_limit'] = char_limit
                        qa_pairs[i]['answer_length'] = len(new_answer)
                        updated_count += 1
                        time.sleep(0.3)  # Rate limiting between questions
            
            # Update metadata
            qa_data['comprehensive_text_length'] = comprehensive_text_length
            qa_data['main_entry_length'] = compound_data.get('main_entry_length', 0)  # Keep original
            qa_data['qa_pairs'] = qa_pairs
            qa_data['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            qa_data['updated_by'] = 'comprehensive_text_generator_with_limits'
            qa_data['note'] = "Questions 2, 3, 4 answers regenerated using comprehensive_text with character limits (Q2:1000, Q3:1800, Q4:2000). Question 1 (image-based) preserved."
            qa_data['char_limits'] = {
                'q2': CHAR_LIMIT_Q2,
                'q3': CHAR_LIMIT_Q3,
                'q4': CHAR_LIMIT_Q4
            }
            
            # Save updated file
            with open(qa_file_path, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  Updated {updated_count} answers for {compound_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {qa_file_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            "rate_limit_retries": 0,
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
        results["rate_limit_retries"] = self.rate_limit_retries
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
        updater = ComprehensiveQAUpdaterWithLimits()
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}")
        return
    
    logger.info("="*70)
    logger.info("Regenerating OpenAI Q2-Q4 Answers with Character Limits")
    logger.info("="*70)
    logger.info(f"Character Limits: Q2={CHAR_LIMIT_Q2}, Q3={CHAR_LIMIT_Q3}, Q4={CHAR_LIMIT_Q4}")
    logger.info(f"Q1: Preserved (not regenerated)")
    logger.info(f"QA Directory: {qa_dir}")
    logger.info(f"Compounds Directory: {compounds_dir}")
    logger.info(f"Rate limit retry delay: {updater.rate_limit_delay} seconds")
    logger.info("="*70)
    
    # Process all files
    results = updater.process_all(qa_dir, compounds_dir, max_files=None)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE QA UPDATE WITH LIMITS - RESULTS")
    print("="*70)
    print(f"Total files processed: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Rate limit retries: {results['rate_limit_retries']}")
    print(f"Duration: {results['duration']:.2f} seconds ({results['duration']/60:.2f} minutes)")
    if results['failed_files']:
        print(f"\nFailed files ({len(results['failed_files'])}):")
        for f in results['failed_files'][:10]:  # Show first 10
            print(f"  - {f}")
        if len(results['failed_files']) > 10:
            print(f"  ... and {len(results['failed_files']) - 10} more")
    print(f"\nOutput directory: {qa_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

