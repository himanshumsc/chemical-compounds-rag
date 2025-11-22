#!/usr/bin/env python3
"""
Generate QA pairs for each JSON in individual_compounds without generating images.
Output files go to data/processed/qa_pairs_individual_components with filename qa_<CompoundName>.json
image_path is set to the plain compound name to be linked later.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
import openai
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert chemistry educator creating high-quality, multi-modal question-answer pairs for chemical compounds. Your task is to generate exactly 3 diverse, educational QA pairs that test different aspects of understanding about the given compound.

REQUIREMENTS:
1. Generate exactly 3 QA pairs
2. Each QA pair should test different cognitive levels (factual, conceptual, applied)
3. Questions should be clear, specific, and educational
4. Answers should be comprehensive, accurate, and educational
5. Include relevant chemical formulas, properties, and applications when appropriate
6. Make questions progressively more challenging

BOUNDARY CONDITION:
All questions and answers must be based ONLY on the provided Compound Information text. Do not use outside knowledge or assumptions beyond that text.

OUTPUT FORMAT:
Return ONLY a valid JSON array with exactly 3 objects, each containing:
{
  "question": "Clear, specific question about the compound",
  "answer": "Comprehensive, educational answer",
  "difficulty_level": "basic|intermediate|advanced",
  "topic_category": "formula|properties|structure|uses|safety|reactions"
}

IMPORTANT: Return ONLY the JSON array, no additional text or explanations."""


def create_image_identification_qa(compound_name: str, compound_content: str) -> Dict[str, Any]:
    clean_name = compound_name.replace('.', '').strip()
    lines = compound_content.split('\n')
    formula = ""
    compound_type = ""
    state = ""
    molecular_weight = ""
    for line in lines:
        if 'FORMULA:' in line:
            formula = line.split('FORMULA:')[1].strip()
        elif 'COMPOUND TYPE:' in line:
            compound_type = line.split('COMPOUND TYPE:')[1].strip()
        elif 'STATE:' in line:
            state = line.split('STATE:')[1].strip()
        elif 'MOLECULAR WEIGHT:' in line:
            molecular_weight = line.split('MOLECULAR WEIGHT:')[1].strip()

    parts: List[str] = [f"The compound shown in the image is {clean_name}."]
    if formula:
        parts.append(f"Its chemical formula is {formula}.")
    if compound_type:
        parts.append(f"It is a {compound_type.lower()}.")
    if state:
        parts.append(f"At room temperature, it exists as a {state.lower()}.")
    if molecular_weight:
        parts.append(f"Its molecular weight is {molecular_weight}.")

    if compound_type.lower().find('acid') != -1:
        parts.append("As an acid, it can donate protons (H⁺) in aqueous solutions.")
    elif compound_type.lower().find('alcohol') != -1:
        parts.append("As an alcohol, it contains hydroxyl (-OH) functional groups.")
    elif compound_type.lower().find('aromatic') != -1:
        parts.append("As an aromatic compound, it has delocalized π-electrons and added stability.")
    elif compound_type.lower().find('ionic') != -1:
        parts.append("As an ionic compound, it consists of ions held by electrostatic forces.")

    return {
        "question": "Look at the molecular structure diagram in the image. What chemical compound is shown, and what are its key properties?",
        "answer": " ".join(parts),
        "difficulty_level": "basic",
        "topic_category": "identification"
    }


class QAOnlyGenerator:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY.")
        self.client = openai.OpenAI(api_key=api_key)

    def generate_qa_pairs(self, compound_content: str, compound_name: str) -> List[Dict[str, Any]]:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate 3 educational QA pairs for the compound: {compound_name}. Base all questions and answers strictly on the provided Compound Information; do not use outside knowledge.\n\nCompound Information:\n{compound_content}"}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        qa_pairs = json.loads(content)
        if not isinstance(qa_pairs, list) or len(qa_pairs) != 3:
            raise ValueError("Expected exactly 3 QA pairs")
        # Prepend image identification QA (we will link images later)
        qa_pairs.insert(0, create_image_identification_qa(compound_name, compound_content))
        return qa_pairs

    def process_compound_file(self, json_file_path: Path, output_dir: Path) -> bool:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            compound_data = json.load(f)
        raw_name = compound_data.get('name', 'Unknown')
        # Remove dot leaders and normalize whitespace
        compound_name = re.sub(r"\.", "", raw_name)
        compound_name = " ".join(compound_name.split()).strip()
        main_entry_content = compound_data.get('main_entry_content', '')
        if not main_entry_content:
            logger.warning(f"No main_entry_content found for {compound_name}")
            return False
        qa_pairs = self.generate_qa_pairs(main_entry_content, compound_name)
        output_data = {
            "compound_id": compound_data.get('compound_id'),
            "compound_name": compound_name,
            "source_file": json_file_path.name,
            "main_entry_length": compound_data.get('main_entry_length', 0),
            "qa_pairs_count": len(qa_pairs),
            "image_path": compound_name,  # per requirement: just the name (cleaned)
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qa_pairs": qa_pairs,
            "note": "Image path is the compound name only; link to image index later."
        }
        sanitized = compound_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        output_filename = f"qa_{sanitized}.json"
        output_file_path = output_dir / output_filename
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved QA for {compound_name} -> {output_filename}")
        return True

    def process_all(self, compounds_dir: Path, output_dir: Path, max_files: int | None = None) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(list(compounds_dir.glob('*.json')))
        if max_files:
            files = files[:max_files]
        results = {"total_files": len(files), "successful": 0, "failed": 0, "failed_files": [], "start_time": time.time()}
        for i, path in enumerate(files, 1):
            logger.info(f"Processing {i}/{len(files)}: {path.name}")
            try:
                ok = self.process_compound_file(path, output_dir)
                results["successful"] += 1 if ok else 0
                results["failed"] += 0 if ok else 1
                if not ok:
                    results["failed_files"].append(path.name)
            except Exception as e:
                logger.error(f"Failed {path.name}: {e}")
                results["failed"] += 1
                results["failed_files"].append(path.name)
            time.sleep(0.5)
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        return results


def main():
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    output_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs_individual_components")
    if not compounds_dir.exists():
        logger.error(f"Compounds directory not found: {compounds_dir}")
        return
    try:
        gen = QAOnlyGenerator()
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}")
        return
    logger.info("Generating QA (no images) for individual_compounds...")
    results = gen.process_all(compounds_dir, output_dir, max_files=None)
    print("\n" + "="*60)
    print("QA GENERATION RESULTS - INDIVIDUAL COMPONENTS (NO IMAGES)")
    print("="*60)
    print(f"Total files processed: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    if results['failed_files']:
        print("Failed files:")
        for f in results['failed_files']:
            print(f"  - {f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()


