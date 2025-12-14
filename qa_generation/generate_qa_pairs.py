#!/usr/bin/env python3
"""
QA Generation Script for Chemical Compounds
Generates multi-modal QA pairs using GPT-4o from compound main_entry_content.
"""

import json
import os
import time
from pathlib import Path
import openai
from typing import List, Dict, Any
import logging
import requests
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QAGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the QA Generator with OpenAI API key."""
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = openai.OpenAI(api_key=api_key)
        
        self.system_prompt = """You are an expert chemistry educator creating high-quality, multi-modal question-answer pairs for chemical compounds. Your task is to generate exactly 3 diverse, educational QA pairs that test different aspects of understanding about the given compound.

REQUIREMENTS:
1. Generate exactly 3 QA pairs
2. Each QA pair should test different cognitive levels (factual, conceptual, applied)
3. Questions should be clear, specific, and educational
4. Answers should be comprehensive, accurate, and educational
5. Include relevant chemical formulas, properties, and applications
6. Make questions progressively more challenging

COGNITIVE LEVELS TO COVER:
- Level 1: Basic facts (formula, properties, common names)
- Level 2: Conceptual understanding (structure, bonding, behavior)
- Level 3: Applied knowledge (uses, reactions, real-world applications)

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

EXAMPLE OUTPUT:
[
  {
    "question": "What is the molecular formula of benzene?",
    "answer": "The molecular formula of benzene is C6H6. This represents a six-membered ring of carbon atoms, each bonded to one hydrogen atom, forming a planar hexagonal structure.",
    "difficulty_level": "basic",
    "topic_category": "formula"
  },
  {
    "question": "Why is benzene considered aromatic?",
    "answer": "Benzene is considered aromatic because it exhibits aromaticity, a special stability due to its conjugated π-electron system. The six π-electrons are delocalized around the ring, following Hückel's rule (4n+2 π-electrons where n=1). This delocalization provides exceptional stability and unique chemical properties.",
    "difficulty_level": "intermediate",
    "topic_category": "structure"
  },
  {
    "question": "What are the main industrial uses of benzene and what safety considerations are important?",
    "answer": "Benzene is primarily used in the production of plastics, resins, synthetic fibers, rubber, dyes, detergents, and pharmaceuticals. It's a key precursor for styrene (polystyrene), phenol (phenolic resins), and cyclohexane (nylon). However, benzene is a known carcinogen and poses serious health risks. Safety considerations include proper ventilation, personal protective equipment, exposure monitoring, and strict handling protocols in industrial settings.",
    "difficulty_level": "advanced",
    "topic_category": "uses"
  }
]

IMPORTANT: Return ONLY the JSON array, no additional text or explanations."""

    def generate_image_prompt(self, compound_name: str, compound_content: str) -> str:
        """Generate a DALL-E prompt for the chemical compound using the simple template."""
        # Clean compound name for image generation
        clean_name = compound_name.replace('.', '').strip()
        
        # Extract formula from compound content
        lines = compound_content.split('\n')
        formula = ""
        
        for line in lines:
            if 'FORMULA:' in line:
                formula = line.split('FORMULA:')[1].strip()
                break
        
        # Use the skeletal formula (line-angle) diagram template with size specification
        if formula:
            image_prompt = (
                f"A skeletal formula (line-angle structural diagram) of a {clean_name} ({formula}) molecule, "
                f"minimalist scientific diagram, isolated on a plain white background. "
                f"Ensure the complete molecular structure fits fully within the 512x512 image size."
            )
        else:
            image_prompt = (
                f"A skeletal formula (line-angle structural diagram) of a {clean_name} molecule, "
                f"minimalist scientific diagram, isolated on a plain white background. "
                f"Ensure the complete molecular structure fits fully within the 512x512 image size."
            )
        
        return image_prompt

    def download_and_save_image(self, image_url: str, compound_name: str, output_dir: Path) -> str:
        """Download image from URL and save it locally."""
        try:
            # Create images subdirectory
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from compound name
            clean_name = compound_name.replace('.', '').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            filename = f"{clean_name}_molecular_diagram.png"
            file_path = images_dir / filename
            
            # Download the image
            logger.info(f"Downloading image for {compound_name}...")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Save the image
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Image saved locally: {file_path}")
            
            # Return relative path from output directory
            relative_path = f"images/{filename}"
            return relative_path
            
        except Exception as e:
            logger.error(f"Error downloading image for {compound_name}: {e}")
            return ""

    def generate_compound_image(self, compound_name: str, compound_content: str, output_dir: Path) -> str:
        """Generate an image for the compound using DALL-E 2 and save it locally."""
        try:
            logger.info(f"Generating image for: {compound_name}")
            
            image_prompt = self.generate_image_prompt(compound_name, compound_content)
            logger.info(f"Image prompt: {image_prompt}")
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                n=1
            )
            
            image_url = response.data[0].url
            logger.info(f"Successfully generated image for {compound_name}")
            
            # Download and save the image locally
            local_image_path = self.download_and_save_image(image_url, compound_name, output_dir)
            
            if local_image_path:
                logger.info(f"Image saved locally as: {local_image_path}")
                return local_image_path
            else:
                logger.error(f"Failed to save image locally for {compound_name}")
                return ""
            
        except Exception as e:
            logger.error(f"Error generating image for {compound_name}: {e}")
            return ""

    def generate_qa_pairs(self, compound_content: str, compound_name: str) -> List[Dict[str, Any]]:
        """Generate QA pairs for a given compound."""
        try:
            logger.info(f"Generating QA pairs for: {compound_name}")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Generate 3 educational QA pairs for the compound: {compound_name}. Base all questions and answers strictly on the provided Compound Information; do not use outside knowledge.\n\nCompound Information:\n{compound_content}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content.strip()
            
            # Clean up the response (remove any markdown formatting)
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Parse JSON
            qa_pairs = json.loads(content)
            
            # Validate the response
            if not isinstance(qa_pairs, list) or len(qa_pairs) != 3:
                raise ValueError(f"Expected 3 QA pairs, got {len(qa_pairs) if isinstance(qa_pairs, list) else 'invalid format'}")
            
            for i, qa in enumerate(qa_pairs):
                required_fields = ['question', 'answer', 'difficulty_level', 'topic_category']
                for field in required_fields:
                    if field not in qa:
                        raise ValueError(f"Missing required field '{field}' in QA pair {i+1}")
            
            # Add the first question about identifying the compound in the image
            image_identification_qa = self.create_image_identification_qa(compound_name, compound_content)
            
            # Insert the image identification question at the beginning
            qa_pairs.insert(0, image_identification_qa)
            
            logger.info(f"Successfully generated {len(qa_pairs)} QA pairs for {compound_name} (including image identification)")
            return qa_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {compound_name}: {e}")
            logger.error(f"Raw response: {content}")
            return []
        except Exception as e:
            logger.error(f"Error generating QA pairs for {compound_name}: {e}")
            return []

    def create_image_identification_qa(self, compound_name: str, compound_content: str) -> Dict[str, Any]:
        """Create a QA pair for identifying the compound in the image and describing its properties."""
        # Clean compound name
        clean_name = compound_name.replace('.', '').strip()
        
        # Extract key properties from compound content
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
        
        # Create the question
        question = f"Look at the molecular structure diagram in the image. What chemical compound is shown, and what are its key properties?"
        
        # Create the answer with compound information
        answer_parts = [f"The compound shown in the image is {clean_name}."]
        
        if formula:
            answer_parts.append(f"Its chemical formula is {formula}.")
        
        if compound_type:
            answer_parts.append(f"It is a {compound_type.lower()}.")
        
        if state:
            answer_parts.append(f"At room temperature, it exists as a {state.lower()}.")
        
        if molecular_weight:
            answer_parts.append(f"Its molecular weight is {molecular_weight}.")
        
        # Add some general properties based on compound type
        if 'acid' in compound_type.lower():
            answer_parts.append("As an acid, it can donate protons (H⁺ ions) in aqueous solutions.")
        elif 'alcohol' in compound_type.lower():
            answer_parts.append("As an alcohol, it contains hydroxyl (-OH) functional groups.")
        elif 'aromatic' in compound_type.lower():
            answer_parts.append("As an aromatic compound, it exhibits special stability due to delocalized π-electrons.")
        elif 'ionic' in compound_type.lower():
            answer_parts.append("As an ionic compound, it consists of positively and negatively charged ions held together by electrostatic forces.")
        
        answer = " ".join(answer_parts)
        
        return {
            "question": question,
            "answer": answer,
            "difficulty_level": "basic",
            "topic_category": "identification"
        }

    def process_compound_file(self, json_file_path: Path, output_dir: Path) -> bool:
        """Process a single compound JSON file and generate QA pairs with image."""
        try:
            # Load compound data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                compound_data = json.load(f)
            
            compound_name = compound_data.get('name', 'Unknown')
            main_entry_content = compound_data.get('main_entry_content', '')
            
            if not main_entry_content:
                logger.warning(f"No main_entry_content found for {compound_name}")
                return False
            
            # Generate QA pairs
            qa_pairs = self.generate_qa_pairs(main_entry_content, compound_name)
            
            if not qa_pairs:
                logger.warning(f"No QA pairs generated for {compound_name}")
                return False
            
            # Generate image and save locally
            local_image_path = self.generate_compound_image(compound_name, main_entry_content, output_dir)
            
            if not local_image_path:
                logger.warning(f"No image generated for {compound_name}")
                return False
            
            # Create output data structure
            output_data = {
                "compound_id": compound_data.get('compound_id'),
                "compound_name": compound_name,
                "source_file": json_file_path.name,
                "main_entry_length": compound_data.get('main_entry_length', 0),
                "qa_pairs_count": len(qa_pairs),
                "image_path": local_image_path,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "qa_pairs": qa_pairs,
                "note": "First QA pair is for image identification, followed by 3 additional educational QA pairs. Image is saved locally."
            }
            
            # Save QA pairs to file
            output_filename = f"qa_{compound_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.json"
            output_file_path = output_dir / output_filename
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved QA pairs and image for {compound_name} to {output_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {json_file_path}: {e}")
            return False

    def process_all_compounds(self, compounds_dir: Path, output_dir: Path, max_files: int = None) -> Dict[str, Any]:
        """Process all compound files and generate QA pairs."""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all JSON files
        json_files = sorted(list(compounds_dir.glob("*.json")))
        
        if max_files:
            json_files = json_files[:max_files]
        
        logger.info(f"Processing {len(json_files)} compound files...")
        
        results = {
            "total_files": len(json_files),
            "successful": 0,
            "failed": 0,
            "failed_files": [],
            "start_time": time.time()
        }
        
        for i, json_file in enumerate(json_files, 1):
            logger.info(f"Processing {i}/{len(json_files)}: {json_file.name}")
            
            success = self.process_compound_file(json_file, output_dir)
            
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["failed_files"].append(json_file.name)
            
            # Add delay to avoid rate limiting (longer for image generation)
            time.sleep(3)
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        return results

def main():
    """Main function to run QA generation."""
    # Paths
    compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
    output_dir = Path("/home/himanshu/dev/test/data/processed/qa_pairs")
    
    # Check if compounds directory exists
    if not compounds_dir.exists():
        logger.error(f"Compounds directory not found: {compounds_dir}")
        return
    
    # Initialize QA Generator
    try:
        qa_generator = QAGenerator()
        logger.info("QA Generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize QA Generator: {e}")
        return
    
    # Process compounds (start with first 5 for testing)
    logger.info("Starting QA generation process...")
    results = qa_generator.process_all_compounds(compounds_dir, output_dir, max_files=5)
    
    # Print results
    print("\n" + "="*60)
    print("QA GENERATION RESULTS")
    print("="*60)
    print(f"Total files processed: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    if results['failed_files']:
        print(f"\nFailed files:")
        for file in results['failed_files']:
            print(f"  - {file}")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
