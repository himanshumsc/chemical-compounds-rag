#!/usr/bin/env python3
"""
Multi-Modal QA Dataset Generator for Chemical Compounds
Generates QA pairs and ball-and-stick images using OpenAI APIs
"""

import openai
import os
import json
import requests
import logging
from pathlib import Path
from typing import Dict, List, Any
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChemicalQAGenerator:
    """Generate QA pairs and images for chemical compounds using OpenAI APIs"""
    
    def __init__(self):
        """Initialize the generator with API key"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Setup directories
        self.output_dir = Path("/home/himanshu/dev/test/QA_set")
        self.images_dir = self.output_dir / "generated_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized QA Generator")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Images directory: {self.images_dir}")
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize compound name for filename"""
        # Replace spaces and special characters with underscores
        sanitized = name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("'", "")
        # Remove multiple underscores
        sanitized = sanitized.replace("__", "_")
        return sanitized
    
    def generate_image(self, compound_name: str) -> str:
        """Generate ball-and-stick image using DALL-E 2"""
        logger.info(f"Generating image for {compound_name}")
        
        # Create prompt for DALL-E 2
        prompt = f"A simple 2D ball-and-stick model of a {compound_name} molecule, minimalist style, on a plain white background."
        
        try:
            # Call DALL-E 2 API
            response = self.client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="256x256",
                n=1,
                quality="standard"
            )
            
            # Get image URL
            image_url = response.data[0].url
            logger.info(f"Image generated successfully for {compound_name}")
            
            return image_url
            
        except Exception as e:
            logger.error(f"Failed to generate image for {compound_name}: {e}")
            raise
    
    def download_image(self, image_url: str, compound_name: str) -> str:
        """Download and save image from URL"""
        logger.info(f"Downloading image for {compound_name}")
        
        try:
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Create filename
            sanitized_name = self.sanitize_filename(compound_name)
            filename = f"{sanitized_name}_structure.png"
            filepath = self.images_dir / filename
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Image saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download image for {compound_name}: {e}")
            raise
    
    def generate_qa_pairs(self, compound_text: str) -> List[Dict[str, str]]:
        """Generate QA pairs using GPT-4o"""
        logger.info("Generating QA pairs using GPT-4o")
        
        # System prompt for QA generation
        system_prompt = """You are an expert in chemistry and educational material design. Your task is to generate 3 high-quality, multi-modal question-answer pairs based ONLY on the provided text.

**CRITICAL INSTRUCTIONS:**
1. **Assume an Image:** You must assume the user is viewing a simple ball-and-stick diagram of the molecule alongside the text.
2. **Multi-Modal Questions:** Each question MUST be phrased to require information from BOTH the assumed diagram AND the provided text.
3. **Answer from Text:** The answer must be derived SOLELY from the provided text.
4. **Vary Question Types:** Generate questions that link visual features of the assumed diagram to factual data in the text. Examples:
   * Ask to identify a feature in the diagram and explain its properties using the text.
   * Ask to relate a structural concept from the diagram to a fact in the text.
   * Ask to count something in the diagram and combine it with a fact from the text.
5. **Output Format:** Provide the output as a valid JSON array of objects, where each object has a "question" and "answer" key."""
        
        try:
            # Call GPT-4o API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": compound_text}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            # Parse JSON response
            qa_text = response.choices[0].message.content
            logger.info(f"QA pairs generated successfully")
            
            # Parse JSON
            qa_pairs = json.loads(qa_text)
            
            # Validate format
            if not isinstance(qa_pairs, list):
                raise ValueError("Response is not a list")
            
            for qa in qa_pairs:
                if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                    raise ValueError("Invalid QA pair format")
            
            logger.info(f"Generated {len(qa_pairs)} QA pairs")
            return qa_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {qa_text}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate QA pairs: {e}")
            raise
    
    def process_compound(self, compound_data: Dict[str, str]) -> Dict[str, Any]:
        """Process a single compound: generate image and QA pairs"""
        compound_name = compound_data["compound_name"]
        compound_text = compound_data["text"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing compound: {compound_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Generate image
            image_url = self.generate_image(compound_name)
            
            # Step 2: Download and save image
            image_path = self.download_image(image_url, compound_name)
            
            # Step 3: Generate QA pairs
            qa_pairs = self.generate_qa_pairs(compound_text)
            
            # Create result
            result = {
                "compound_name": compound_name,
                "image_path": image_path,
                "qa_pairs": qa_pairs,
                "metadata": {
                    "image_url": image_url,
                    "qa_count": len(qa_pairs),
                    "text_length": len(compound_text)
                }
            }
            
            logger.info(f"✅ Successfully processed {compound_name}")
            logger.info(f"   Image: {image_path}")
            logger.info(f"   QA pairs: {len(qa_pairs)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process {compound_name}: {e}")
            raise
    
    def create_test_data(self) -> List[Dict[str, str]]:
        """Create test data for Benzene, Methane, and Niacin"""
        logger.info("Creating test data for Benzene, Methane, and Niacin")
        
        # Load compound data from our existing JSONs
        compounds_dir = Path("/home/himanshu/dev/test/data/processed/individual_compounds")
        
        test_compounds = []
        
        # Load Benzene
        benzene_file = compounds_dir / "compound_018_Benzene.json"
        if benzene_file.exists():
            with open(benzene_file, 'r', encoding='utf-8') as f:
                benzene_data = json.load(f)
            test_compounds.append({
                "compound_name": "Benzene",
                "text": benzene_data["main_entry_content"]
            })
            logger.info("✅ Loaded Benzene data")
        
        # Load Methane
        methane_file = compounds_dir / "compound_088_Methane.json"
        if methane_file.exists():
            with open(methane_file, 'r', encoding='utf-8') as f:
                methane_data = json.load(f)
            test_compounds.append({
                "compound_name": "Methane",
                "text": methane_data["main_entry_content"]
            })
            logger.info("✅ Loaded Methane data")
        
        # Load Niacin
        niacin_file = compounds_dir / "compound_096_Niacin.json"
        if niacin_file.exists():
            with open(niacin_file, 'r', encoding='utf-8') as f:
                niacin_data = json.load(f)
            test_compounds.append({
                "compound_name": "Niacin",
                "text": niacin_data["main_entry_content"]
            })
            logger.info("✅ Loaded Niacin data")
        
        logger.info(f"Created test data for {len(test_compounds)} compounds")
        return test_compounds
    
    def generate_test_dataset(self):
        """Generate test dataset for 3 compounds"""
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING TEST QA DATASET")
        logger.info(f"{'='*80}")
        
        # Create test data
        test_compounds = self.create_test_data()
        
        if not test_compounds:
            logger.error("No test compounds found!")
            return
        
        # Process each compound
        results = []
        
        for i, compound_data in enumerate(test_compounds):
            logger.info(f"\n--- Processing compound {i+1}/{len(test_compounds)} ---")
            
            try:
                result = self.process_compound(compound_data)
                results.append(result)
                
                # Add delay between API calls to avoid rate limiting
                if i < len(test_compounds) - 1:
                    logger.info("Waiting 2 seconds before next compound...")
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Failed to process {compound_data['compound_name']}: {e}")
                continue
        
        # Save results
        output_file = self.output_dir / "test_qa_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*80}")
        logger.info("TEST GENERATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Processed compounds: {len(results)}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Images directory: {self.images_dir}")
        
        # Show summary
        for result in results:
            logger.info(f"  - {result['compound_name']}: {result['metadata']['qa_count']} QA pairs")
        
        return results

def main():
    """Main function"""
    try:
        generator = ChemicalQAGenerator()
        results = generator.generate_test_dataset()
        
        if results:
            logger.info(f"\n🎉 SUCCESS! Test QA dataset generated!")
            logger.info(f"📁 Results saved to: /home/himanshu/dev/test/QA_set/")
        else:
            logger.info(f"\n❌ No results generated!")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
