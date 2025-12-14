#!/usr/bin/env python3
"""
Download Phi-4 Multimodal Model
Downloads the specific version of Phi-4 model used in this project
"""

import os
import sys
from pathlib import Path
from transformers import AutoModel, AutoProcessor
import torch

def download_phi4_model():
    """Download Phi-4 multimodal model to local directory"""
    
    # Model configuration
    model_name = "microsoft/phi-4"
    local_model_path = Path("/home/himanshu/dev/models/PHI4")
    
    print(f"Downloading Phi-4 model: {model_name}")
    print(f"Local path: {local_model_path}")
    
    # Create directory if it doesn't exist
    local_model_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model and processor
        print("Downloading model...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Save to local directory
        print(f"Saving model to {local_model_path}...")
        model.save_pretrained(local_model_path)
        processor.save_pretrained(local_model_path)
        
        print("✅ Model download completed successfully!")
        print(f"Model saved to: {local_model_path}")
        
        # Print model info
        print(f"\nModel Information:")
        print(f"- Model: {model_name}")
        print(f"- Local Path: {local_model_path}")
        print(f"- Model Type: {type(model).__name__}")
        print(f"- Processor Type: {type(processor).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def verify_model():
    """Verify that the model can be loaded"""
    local_model_path = Path("/home/himanshu/dev/models/PHI4")
    
    if not local_model_path.exists():
        print("❌ Model directory does not exist")
        return False
    
    try:
        print("Verifying model...")
        model = AutoModel.from_pretrained(
            str(local_model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            str(local_model_path),
            trust_remote_code=True
        )
        
        print("✅ Model verification successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def main():
    """Main function"""
    print("🧪 Phi-4 Model Download Script")
    print("=" * 50)
    
    # Check if model already exists
    local_model_path = Path("/home/himanshu/dev/models/PHI4")
    if local_model_path.exists() and any(local_model_path.iterdir()):
        print(f"Model already exists at {local_model_path}")
        choice = input("Do you want to re-download? (y/N): ").strip().lower()
        if choice != 'y':
            print("Skipping download. Verifying existing model...")
            verify_model()
            return
    
    # Download model
    success = download_phi4_model()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Run PDF preprocessing: python pdf_preprocess.py")
        print("2. Run OCR enrichment: python ocr_enrich_phi4_multithreaded.py")
        print("3. Create embeddings: python setup_multimodal_embeddings.py")
        print("4. Use RAG system: python rag_system.py --interactive")
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
