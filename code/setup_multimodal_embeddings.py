#!/usr/bin/env python3
"""
Simplified Multimodal Embedding Setup
Works around SQLite version limitations by using alternative storage
"""

import json
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from tqdm import tqdm
import logging
import pickle
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalEmbeddingProcessor:
    """Handles both text and image embeddings with CLIP"""
    
    def __init__(self, device: str = "auto", clip_model: str = "ViT-B/32"):
        self.device = self._get_device(device)
        self.clip_model_name = clip_model
        self.clip_model = None
        self.clip_preprocess = None
        self._load_clip_model()
        
    def _get_device(self, device: str) -> str:
        """Determine best device for processing"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_clip_model(self):
        """Load CLIP model for image embeddings"""
        try:
            logger.info(f"Loading CLIP model {self.clip_model_name} on {self.device}")
            self.clip_model, self.clip_preprocess = clip.load(self.clip_model_name, device=self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_preprocess = None
    
    def get_image_embeddings(self, image_paths: List[str], batch_size: int = 16) -> Optional[np.ndarray]:
        """Get CLIP embeddings for images with batch processing"""
        if not self.clip_model or not image_paths:
            return None
            
        try:
            embeddings = []
            
            # Process images in batches
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                valid_paths = []
                
                # Load and preprocess batch
                for path in batch_paths:
                    try:
                        image = Image.open(path).convert("RGB")
                        image_tensor = self.clip_preprocess(image).unsqueeze(0)
                        batch_images.append(image_tensor)
                        valid_paths.append(path)
                    except Exception as e:
                        logger.warning(f"Failed to load image {path}: {e}")
                        continue
                
                if batch_images:
                    # Stack batch and move to device
                    batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                    
                    # Get embeddings
                    with torch.no_grad():
                        batch_embeddings = self.clip_model.encode_image(batch_tensor)
                        # Normalize to unit vectors
                        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                        embeddings.extend(batch_embeddings.cpu().numpy())
            
            if embeddings:
                # Average multiple images per chunk
                stacked = np.stack(embeddings)
                averaged = np.mean(stacked, axis=0)
                return averaged.flatten()  # 512-dim vector
            
        except Exception as e:
            logger.error(f"Error processing image embeddings: {e}")
        
        return None
    
    def get_placeholder_embedding(self) -> np.ndarray:
        """Get zero vector for image-less chunks"""
        return np.zeros(512, dtype=np.float32)

def load_enriched_data(jsonl_file: Path) -> List[Dict[str, Any]]:
    """Load enriched JSON data from JSONL file"""
    logger.info(f"Loading enriched data from {jsonl_file}")
    chunks = []
    
    with jsonl_file.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
                    continue
    
    logger.info(f"Loaded {len(chunks)} chunks from enriched data")
    return chunks

def prepare_multimodal_embeddings(processor: MultimodalEmbeddingProcessor, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare multimodal embeddings and metadata"""
    logger.info("Preparing multimodal embeddings...")
    
    embeddings_data = {
        'chunks': [],
        'text_embeddings': [],  # Will be None for now (ChromaDB handles text)
        'image_embeddings': [],
        'metadata': [],
        'ids': []
    }
    
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        # Extract text content
        text = chunk.get('text', '')
        
        # Get image paths
        image_paths = chunk.get('images', [])
        
        # Get image embedding
        if image_paths:
            img_embedding = processor.get_image_embeddings(image_paths)
            has_image_embedding = img_embedding is not None
        else:
            img_embedding = processor.get_placeholder_embedding()
            has_image_embedding = False
        
        # Create enhanced metadata
        metadata = {
            'page_range': chunk.get('page_range', ''),
            'text_length': len(text),
            'has_tables': len(chunk.get('tables', [])) > 0,
            'table_count': len(chunk.get('tables', [])),
            'has_images': len(image_paths) > 0,
            'image_count': len(image_paths),
            'enriched_via_ocr': chunk.get('enriched_via_ocr', False),
            'image_paths': image_paths,
            'has_image_embedding': has_image_embedding,
            'text_content': text,  # Store text for later use
        }
        
        # Parse extracted_info if it's a string
        extracted_info = chunk.get('extracted_info', '')
        if isinstance(extracted_info, str):
            try:
                extracted_info = json.loads(extracted_info)
            except json.JSONDecodeError:
                extracted_info = {}
        
        if isinstance(extracted_info, dict):
            metadata.update({
                'processing_method': extracted_info.get('processing_method', ''),
                'content_types': ','.join(extracted_info.get('content_types', [])),
                'has_original_text': extracted_info.get('has_original_text', False),
                'text_length_extracted': extracted_info.get('text_length', 0)
            })
        
        embeddings_data['chunks'].append(chunk)
        embeddings_data['text_embeddings'].append(None)  # Placeholder
        embeddings_data['image_embeddings'].append(img_embedding)
        embeddings_data['metadata'].append(metadata)
        embeddings_data['ids'].append(f"chunk_{i+1}")
    
    logger.info(f"Prepared {len(embeddings_data['chunks'])} multimodal embeddings")
    logger.info(f"Image embeddings: {sum(1 for emb in embeddings_data['image_embeddings'] if emb is not None)}/{len(embeddings_data['image_embeddings'])}")
    
    return embeddings_data

def save_embeddings_data(embeddings_data: Dict[str, Any], output_path: Path):
    """Save embeddings data to disk for later use"""
    logger.info(f"Saving embeddings data to {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for efficient loading
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    logger.info(f"Embeddings data saved successfully")

def create_simple_search_index(embeddings_data: Dict[str, Any]):
    """Create a simple search index for testing"""
    logger.info("Creating simple search index...")
    
    # Simple text search function
    def search_text(query: str, n_results: int = 10) -> List[Dict]:
        query_lower = query.lower()
        results = []
        
        for i, metadata in enumerate(embeddings_data['metadata']):
            text = metadata.get('text_content', '').lower()
            if query_lower in text:
                score = text.count(query_lower) / len(text) if text else 0
                results.append({
                    'id': embeddings_data['ids'][i],
                    'score': score,
                    'text': metadata.get('text_content', '')[:200] + '...',
                    'metadata': metadata
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    # Simple image similarity search
    def search_image_similarity(query_image_path: str, n_results: int = 10) -> List[Dict]:
        if not Path(query_image_path).exists():
            return []
        
        try:
            processor = MultimodalEmbeddingProcessor()
            query_embedding = processor.get_image_embeddings([query_image_path])
            
            if query_embedding is None:
                return []
            
            similarities = []
            for i, img_embedding in enumerate(embeddings_data['image_embeddings']):
                if img_embedding is not None and not np.allclose(img_embedding, 0):
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, img_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(img_embedding)
                    )
                    similarities.append({
                        'id': embeddings_data['ids'][i],
                        'score': float(similarity),
                        'text': embeddings_data['metadata'][i].get('text_content', '')[:200] + '...',
                        'metadata': embeddings_data['metadata'][i]
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['score'], reverse=True)
            return similarities[:n_results]
            
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return []
    
    return search_text, search_image_similarity

def main():
    parser = argparse.ArgumentParser(description="Simplified multimodal embedding setup")
    parser.add_argument("--enriched-file", type=Path, 
                       default=Path("/home/himanshu/dev/data/processed/chemical-compounds_chunks_enriched.jsonl"),
                       help="Path to enriched JSONL file")
    parser.add_argument("--output-path", type=Path,
                       default=Path("/home/himanshu/dev/data/embeddings/multimodal_embeddings.pkl"),
                       help="Path to save embeddings data")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32",
                       help="CLIP model to use for image embeddings")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device for processing (auto/cuda/cpu)")
    parser.add_argument("--test-queries", action="store_true",
                       help="Run test queries after setup")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of chunks to process (for testing)")
    
    args = parser.parse_args()
    
    # Initialize multimodal processor
    processor = MultimodalEmbeddingProcessor(device=args.device, clip_model=args.clip_model)
    
    # Load enriched data
    chunks = load_enriched_data(args.enriched_file)
    
    if not chunks:
        logger.error("No chunks loaded. Exiting.")
        return
    
    # Limit chunks if specified
    if args.limit:
        chunks = chunks[:args.limit]
        logger.info(f"Limited to {len(chunks)} chunks for testing")
    
    # Prepare multimodal embeddings
    embeddings_data = prepare_multimodal_embeddings(processor, chunks)
    
    # Save embeddings data
    save_embeddings_data(embeddings_data, args.output_path)
    
    # Show statistics
    total_chunks = len(embeddings_data['chunks'])
    with_image_embeddings = sum(1 for emb in embeddings_data['image_embeddings'] if emb is not None)
    ocr_enriched = sum(1 for meta in embeddings_data['metadata'] if meta.get('enriched_via_ocr', False))
    
    logger.info(f"Multimodal embedding setup complete!")
    logger.info(f"Total chunks processed: {total_chunks}")
    logger.info(f"Chunks with image embeddings: {with_image_embeddings}")
    logger.info(f"OCR-enriched chunks: {ocr_enriched}")
    logger.info(f"Embeddings data saved to: {args.output_path}")
    
    # Test queries if requested
    if args.test_queries:
        logger.info("Running test queries...")
        search_text, search_image_similarity = create_simple_search_index(embeddings_data)
        
        # Test text search
        logger.info("Testing text search...")
        results = search_text("chemical compounds", n_results=5)
        logger.info(f"Text search results: {len(results)} chunks found")
        for i, result in enumerate(results[:3]):
            logger.info(f"  {i+1}. Score: {result['score']:.3f} - {result['text']}")
        
        # Test filtered search
        logger.info("Testing OCR-enriched search...")
        ocr_results = [r for r in search_text("chemical formulas", n_results=10) 
                      if r['metadata'].get('enriched_via_ocr', False)]
        logger.info(f"OCR-enriched search results: {len(ocr_results)} chunks found")
        for i, result in enumerate(ocr_results[:3]):
            logger.info(f"  {i+1}. Score: {result['score']:.3f} - {result['text']}")
    
    # Show disk usage
    import shutil
    total, used, free = shutil.disk_usage(args.output_path.parent)
    logger.info(f"Disk usage: {used//1024//1024//1024}GB used, {free//1024//1024//1024}GB free")

if __name__ == "__main__":
    main()
