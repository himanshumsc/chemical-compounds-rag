#!/usr/bin/env python3
"""
Hybrid Search Interface for Multimodal Embeddings
Provides text search, image similarity search, and weighted hybrid search
"""

import json
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import pickle
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """Hybrid search engine combining text and image embeddings"""
    
    def __init__(self, embeddings_path: Path, device: str = "auto"):
        self.device = self._get_device(device)
        self.embeddings_data = self._load_embeddings(embeddings_path)
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
    
    def _load_embeddings(self, embeddings_path: Path) -> Dict[str, Any]:
        """Load embeddings data from pickle file"""
        logger.info(f"Loading embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data['chunks'])} chunks with embeddings")
        return data
    
    def _load_clip_model(self):
        """Load CLIP model for image similarity search"""
        try:
            logger.info(f"Loading CLIP model on {self.device}")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_preprocess = None
    
    def text_search(self, query: str, n_results: int = 10, 
                   where_filter: Optional[Dict] = None) -> List[Dict]:
        """Perform text-based search with optional filtering"""
        # Extract key terms from query (remove common question words)
        query_lower = query.lower()
        question_words = {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_terms = [word for word in query_lower.split() if word not in question_words and len(word) > 2]
        
        # If no key terms found, use original query
        if not key_terms:
            key_terms = [query_lower]
        
        logger.info(f"DEBUG: text_search called with query='{query}', key_terms={key_terms}, n_results={n_results}")
        logger.info(f"DEBUG: Searching through {len(self.embeddings_data['metadata'])} chunks")
        
        results = []
        
        for i, metadata in enumerate(self.embeddings_data['metadata']):
            # Apply where filter if provided
            if where_filter:
                if not self._matches_filter(metadata, where_filter):
                    continue
            
            text = metadata.get('text_content', '').lower()
            
            # Check if any key term matches
            match_found = False
            best_score = 0
            
            for term in key_terms:
                if term in text:
                    match_found = True
                    # Calculate relevance score based on frequency and position
                    score = self._calculate_text_score(term, text)
                    best_score = max(best_score, score)
            
            if match_found:
                results.append({
                    'id': self.embeddings_data['ids'][i],
                    'score': best_score,
                    'text': metadata.get('text_content', ''),
                    'metadata': metadata,
                    'search_type': 'text'
                })
                logger.info(f"DEBUG: Found match in chunk {i}, score={best_score:.3f}")
                logger.info(f"DEBUG: Text preview: {metadata.get('text_content', '')[:100]}...")
        
        logger.info(f"DEBUG: Found {len(results)} total matches")
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    def image_similarity_search(self, query_image_path: str, n_results: int = 10,
                               where_filter: Optional[Dict] = None) -> List[Dict]:
        """Perform image similarity search using CLIP embeddings"""
        if not self.clip_model or not Path(query_image_path).exists():
            logger.warning("CLIP model not available or image path invalid")
            return []
        
        try:
            # Get query image embedding
            image = Image.open(query_image_path).convert("RGB")
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                query_embedding = self.clip_model.encode_image(image_tensor)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
                query_embedding = query_embedding.cpu().numpy().flatten()
            
            similarities = []
            for i, img_embedding in enumerate(self.embeddings_data['image_embeddings']):
                # Apply where filter if provided
                if where_filter:
                    if not self._matches_filter(self.embeddings_data['metadata'][i], where_filter):
                        continue
                
                if img_embedding is not None and not np.allclose(img_embedding, 0):
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, img_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(img_embedding)
                    )
                    similarities.append({
                        'id': self.embeddings_data['ids'][i],
                        'score': float(similarity),
                        'text': self.embeddings_data['metadata'][i].get('text_content', '')[:200] + '...',
                        'metadata': self.embeddings_data['metadata'][i],
                        'search_type': 'image'
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['score'], reverse=True)
            return similarities[:n_results]
            
        except Exception as e:
            logger.error(f"Image similarity search error: {e}")
            return []
    
    def hybrid_search(self, query_text: str, query_image_path: Optional[str] = None,
                     text_weight: float = 0.7, image_weight: float = 0.3,
                     n_results: int = 10, where_filter: Optional[Dict] = None) -> List[Dict]:
        """Perform hybrid search combining text and image similarity"""
        
        # Get text search results
        text_results = self.text_search(query_text, n_results * 2, where_filter)
        
        # Get image search results if image provided
        image_results = []
        if query_image_path:
            image_results = self.image_similarity_search(query_image_path, n_results * 2, where_filter)
        
        # Combine results with weighted scoring
        if image_results:
            # Create score mapping
            text_scores = {result['id']: result['score'] for result in text_results}
            image_scores = {result['id']: result['score'] for result in image_results}
            
            # Get all unique chunk IDs
            all_ids = set(text_scores.keys()) | set(image_scores.keys())
            
            # Calculate hybrid scores
            hybrid_results = []
            for chunk_id in all_ids:
                text_score = text_scores.get(chunk_id, 0)
                image_score = image_scores.get(chunk_id, 0)
                hybrid_score = text_weight * text_score + image_weight * image_score
                
                # Get metadata
                chunk_idx = self.embeddings_data['ids'].index(chunk_id)
                metadata = self.embeddings_data['metadata'][chunk_idx]
                
                hybrid_results.append({
                    'id': chunk_id,
                    'score': hybrid_score,
                    'text_score': text_score,
                    'image_score': image_score,
                    'text': metadata.get('text_content', ''),
                    'metadata': metadata,
                    'search_type': 'hybrid'
                })
            
            # Sort by hybrid score
            hybrid_results.sort(key=lambda x: x['score'], reverse=True)
            return hybrid_results[:n_results]
        else:
            # Return text results only
            return text_results[:n_results]
    
    def _calculate_text_score(self, query: str, text: str) -> float:
        """Calculate text relevance score"""
        if not text:
            return 0.0
        
        # Count occurrences
        count = text.count(query)
        if count == 0:
            return 0.0
        
        # Normalize by text length and add position bonus
        base_score = count / len(text)
        
        # Add bonus for early occurrence
        position = text.find(query)
        position_bonus = max(0, (len(text) - position) / len(text)) * 0.1
        
        return base_score + position_bonus
    
    def _matches_filter(self, metadata: Dict, where_filter: Dict) -> bool:
        """Check if metadata matches where filter"""
        for key, value in where_filter.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # Handle operators like {"$contains": "value"}
                if "$contains" in value:
                    if value["$contains"] not in str(metadata[key]):
                        return False
                elif "$eq" in value:
                    if metadata[key] != value["$eq"]:
                        return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get full chunk data by ID"""
        try:
            idx = self.embeddings_data['ids'].index(chunk_id)
            return {
                'id': chunk_id,
                'chunk': self.embeddings_data['chunks'][idx],
                'metadata': self.embeddings_data['metadata'][idx],
                'image_embedding': self.embeddings_data['image_embeddings'][idx]
            }
        except ValueError:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        total_chunks = len(self.embeddings_data['chunks'])
        with_image_embeddings = sum(1 for emb in self.embeddings_data['image_embeddings'] 
                                  if emb is not None and not np.allclose(emb, 0))
        ocr_enriched = sum(1 for meta in self.embeddings_data['metadata'] 
                          if meta.get('enriched_via_ocr', False))
        
        return {
            'total_chunks': total_chunks,
            'chunks_with_image_embeddings': with_image_embeddings,
            'ocr_enriched_chunks': ocr_enriched,
            'clip_model_loaded': self.clip_model is not None
        }

def main():
    parser = argparse.ArgumentParser(description="Hybrid search interface for multimodal embeddings")
    parser.add_argument("--embeddings-path", type=Path,
                       default=Path("/home/himanshu/dev/data/embeddings/multimodal_embeddings.pkl"),
                       help="Path to embeddings data")
    parser.add_argument("--query-text", type=str, help="Text query for search")
    parser.add_argument("--query-image", type=str, help="Image path for similarity search")
    parser.add_argument("--n-results", type=int, default=10, help="Number of results to return")
    parser.add_argument("--text-weight", type=float, default=0.7, help="Weight for text search in hybrid")
    parser.add_argument("--image-weight", type=float, default=0.3, help="Weight for image search in hybrid")
    parser.add_argument("--filter-ocr", action="store_true", help="Filter to OCR-enriched chunks only")
    parser.add_argument("--filter-images", action="store_true", help="Filter to chunks with images only")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    
    args = parser.parse_args()
    
    # Initialize search engine
    search_engine = HybridSearchEngine(args.embeddings_path)
    
    # Show statistics if requested
    if args.stats:
        stats = search_engine.get_statistics()
        print("\n=== DATABASE STATISTICS ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        return
    
    # Build where filter
    where_filter = {}
    if args.filter_ocr:
        where_filter['enriched_via_ocr'] = True
    if args.filter_images:
        where_filter['has_images'] = True
    
    # Interactive mode
    if args.interactive:
        print("\n=== HYBRID SEARCH INTERFACE ===")
        print("Commands: 'text <query>', 'image <path>', 'hybrid <query> [image_path]', 'stats', 'quit'")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                if not command:
                    continue
                
                if command[0] == 'quit':
                    break
                elif command[0] == 'stats':
                    stats = search_engine.get_statistics()
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                elif command[0] == 'text' and len(command) > 1:
                    query = ' '.join(command[1:])
                    results = search_engine.text_search(query, args.n_results, where_filter)
                    print(f"\nText search results ({len(results)}):")
                    for i, result in enumerate(results):
                        print(f"{i+1}. Score: {result['score']:.3f} - {result['text']}")
                elif command[0] == 'image' and len(command) > 1:
                    image_path = command[1]
                    results = search_engine.image_similarity_search(image_path, args.n_results, where_filter)
                    print(f"\nImage similarity results ({len(results)}):")
                    for i, result in enumerate(results):
                        print(f"{i+1}. Score: {result['score']:.3f} - {result['text']}")
                elif command[0] == 'hybrid' and len(command) > 1:
                    query = command[1]
                    image_path = command[2] if len(command) > 2 else None
                    results = search_engine.hybrid_search(query, image_path, 
                                                        args.text_weight, args.image_weight,
                                                        args.n_results, where_filter)
                    print(f"\nHybrid search results ({len(results)}):")
                    for i, result in enumerate(results):
                        print(f"{i+1}. Score: {result['score']:.3f} (T:{result.get('text_score', 0):.3f}, I:{result.get('image_score', 0):.3f}) - {result['text']}")
                else:
                    print("Invalid command. Use 'text <query>', 'image <path>', 'hybrid <query> [image_path]', 'stats', or 'quit'")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        return
    
    # Single query mode
    if args.query_text:
        if args.query_image:
            # Hybrid search
            results = search_engine.hybrid_search(
                args.query_text, args.query_image,
                args.text_weight, args.image_weight,
                args.n_results, where_filter
            )
            print(f"\nHybrid search results ({len(results)}):")
            for i, result in enumerate(results):
                print(f"{i+1}. Score: {result['score']:.3f} (T:{result.get('text_score', 0):.3f}, I:{result.get('image_score', 0):.3f}) - {result['text']}")
        else:
            # Text search
            results = search_engine.text_search(args.query_text, args.n_results, where_filter)
            print(f"\nText search results ({len(results)}):")
            for i, result in enumerate(results):
                print(f"{i+1}. Score: {result['score']:.3f} - {result['text']}")
    elif args.query_image:
        # Image search
        results = search_engine.image_similarity_search(args.query_image, args.n_results, where_filter)
        print(f"\nImage similarity results ({len(results)}):")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.3f} - {result['text']}")
    else:
        print("Please provide --query-text and/or --query-image, or use --interactive mode")

if __name__ == "__main__":
    main()
