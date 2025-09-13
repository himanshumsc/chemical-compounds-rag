#!/usr/bin/env python3
"""
ChromaDB Multimodal Search Engine
Fast search using ChromaDB with text and image embeddings
"""

import sys
import json
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import logging

# Patch SQLite for ChromaDB compatibility
sys.modules['sqlite3'] = __import__('pysqlite3')

import chromadb
from chromadb.config import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBSearchEngine:
    def __init__(self, chromadb_path: str, device: str = "cuda"):
        self.chromadb_path = Path(chromadb_path)
        self.device = device
        
        # Initialize CLIP model
        self.clip_model = None
        self.clip_preprocess = None
        self._load_clip_model()
        
        # Initialize ChromaDB
        self.client = None
        self.collection = None
        self._setup_chromadb()
    
    def _load_clip_model(self):
        """Load CLIP model for image embeddings"""
        try:
            logger.info("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _setup_chromadb(self):
        """Setup ChromaDB client and collection"""
        try:
            logger.info(f"Loading ChromaDB from {self.chromadb_path}")
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.chromadb_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get collection
            self.collection = self.client.get_collection("chemical_compounds_multimodal")
            logger.info("ChromaDB collection loaded successfully")
            
            # Print collection stats
            count = self.collection.count()
            logger.info(f"Collection contains {count} documents")
            
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {e}")
            raise
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get CLIP embedding for text"""
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy().flatten().tolist()
        
        except Exception as e:
            logger.error(f"Failed to process text: {e}")
            return [0.0] * 512  # Default embedding
    
    def _get_image_embedding(self, image_path: str) -> Optional[List[float]]:
        """Get CLIP embedding for an image"""
        try:
            if not Path(image_path).exists():
                return None
            
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy().flatten().tolist()
        
        except Exception as e:
            logger.warning(f"Failed to process image {image_path}: {e}")
            return None
    
    def text_search(self, query: str, n_results: int = 10, where_filter: Optional[Dict] = None) -> List[Dict]:
        """Perform text-based search using ChromaDB"""
        try:
            logger.info(f"Text search: '{query}' (n_results={n_results})")
            
            # Get query embedding
            query_embedding = self._get_text_embedding(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'search_type': 'text'
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def image_similarity_search(self, query_image_path: str, n_results: int = 10, 
                               where_filter: Optional[Dict] = None) -> List[Dict]:
        """Perform image similarity search using ChromaDB"""
        try:
            logger.info(f"Image search: '{query_image_path}' (n_results={n_results})")
            
            # Get query image embedding
            query_embedding = self._get_image_embedding(query_image_path)
            if not query_embedding:
                logger.warning("Could not process query image")
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'search_type': 'image'
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []
    
    def hybrid_search(self, query_text: str, query_image: Optional[str] = None,
                     text_weight: float = 0.7, image_weight: float = 0.3,
                     n_results: int = 10, where_filter: Optional[Dict] = None) -> List[Dict]:
        """Perform hybrid search combining text and image"""
        try:
            logger.info(f"Hybrid search: text='{query_text}', image={query_image}")
            
            # Get text results
            text_results = self.text_search(query_text, n_results * 2, where_filter)
            
            # Get image results if image provided
            image_results = []
            if query_image:
                image_results = self.image_similarity_search(query_image, n_results * 2, where_filter)
            
            # Combine and score results
            combined_results = {}
            
            # Add text results
            for result in text_results:
                chunk_id = result['id']
                combined_results[chunk_id] = {
                    'id': chunk_id,
                    'text_score': result['score'] * text_weight,
                    'image_score': 0.0,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'search_type': 'hybrid'
                }
            
            # Add image results
            for result in image_results:
                chunk_id = result['id']
                if chunk_id in combined_results:
                    combined_results[chunk_id]['image_score'] = result['score'] * image_weight
                else:
                    combined_results[chunk_id] = {
                        'id': chunk_id,
                        'text_score': 0.0,
                        'image_score': result['score'] * image_weight,
                        'text': result['text'],
                        'metadata': result['metadata'],
                        'search_type': 'hybrid'
                    }
            
            # Calculate hybrid scores
            hybrid_results = []
            for result in combined_results.values():
                hybrid_score = result['text_score'] + result['image_score']
                result['score'] = hybrid_score
                hybrid_results.append(result)
            
            # Sort by hybrid score and return top results
            hybrid_results.sort(key=lambda x: x['score'], reverse=True)
            return hybrid_results[:n_results]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description="ChromaDB Multimodal Search Engine")
    parser.add_argument("--chromadb-path", required=True, help="Path to ChromaDB storage")
    parser.add_argument("--query-text", help="Text query for search")
    parser.add_argument("--query-image", help="Image path for search")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return")
    parser.add_argument("--search-type", choices=["text", "image", "hybrid"], default="hybrid", 
                       help="Type of search to perform")
    parser.add_argument("--device", default="cuda", help="Device for CLIP model")
    
    args = parser.parse_args()
    
    try:
        search_engine = ChromaDBSearchEngine(
            chromadb_path=args.chromadb_path,
            device=args.device
        )
        
        if args.search_type == "text" and args.query_text:
            results = search_engine.text_search(args.query_text, args.n_results)
        elif args.search_type == "image" and args.query_image:
            results = search_engine.image_similarity_search(args.query_image, args.n_results)
        elif args.search_type == "hybrid" and args.query_text:
            results = search_engine.hybrid_search(args.query_text, args.query_image, 
                                                n_results=args.n_results)
        else:
            logger.error("Invalid search parameters")
            return 1
        
        # Print results
        print(f"\n=== {args.search_type.upper()} SEARCH RESULTS ===")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   ID: {result['id']}")
            print(f"   Text: {result['text'][:200]}...")
            if 'image_path' in result['metadata']:
                print(f"   Image: {result['metadata']['image_path']}")
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
