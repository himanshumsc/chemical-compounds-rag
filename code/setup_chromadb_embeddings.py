#!/usr/bin/env python3
"""
ChromaDB Multimodal Embeddings Setup
Creates ChromaDB collection with text and image embeddings for fast search
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
import pickle
from tqdm import tqdm

# Patch SQLite for ChromaDB compatibility
sys.modules['sqlite3'] = __import__('pysqlite3')

import chromadb
from chromadb.config import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBEmbeddingsSetup:
    def __init__(self, jsonl_path: str, chromadb_path: str, device: str = "cuda"):
        self.jsonl_path = Path(jsonl_path)
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
            # Create ChromaDB directory
            self.chromadb_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Setting up ChromaDB at {self.chromadb_path}")
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.chromadb_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            collection_name = "chemical_compounds_multimodal"
            try:
                self.collection = self.client.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except:
                logger.info(f"Creating new collection: {collection_name}")
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Chemical compounds with multimodal embeddings"}
                )
            
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {e}")
            raise
    
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
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get CLIP embedding for text"""
        try:
            # Truncate text to avoid CLIP context length issues
            max_length = 77  # CLIP's context length
            if len(text) > max_length:
                text = text[:max_length]
            
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy().flatten().tolist()
        
        except Exception as e:
            logger.warning(f"Failed to process text (using default): {e}")
            return [0.0] * 512  # Default embedding
    
    def process_chunks(self, limit: Optional[int] = None):
        """Process chunks and create ChromaDB embeddings"""
        logger.info(f"Processing chunks from {self.jsonl_path}")
        
        # Read JSONL file
        chunks = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
                if limit and len(chunks) >= limit:
                    break
        
        logger.info(f"Found {len(chunks)} chunks to process")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            chunk_id = f"chunk_{i}"
            text_content = chunk.get('text', '')
            
            # Skip empty chunks
            if not text_content.strip():
                continue
            
            # Get text embedding
            text_embedding = self._get_text_embedding(text_content)
            
            # Get image embedding if available
            image_path = chunk.get('image_path', '')
            image_embedding = None
            if image_path and Path(image_path).exists():
                image_embedding = self._get_image_embedding(image_path)
            
            # Use text embedding as primary (ChromaDB expects single embedding per document)
            primary_embedding = text_embedding
            
            # Prepare metadata
            metadata = {
                'chunk_id': chunk_id,
                'page_range': chunk.get('page_range', ''),
                'text_content': text_content,
                'image_path': image_path,
                'has_image': bool(image_embedding),
                'text_length': len(text_content)
            }
            
            # Add image embedding to metadata if available
            if image_embedding:
                metadata['image_embedding'] = image_embedding
            
            ids.append(chunk_id)
            documents.append(text_content)
            metadatas.append(metadata)
            embeddings.append(primary_embedding)
        
        logger.info(f"Prepared {len(ids)} embeddings for ChromaDB")
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_embeds = embeddings[i:i+batch_size]
            
            logger.info(f"Adding batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embeds
            )
        
        logger.info("ChromaDB setup complete!")
        
        # Print collection stats
        count = self.collection.count()
        logger.info(f"Collection now contains {count} documents")
    
    def reset_collection(self):
        """Reset the collection (delete all data)"""
        try:
            self.client.delete_collection("chemical_compounds_multimodal")
            logger.info("Collection deleted")
            self._setup_chromadb()
            logger.info("Collection recreated")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")

def main():
    parser = argparse.ArgumentParser(description="Setup ChromaDB with multimodal embeddings")
    parser.add_argument("--jsonl", required=True, help="Path to enriched JSONL file")
    parser.add_argument("--chromadb-path", required=True, help="Path to ChromaDB storage")
    parser.add_argument("--device", default="cuda", help="Device for CLIP model")
    parser.add_argument("--limit", type=int, help="Limit number of chunks to process")
    parser.add_argument("--reset", action="store_true", help="Reset collection before processing")
    
    args = parser.parse_args()
    
    try:
        setup = ChromaDBEmbeddingsSetup(
            jsonl_path=args.jsonl,
            chromadb_path=args.chromadb_path,
            device=args.device
        )
        
        if args.reset:
            setup.reset_collection()
        
        setup.process_chunks(limit=args.limit)
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
