#!/usr/bin/env python3
"""
Persistent Multimodal RAG System
User Input → Embed Query → Retrieve from persistent embeddings → Augment with Phi-4 → Generate Response
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
from transformers import AutoModelForCausalLM, AutoProcessor
import sqlite3
import hashlib
import time

# Import our existing hybrid search system
from hybrid_search import HybridSearchEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersistentMultimodalRAG:
    """Persistent multimodal RAG system with SQLite storage"""
    
    def __init__(self, embeddings_path: Path, phi4_model_path: Path, 
                 db_path: Path, device: str = "auto"):
        self.device = self._get_device(device)
        self.embeddings_path = embeddings_path
        self.phi4_model_path = phi4_model_path
        self.db_path = db_path
        
        # Initialize components
        self.search_system = None
        self.phi4_model = None
        self.phi4_processor = None
        self.db_conn = None
        
        # Load systems
        self._load_search_system()
        self._load_phi4_model()
        self._setup_database()
    
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
    
    def _load_search_system(self):
        """Load our existing hybrid search system"""
        logger.info("Loading hybrid search system...")
        self.search_system = HybridSearchEngine(self.embeddings_path, self.device)
        logger.info("Hybrid search system loaded")
    
    def _load_phi4_model(self):
        """Load Phi-4 multimodal model"""
        try:
            logger.info(f"Loading Phi-4 model from {self.phi4_model_path}")
            
            # Load config first
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(
                str(self.phi4_model_path),
                trust_remote_code=True
            )
            
            # Set attention implementation for compatibility
            try:
                cfg.attn_implementation = "sdpa"
                cfg._attn_implementation_internal = "sdpa"
                logger.info("Set attention implementation to sdpa")
            except Exception as e:
                logger.warning(f"Could not set attention implementation: {e}")
            
            # Load model
            self.phi4_model = AutoModelForCausalLM.from_pretrained(
                str(self.phi4_model_path),
                config=cfg,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load processor
            self.phi4_processor = AutoProcessor.from_pretrained(
                str(self.phi4_model_path),
                trust_remote_code=True,
                use_fast=False
            )
            
            logger.info("Phi-4 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Phi-4 model: {e}")
            self.phi4_model = None
            self.phi4_processor = None
    
    def _setup_database(self):
        """Setup SQLite database for persistent storage"""
        try:
            # Create database directory
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.db_conn = sqlite3.connect(str(self.db_path))
            cursor = self.db_conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    query_image_path TEXT,
                    query_hash TEXT UNIQUE,
                    timestamp REAL,
                    n_chunks INTEGER,
                    search_type TEXT,
                    response_text TEXT,
                    context_text TEXT,
                    image_paths TEXT,
                    retrieved_chunks TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    text_content TEXT,
                    metadata TEXT,
                    text_embedding BLOB,
                    image_embedding BLOB,
                    images TEXT,
                    page_range TEXT,
                    enriched_via_ocr BOOLEAN,
                    created_at REAL
                )
            ''')
            
            self.db_conn.commit()
            logger.info(f"Database setup complete at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            self.db_conn = None
    
    def _get_query_hash(self, query_text: str, query_image: Optional[str] = None) -> str:
        """Generate hash for query"""
        content = query_text
        if query_image:
            content += f"|{query_image}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def retrieve_relevant_chunks(self, query_text: str, query_image: Optional[str] = None,
                               n_chunks: int = 5, search_type: str = "hybrid",
                               text_weight: float = 0.7, image_weight: float = 0.3,
                               where_filter: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant chunks using our existing search system"""
        
        logger.info(f"DEBUG: retrieve_relevant_chunks called with:")
        logger.info(f"  query_text: '{query_text}'")
        logger.info(f"  query_image: {query_image}")
        logger.info(f"  n_chunks: {n_chunks}")
        logger.info(f"  search_type: {search_type}")
        logger.info(f"  text_weight: {text_weight}")
        logger.info(f"  image_weight: {image_weight}")
        logger.info(f"  where_filter: {where_filter}")
        
        if search_type == "text":
            logger.info("DEBUG: Using text search")
            chunks = self.search_system.text_search(query_text, n_chunks, where_filter)
        elif search_type == "image" and query_image:
            logger.info("DEBUG: Using image search")
            chunks = self.search_system.image_similarity_search(query_image, n_chunks, where_filter)
        elif search_type == "hybrid":
            logger.info("DEBUG: Using hybrid search")
            chunks = self.search_system.hybrid_search(
                query_text, query_image, text_weight, image_weight,
                n_chunks, where_filter
            )
        else:
            logger.info("DEBUG: Using default text search")
            chunks = self.search_system.text_search(query_text, n_chunks, where_filter)
        
        logger.info(f"DEBUG: Retrieved {len(chunks)} chunks")
        if chunks:
            logger.info(f"DEBUG: First chunk preview: {chunks[0].get('text', '')[:100]}...")
        
        return chunks
    
    def prepare_context_for_phi4(self, query: str, retrieved_chunks: List[Dict]) -> Tuple[str, List[str]]:
        """Prepare context and images for Phi-4 generation"""
        
        # Build context text
        context_parts = []
        image_paths = []
        
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk['text']
            chunk_score = chunk['score']
            
            # Add chunk to context
            context_parts.append(f"[Source {i+1} (Relevance: {chunk_score:.3f})]\n{chunk_text}")
            
            # Collect image paths from metadata
            chunk_images = chunk['metadata'].get('images', [])
            image_paths.extend(chunk_images)
        
        # Combine context
        context_text = "\n\n".join(context_parts)
        
        # Create augmented prompt
        augmented_prompt = f"""Based on the following chemical compounds database information, please answer the user's question comprehensively and accurately.

CONTEXT:
{context_text}

USER QUESTION: {query}

Please provide a detailed, accurate response based on the provided context. Include relevant chemical formulas, properties, uses, and any safety information when available. If the information is not sufficient to answer the question completely, please indicate what additional information would be helpful."""

        return augmented_prompt, image_paths[:5]  # Limit to 5 images for processing
    
    def generate_response_with_phi4(self, prompt: str, images: List[str]) -> str:
        """Generate response using Phi-4 multimodal model"""
        
        if not self.phi4_model or not self.phi4_processor:
            return "Phi-4 model not available. Please check model loading."
        
        try:
            # Prepare images for Phi-4
            processed_images = []
            for img_path in images:
                if Path(img_path).exists():
                    try:
                        image = Image.open(img_path).convert("RGB")
                        processed_images.append(image)
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path}: {e}")
                        continue
            
            # Prepare inputs for Phi-4
            if processed_images:
                # Use multimodal input
                inputs = self.phi4_processor(
                    text=prompt,
                    images=processed_images,
                    return_tensors="pt"
                )
            else:
                # Use text-only input
                inputs = self.phi4_processor(
                    text=prompt,
                    return_tensors="pt"
                )
            
            # Move inputs to device - handle None values
            device_inputs = {}
            for k, v in inputs.items():
                if v is not None:
                    device_inputs[k] = v.to(self.device)
                else:
                    logger.warning(f"Input {k} is None, skipping")
            
            # Generate response
            with torch.no_grad():
                outputs = self.phi4_model.generate(
                    **device_inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.phi4_processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.phi4_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove input prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Phi-4 generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def save_query_to_db(self, query_text: str, query_image: Optional[str], 
                        response: str, context_text: str, image_paths: List[str],
                        retrieved_chunks: List[Dict], n_chunks: int, search_type: str):
        """Save query and response to database"""
        if not self.db_conn:
            return
        
        try:
            cursor = self.db_conn.cursor()
            query_hash = self._get_query_hash(query_text, query_image)
            
            cursor.execute('''
                INSERT OR REPLACE INTO queries 
                (query_text, query_image_path, query_hash, timestamp, n_chunks, 
                 search_type, response_text, context_text, image_paths, retrieved_chunks)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_text,
                query_image,
                query_hash,
                time.time(),
                n_chunks,
                search_type,
                response,
                context_text,
                json.dumps(image_paths),
                json.dumps(retrieved_chunks)
            ))
            
            self.db_conn.commit()
            logger.info(f"Saved query to database: {query_hash}")
            
        except Exception as e:
            logger.error(f"Failed to save query to database: {e}")
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """Get recent query history"""
        if not self.db_conn:
            return []
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT query_text, query_image_path, timestamp, response_text, 
                       n_chunks, search_type
                FROM queries 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'query_text': row[0],
                    'query_image_path': row[1],
                    'timestamp': row[2],
                    'response_text': row[3],
                    'n_chunks': row[4],
                    'search_type': row[5]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get query history: {e}")
            return []
    
    def ask(self, query: str, query_image: Optional[str] = None,
            n_chunks: int = 5, search_type: str = "hybrid",
            text_weight: float = 0.7, image_weight: float = 0.3,
            where_filter: Optional[Dict] = None, save_to_db: bool = True) -> Dict[str, Any]:
        """Main RAG interface - ask a question and get a response"""
        
        logger.info(f"Processing query: '{query}'")
        
        # Check if we have this query in database
        if self.db_conn and save_to_db:
            query_hash = self._get_query_hash(query, query_image)
            cursor = self.db_conn.cursor()
            cursor.execute('SELECT response_text, context_text, image_paths, retrieved_chunks FROM queries WHERE query_hash = ?', (query_hash,))
            result = cursor.fetchone()
            
            if result:
                logger.info("Found cached response in database")
                return {
                    'query': query,
                    'response': result[0],
                    'retrieved_chunks': json.loads(result[3]) if result[3] else [],
                    'context_text': result[1],
                    'image_paths': json.loads(result[2]) if result[2] else [],
                    'search_type': search_type,
                    'timestamp': time.time(),
                    'cached': True
                }
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(
            query, query_image, n_chunks, search_type,
            text_weight, image_weight, where_filter
        )
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        if not retrieved_chunks:
            response = "I couldn't find relevant information to answer your query. Please try rephrasing your question or providing more specific terms."
            result = {
                'query': query,
                'response': response,
                'retrieved_chunks': [],
                'context_text': "",
                'image_paths': [],
                'search_type': search_type,
                'timestamp': time.time(),
                'cached': False
            }
            
            if save_to_db:
                self.save_query_to_db(query, query_image, response, "", [], [], n_chunks, search_type)
            
            return result
        
        # Step 2: Prepare context for Phi-4
        context_text, image_paths = self.prepare_context_for_phi4(query, retrieved_chunks)
        
        logger.info(f"Prepared context with {len(image_paths)} images")
        
        # Step 3: Generate response with Phi-4
        response = self.generate_response_with_phi4(context_text, image_paths)
        
        logger.info("Generated response with Phi-4")
        
        # Prepare result
        result = {
            'query': query,
            'response': response,
            'retrieved_chunks': retrieved_chunks,
            'context_text': context_text,
            'image_paths': image_paths,
            'search_type': search_type,
            'timestamp': time.time(),
            'cached': False
        }
        
        # Save to database
        if save_to_db:
            self.save_query_to_db(query, query_image, response, context_text, 
                                image_paths, retrieved_chunks, n_chunks, search_type)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = self.search_system.get_statistics()
        stats.update({
            'phi4_model_loaded': self.phi4_model is not None,
            'phi4_processor_loaded': self.phi4_processor is not None,
            'device': self.device,
            'database_connected': self.db_conn is not None
        })
        
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM queries')
                stats['total_queries'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM chunks')
                stats['total_chunks'] = cursor.fetchone()[0]
            except:
                pass
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()

def main():
    parser = argparse.ArgumentParser(description="Persistent Multimodal RAG System")
    parser.add_argument("--embeddings-path", type=Path,
                       default=Path("/home/himanshu/dev/data/embeddings/multimodal_embeddings.pkl"),
                       help="Path to embeddings data")
    parser.add_argument("--phi4-model-path", type=Path,
                       default=Path("/home/himanshu/dev/models/PHI4"),
                       help="Path to Phi-4 model")
    parser.add_argument("--db-path", type=Path,
                       default=Path("/home/himanshu/dev/data/rag_queries.db"),
                       help="Path to SQLite database")
    parser.add_argument("--query", type=str, help="Text query")
    parser.add_argument("--query-image", type=str, help="Image path for query")
    parser.add_argument("--n-chunks", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--search-type", choices=["text", "image", "hybrid"], default="hybrid",
                       help="Type of search to perform")
    parser.add_argument("--text-weight", type=float, default=0.7, help="Weight for text search")
    parser.add_argument("--image-weight", type=float, default=0.3, help="Weight for image search")
    parser.add_argument("--filter-ocr", action="store_true", help="Filter to OCR-enriched chunks")
    parser.add_argument("--filter-images", action="store_true", help="Filter to chunks with images")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--history", type=int, help="Show query history (number of recent queries)")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag_system = PersistentMultimodalRAG(
        args.embeddings_path, 
        args.phi4_model_path,
        args.db_path
    )
    
    try:
        # Show statistics if requested
        if args.stats:
            stats = rag_system.get_statistics()
            print("\n=== PERSISTENT MULTIMODAL RAG SYSTEM STATISTICS ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
            return
        
        # Show history if requested
        if args.history:
            history = rag_system.get_query_history(args.history)
            print(f"\n=== QUERY HISTORY (Last {args.history}) ===")
            for i, query in enumerate(history):
                print(f"{i+1}. Query: {query['query_text']}")
                print(f"   Response: {query['response_text'][:100]}...")
                print(f"   Time: {time.ctime(query['timestamp'])}")
                print()
            return
        
        # Build where filter
        where_filter = {}
        if args.filter_ocr:
            where_filter['enriched_via_ocr'] = True
        if args.filter_images:
            where_filter['has_images'] = True
        
        # Interactive mode
        if args.interactive:
            print("\n=== PERSISTENT MULTIMODAL RAG SYSTEM ===")
            print("Ask questions about chemical compounds with optional image context")
            print("Commands: 'ask <question>', 'image <path> <question>', 'history', 'help', 'quit'")
            
            while True:
                try:
                    command = input("\n> ").strip().split()
                    if not command:
                        continue
                    
                    if command[0] == 'quit':
                        break
                    elif command[0] == 'help':
                        print("\nAvailable commands:")
                        print("  ask <question>           - Ask a text question")
                        print("  image <path> <question> - Ask with image context")
                        print("  history                  - Show recent queries")
                        print("  help                     - Show this help")
                        print("  quit                     - Exit the program")
                    elif command[0] == 'history':
                        history = rag_system.get_query_history(5)
                        print("\n=== RECENT QUERIES ===")
                        for i, query in enumerate(history):
                            print(f"{i+1}. {query['query_text']} ({time.ctime(query['timestamp'])})")
                    elif command[0] == 'ask' and len(command) > 1:
                        query = ' '.join(command[1:])
                        result = rag_system.ask(
                            query, n_chunks=args.n_chunks,
                            search_type=args.search_type,
                            where_filter=where_filter
                        )
                        print(f"\n{result['response']}")
                        if result.get('cached'):
                            print("(Response from cache)")
                    elif command[0] == 'image' and len(command) > 2:
                        image_path = command[1]
                        query = ' '.join(command[2:])
                        result = rag_system.ask(
                            query, query_image=image_path,
                            n_chunks=args.n_chunks,
                            search_type="hybrid",
                            where_filter=where_filter
                        )
                        print(f"\n{result['response']}")
                        if result.get('cached'):
                            print("(Response from cache)")
                    else:
                        print("Invalid command. Use 'ask <question>', 'image <path> <question>', 'history', 'help', or 'quit'")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
            return
        
        # Single query mode
        if args.query:
            result = rag_system.ask(
                args.query, args.query_image,
                args.n_chunks, args.search_type,
                args.text_weight, args.image_weight,
                where_filter
            )
            print(f"\n{result['response']}")
            if result.get('cached'):
                print("(Response from cache)")
        else:
            print("Please provide --query or use --interactive mode")
    
    finally:
        rag_system.close()

if __name__ == "__main__":
    main()
