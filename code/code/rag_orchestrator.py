#!/usr/bin/env python3
"""
RAGOrchestrator: Main coordinator for the modular multimodal RAG system
Integrates ModelManager, ParallelGenerator, ResponseCombiner, and ChromaDB search
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
import hashlib
import time
from datetime import datetime

# Import our modular components
from model_manager import ModelManager
from parallel_generator import ParallelGenerator
from response_combiner import ResponseCombiner
from chromadb_search import ChromaDBSearchEngine

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """Main orchestrator for the modular multimodal RAG system"""
    
    def __init__(self, chromadb_path: Path, phi4_model_path: Path, qwen_model_path: Path,
                 db_path: Path, device: str = "auto"):
        """
        Initialize the RAG orchestrator
        
        Args:
            chromadb_path: Path to ChromaDB data
            phi4_model_path: Path to Phi-4 model
            qwen_model_path: Path to Qwen model
            db_path: Path to SQLite database for persistent storage
            device: Device to use for inference ('auto', 'cuda', 'cpu')
        """
        self.chromadb_path = chromadb_path
        self.phi4_model_path = phi4_model_path
        self.qwen_model_path = qwen_model_path
        self.db_path = db_path
        self.device = device
        
        # Initialize components
        self.model_manager = None
        self.parallel_generator = None
        self.response_combiner = None
        self.search_system = None
        self.db_conn = None
        
        # System status
        self.initialized = False
        self.models_loaded = False
        
        logger.info("RAGOrchestrator initialized")
    
    def initialize(self) -> bool:
        """Initialize all components of the RAG system"""
        try:
            logger.info("Initializing RAG system components...")
            
            # Initialize ModelManager
            self.model_manager = ModelManager(
                self.phi4_model_path, 
                self.qwen_model_path, 
                self.device
            )
            
            # Initialize ParallelGenerator
            self.parallel_generator = ParallelGenerator(self.model_manager)
            
            # Initialize ResponseCombiner
            self.response_combiner = ResponseCombiner()
            
            # Initialize ChromaDB search system
            self._load_search_system()
            
            # Setup database
            self._setup_database()
            
            self.initialized = True
            logger.info("RAG system initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def load_models(self, load_both: bool = True) -> Dict[str, bool]:
        """Load models using ModelManager"""
        if not self.initialized:
            logger.error("System not initialized. Call initialize() first.")
            return {'phi4': False, 'qwen': False, 'both_loaded': False}
        
        logger.info("Loading models...")
        status = self.model_manager.load_all_models()
        self.models_loaded = status['phi4'] or status['qwen']  # At least one model loaded
        
        logger.info(f"Model loading status: {status}")
        return status
    
    def _load_search_system(self):
        """Load ChromaDB search system"""
        logger.info("Loading ChromaDB search system...")
        # Use CPU for CLIP to avoid GPU memory conflicts with models
        if self.device == "cuda":
            search_device = "cpu"
        elif self.device == "auto":
            search_device = "cpu"  # Default to CPU for CLIP
        else:
            search_device = self.device
        
        self.search_system = ChromaDBSearchEngine(str(self.chromadb_path), search_device)
        logger.info(f"ChromaDB search system loaded on {search_device}")
    
    def _setup_database(self):
        """Setup SQLite database for persistent storage"""
        try:
            # Create database directory
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.db_conn = sqlite3.connect(str(self.db_path))
            
            # Create tables
            cursor = self.db_conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    query_image_path TEXT,
                    query_hash TEXT UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    n_chunks INTEGER,
                    search_type TEXT,
                    response_text TEXT,
                    context_text TEXT,
                    image_paths TEXT,
                    retrieved_chunks TEXT,
                    generation_time REAL,
                    models_used TEXT
                )
            ''')
            
            self.db_conn.commit()
            logger.info(f"Database setup complete at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            self.db_conn = None
    
    def process_query(self, query_text: str, query_image: Optional[str] = None,
                     n_chunks: int = 5, search_type: str = "hybrid",
                     response_strategy: str = "side_by_side") -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query_text: Text query from user
            query_image: Optional image path
            n_chunks: Number of chunks to retrieve
            search_type: Type of search ('text', 'image', 'hybrid')
            response_strategy: How to combine responses ('side_by_side', 'intelligent_merge', 'json_format', 'detailed_comparison')
        
        Returns:
            Complete response with metadata
        """
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        if not self.models_loaded:
            return {'error': 'Models not loaded'}
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: '{query_text}'")
            
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self._retrieve_relevant_chunks(
                query_text, query_image, n_chunks, search_type
            )
            
            if not retrieved_chunks:
                return {
                    'error': 'No relevant chunks found',
                    'query': query_text,
                    'processing_time': time.time() - start_time
                }
            
            # Step 2: Prepare context for generation
            context_prompt, image_paths = self._prepare_context_for_generation(
                query_text, retrieved_chunks
            )
            
            # Step 3: Generate responses from both models
            generation_result = self.parallel_generator.generate_parallel_responses(
                context_prompt, image_paths
            )
            
            # Step 4: Combine and format responses
            combined_response = self.response_combiner.combine_responses(
                generation_result, query_text, response_strategy
            )
            
            # Step 5: Save to database
            self._save_query_to_db(
                query_text, query_image, combined_response, 
                retrieved_chunks, generation_result, n_chunks, search_type
            )
            
            processing_time = time.time() - start_time
            
            # Add processing metadata
            combined_response['processing_metadata'] = {
                'total_time': processing_time,
                'chunks_retrieved': len(retrieved_chunks),
                'images_processed': len(image_paths),
                'search_type': search_type,
                'response_strategy': response_strategy
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return combined_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'error': f'Processing failed: {str(e)}',
                'query': query_text,
                'processing_time': time.time() - start_time
            }
    
    def _retrieve_relevant_chunks(self, query_text: str, query_image: Optional[str],
                                n_chunks: int, search_type: str) -> List[Dict]:
        """Retrieve relevant chunks using ChromaDB search"""
        try:
            logger.info(f"Retrieving {n_chunks} chunks using {search_type} search")
            
            if search_type == "text":
                chunks = self.search_system.text_search(query_text, n_chunks)
            elif search_type == "image" and query_image:
                chunks = self.search_system.image_similarity_search(query_image, n_chunks)
            elif search_type == "hybrid":
                if query_image:
                    chunks = self.search_system.hybrid_search(
                        query_text, query_image, n_chunks, text_weight=0.7, image_weight=0.3
                    )
                else:
                    chunks = self.search_system.text_search(query_text, n_chunks)
            else:
                chunks = self.search_system.text_search(query_text, n_chunks)
            
            logger.info(f"Retrieved {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _prepare_context_for_generation(self, query: str, retrieved_chunks: List[Dict]) -> Tuple[str, List[str]]:
        """Prepare context and images for model generation"""
        
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

        return augmented_prompt, image_paths[:5]  # Limit to 5 images
    
    def _save_query_to_db(self, query_text: str, query_image: Optional[str], 
                          response: Dict, retrieved_chunks: List[Dict], 
                          generation_result: Dict, n_chunks: int, search_type: str):
        """Save query and response to database"""
        if not self.db_conn:
            return
        
        try:
            cursor = self.db_conn.cursor()
            query_hash = self._get_query_hash(query_text, query_image)
            
            # Extract individual responses
            phi4_response = None
            qwen_response = None
            
            if generation_result.get('phi4_result'):
                phi4_response = generation_result['phi4_result'].get('response')
            
            if generation_result.get('qwen_result'):
                qwen_response = generation_result['qwen_result'].get('response')
            
            cursor.execute('''
                INSERT OR REPLACE INTO queries 
                (query_text, query_image_path, query_hash, timestamp, n_chunks, 
                 search_type, response_text, context_text, image_paths, retrieved_chunks,
                 generation_time, models_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_text,
                query_image,
                query_hash,
                datetime.now().isoformat(),
                n_chunks,
                search_type,
                response.get('formatted_text', ''),
                json.dumps(retrieved_chunks),
                json.dumps(response.get('processing_metadata', {}).get('images_processed', [])),
                json.dumps(retrieved_chunks),
                generation_result.get('total_time', 0),
                json.dumps(generation_result.get('models_used', []))
            ))
            
            self.db_conn.commit()
            logger.info(f"Query saved to database: {query_hash}")
            
        except Exception as e:
            logger.error(f"Error saving query to database: {e}")
    
    def _get_query_hash(self, query_text: str, query_image: Optional[str]) -> str:
        """Generate hash for query caching"""
        content = query_text
        if query_image:
            content += f"|{query_image}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'initialized': self.initialized,
            'models_loaded': self.models_loaded,
            'device': self.device,
            'database_connected': self.db_conn is not None
        }
        
        if self.model_manager:
            status['model_info'] = self.model_manager.get_model_info()
        
        if self.search_system:
            status['search_system'] = 'loaded'
        
        return status
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """Get recent query history from database"""
        if not self.db_conn:
            return []
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT query_text, timestamp, n_chunks, search_type, 
                       generation_time, models_used
                FROM queries 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            history = []
            
            for row in rows:
                history.append({
                    'query': row[0],
                    'timestamp': row[1],
                    'n_chunks': row[2],
                    'search_type': row[3],
                    'generation_time': row[4],
                    'models_used': json.loads(row[5]) if row[5] else []
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting query history: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the RAG system"""
        logger.info("Shutting down RAG system...")
        
        if self.parallel_generator:
            self.parallel_generator.shutdown()
        
        if self.db_conn:
            self.db_conn.close()
        
        logger.info("RAG system shutdown complete")
