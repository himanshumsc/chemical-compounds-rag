#!/usr/bin/env python3
"""
Modular Multimodal RAG System - Main Entry Point
Uses RAGOrchestrator to coordinate ModelManager, ParallelGenerator, and ResponseCombiner
"""

import argparse
import logging
from pathlib import Path
import sys
import json

# Fix ChromaDB SQLite compatibility
import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sqlite3

# Import our modular components
from rag_orchestrator import RAGOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModularMultimodalRAG:
    """Main interface for the modular multimodal RAG system"""
    
    def __init__(self, chromadb_path: str, phi4_model_path: str, qwen_model_path: str,
                 db_path: str, device: str = "auto"):
        """
        Initialize the modular RAG system
        
        Args:
            chromadb_path: Path to ChromaDB data
            phi4_model_path: Path to Phi-4 model
            qwen_model_path: Path to Qwen model
            db_path: Path to SQLite database
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.chromadb_path = Path(chromadb_path)
        self.phi4_model_path = Path(phi4_model_path)
        self.qwen_model_path = Path(qwen_model_path)
        self.db_path = Path(db_path)
        self.device = device
        
        # Initialize orchestrator
        self.orchestrator = RAGOrchestrator(
            self.chromadb_path,
            self.phi4_model_path,
            self.qwen_model_path,
            self.db_path,
            self.device
        )
        
        logger.info("ModularMultimodalRAG initialized")
    
    def initialize_system(self) -> bool:
        """Initialize the complete RAG system"""
        logger.info("Initializing modular RAG system...")
        
        # Initialize components
        if not self.orchestrator.initialize():
            logger.error("Failed to initialize RAG system")
            return False
        
        # Load models
        model_status = self.orchestrator.load_models()
        
        if not model_status['both_loaded']:
            logger.warning("Not all models loaded successfully")
            logger.info(f"Model status: {model_status}")
        
        logger.info("Modular RAG system initialization completed")
        return True
    
    def ask_question(self, question: str, image_path: str = None, 
                     n_chunks: int = 5, search_type: str = "hybrid",
                     response_strategy: str = "side_by_side") -> str:
        """
        Ask a question to the RAG system
        
        Args:
            question: Text question
            image_path: Optional image path
            n_chunks: Number of chunks to retrieve
            search_type: Search type ('text', 'image', 'hybrid')
            response_strategy: Response combination strategy
        
        Returns:
            Formatted response text
        """
        try:
            result = self.orchestrator.process_query(
                question, image_path, n_chunks, search_type, response_strategy
            )
            
            if 'error' in result:
                return f"Error: {result['error']}"
            
            return result.get('formatted_text', 'No response generated')
            
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return f"Error: {str(e)}"
    
    def get_system_status(self) -> dict:
        """Get system status information"""
        return self.orchestrator.get_system_status()
    
    def get_query_history(self, limit: int = 10) -> list:
        """Get recent query history"""
        return self.orchestrator.get_query_history(limit)
    
    def shutdown(self):
        """Shutdown the system"""
        self.orchestrator.shutdown()

def interactive_mode(rag_system: ModularMultimodalRAG):
    """Run interactive mode for the RAG system"""
    
    # Exit early if no interactive TTY is available
    import sys
    if not sys.stdin.isatty():
        logger.warning("No TTY detected; skipping interactive loop and running sample query.")
        sample_question = "Briefly define methane."
        print("Non-interactive environment detected. Running sample query instead:\n")
        response = rag_system.ask_question(sample_question)
        print(response)
        return

    print("\n" + "="*80)
    print("🧠 MODULAR MULTIMODAL RAG SYSTEM")
    print("="*80)
    print("Ask questions about chemical compounds with optional image context")
    print("Commands:")
    print("  ask <question>                    - Ask a text question")
    print("  image <path> <question>          - Ask with image context")
    print("  strategy <strategy>              - Change response strategy")
    print("  chunks <number>                  - Change number of chunks")
    print("  search <type>                    - Change search type")
    print("  status                           - Show system status")
    print("  history                          - Show query history")
    print("  help                             - Show this help")
    print("  quit                             - Exit the system")
    print("="*80)
    
    # Default settings
    current_strategy = "side_by_side"
    current_chunks = 5
    current_search_type = "hybrid"
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  ask <question>                    - Ask a text question")
                print("  image <path> <question>          - Ask with image context")
                print("  strategy <strategy>              - Change response strategy")
                print("  chunks <number>                  - Change number of chunks")
                print("  search <type>                    - Change search type")
                print("  status                           - Show system status")
                print("  history                          - Show query history")
                print("  help                             - Show this help")
                print("  quit                             - Exit the system")
                print(f"\nCurrent settings:")
                print(f"  Strategy: {current_strategy}")
                print(f"  Chunks: {current_chunks}")
                print(f"  Search Type: {current_search_type}")
            
            elif user_input.lower() == 'status':
                status = rag_system.get_system_status()
                print(f"\n📊 SYSTEM STATUS:")
                print(f"  Initialized: {status.get('initialized', False)}")
                print(f"  Models Loaded: {status.get('models_loaded', False)}")
                print(f"  Device: {status.get('device', 'Unknown')}")
                print(f"  Database Connected: {status.get('database_connected', False)}")
                
                if 'model_info' in status:
                    model_info = status['model_info']
                    print(f"  Phi-4 Loaded: {model_info.get('phi4_loaded', False)}")
                    print(f"  Qwen Loaded: {model_info.get('qwen_loaded', False)}")
                    print(f"  Current GPU Memory: {model_info.get('current_gpu_memory', 0):.2f} GB")
            
            elif user_input.lower() == 'history':
                history = rag_system.get_query_history(5)
                print(f"\n📜 RECENT QUERIES:")
                for i, query in enumerate(history, 1):
                    print(f"  {i}. {query['query'][:50]}...")
                    print(f"     Time: {query['timestamp']}")
                    print(f"     Models: {', '.join(query['models_used'])}")
            
            elif user_input.startswith('strategy '):
                strategy = user_input[9:].strip()
                valid_strategies = ['side_by_side', 'intelligent_merge', 'json_format', 'detailed_comparison']
                if strategy in valid_strategies:
                    current_strategy = strategy
                    print(f"✅ Response strategy changed to: {strategy}")
                else:
                    print(f"❌ Invalid strategy. Valid options: {', '.join(valid_strategies)}")
            
            elif user_input.startswith('chunks '):
                try:
                    chunks = int(user_input[7:].strip())
                    if 1 <= chunks <= 20:
                        current_chunks = chunks
                        print(f"✅ Number of chunks changed to: {chunks}")
                    else:
                        print("❌ Number of chunks must be between 1 and 20")
                except ValueError:
                    print("❌ Invalid number for chunks")
            
            elif user_input.startswith('search '):
                search_type = user_input[7:].strip()
                valid_types = ['text', 'image', 'hybrid']
                if search_type in valid_types:
                    current_search_type = search_type
                    print(f"✅ Search type changed to: {search_type}")
                else:
                    print(f"❌ Invalid search type. Valid options: {', '.join(valid_types)}")
            
            elif user_input.startswith('ask '):
                question = user_input[4:].strip()
                if question:
                    print(f"\n🤔 Processing question: {question}")
                    response = rag_system.ask_question(
                        question, n_chunks=current_chunks, 
                        search_type=current_search_type, 
                        response_strategy=current_strategy
                    )
                    print(f"\n{response}")
                else:
                    print("❌ Please provide a question")
            
            elif user_input.startswith('image '):
                parts = user_input[6:].strip().split(' ', 1)
                if len(parts) == 2:
                    image_path, question = parts
                    if Path(image_path).exists():
                        print(f"\n🖼️ Processing question with image: {question}")
                        print(f"📁 Image: {image_path}")
                        response = rag_system.ask_question(
                            question, image_path, n_chunks=current_chunks,
                            search_type=current_search_type,
                            response_strategy=current_strategy
                        )
                        print(f"\n{response}")
                    else:
                        print(f"❌ Image file not found: {image_path}")
                else:
                    print("❌ Usage: image <path> <question>")
            
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            # Gracefully exit when no input stream is available
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"❌ Error: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Modular Multimodal RAG System")
    parser.add_argument("--query", "-q", help="Text query to process")
    parser.add_argument("--image", "-i", help="Image path for multimodal query")
    parser.add_argument("--n-chunks", "-n", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--search-type", "-s", default="hybrid", 
                       choices=["text", "image", "hybrid"], help="Search type")
    parser.add_argument("--strategy", default="side_by_side",
                       choices=["side_by_side", "intelligent_merge", "json_format", "detailed_comparison"],
                       help="Response combination strategy")
    parser.add_argument("--chromadb-path", default="/home/himanshu/dev/data/chromadb",
                       help="Path to ChromaDB data")
    parser.add_argument("--phi4-path", default="/home/himanshu/dev/models/PHI4_ONNX",
                       help="Path to Phi-4 ONNX model")
    parser.add_argument("--qwen-path", default="/home/himanshu/dev/models/QWEN_AWQ",
                       help="Path to Qwen model")
    parser.add_argument("--db-path", default="/home/himanshu/dev/data/rag_queries.db",
                       help="Path to SQLite database")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag_system = ModularMultimodalRAG(
        args.chromadb_path,
        args.phi4_path,
        args.qwen_path,
        args.db_path,
        args.device
    )
    
    # Initialize system
    if not rag_system.initialize_system():
        logger.error("Failed to initialize RAG system")
        sys.exit(1)
    
    try:
        if args.interactive:
            # Run interactive mode
            interactive_mode(rag_system)
        elif args.query:
            # Process single query
            response = rag_system.ask_question(
                args.query, args.image, args.n_chunks, 
                args.search_type, args.strategy
            )
            print(response)
        else:
            # Default to interactive mode
            interactive_mode(rag_system)
    
    finally:
        # Shutdown system
        rag_system.shutdown()

if __name__ == "__main__":
    main()
