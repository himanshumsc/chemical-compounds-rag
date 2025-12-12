#!/usr/bin/env python3
"""Quick test script to verify RAG integration"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from multimodal_qa_runner_vllm import (
        QwenVLLMWrapper, 
        DEFAULT_CHROMADB_PATH,
        DEFAULT_MODEL_PATH,
        CHROMADB_AVAILABLE,
        VLLM_AVAILABLE
    )
    
    print("="*70)
    print("RAG Integration Test")
    print("="*70)
    print(f"vLLM available: {VLLM_AVAILABLE}")
    print(f"ChromaDB available: {CHROMADB_AVAILABLE}")
    print(f"ChromaDB path: {DEFAULT_CHROMADB_PATH}")
    print(f"Model path: {DEFAULT_MODEL_PATH}")
    print("="*70)
    
    if not CHROMADB_AVAILABLE:
        print("ERROR: ChromaDB not available. Cannot test RAG.")
        sys.exit(1)
    
    print("\nInitializing QwenVLLMWrapper with RAG...")
    wrapper = QwenVLLMWrapper(
        chromadb_path=str(DEFAULT_CHROMADB_PATH),
        use_rag=True,
        n_chunks=5
    )
    
    print(f"✓ Wrapper initialized")
    print(f"  - RAG enabled: {wrapper.use_rag}")
    print(f"  - Search system: {wrapper.search_system is not None}")
    print(f"  - vLLM enabled: {wrapper.use_vllm}")
    print(f"  - n_chunks: {wrapper.n_chunks}")
    
    if wrapper.search_system:
        print("\nTesting ChromaDB search...")
        # Test text search
        test_query = "What is carbon dioxide?"
        chunks = wrapper.search_system.text_search(test_query, n_results=3)
        print(f"✓ Text search returned {len(chunks)} chunks")
        if chunks:
            print(f"  Sample chunk: {chunks[0]['text'][:100]}...")
        
        print("\n✓ RAG integration test PASSED!")
    else:
        print("\n✗ Search system not initialized")
        sys.exit(1)
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

