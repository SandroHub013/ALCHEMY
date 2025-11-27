#!/usr/bin/env python3
"""
Smart Chunking + Reranking usage example.

Demonstrates how to use the new features inspired by osgrep:
1. Smart Chunking with tree-sitter for Python code
2. Reranking with CrossEncoder for more accurate results
3. Integration with existing VectorStore

Prerequisites:
    pip install tree-sitter tree-sitter-python

Usage:
    python examples/smart_chunking_example.py
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_smart_chunking():
    """Demonstrate intelligent Python code chunking."""
    from src.memory import SmartChunker, ChunkType
    
    print("\n" + "=" * 60)
    print("🔧 DEMO: Smart Chunking with Tree-sitter")
    print("=" * 60)
    
    # Sample Python code to chunk
    sample_code = '''
"""Example module for chunking demonstration."""

import os
import sys
from typing import List, Optional

# Constants
MAX_ITEMS = 100
DEFAULT_TIMEOUT = 30


class DataProcessor:
    """
    Class for processing data.
    
    Handles data transformation and validation
    before saving.
    """
    
    def __init__(self, config: dict):
        """Initialize the processor with configuration."""
        self.config = config
        self.cache = {}
    
    def process(self, data: List[dict]) -> List[dict]:
        """
        Process a list of records.
        
        Args:
            data: List of dictionaries to process
            
        Returns:
            List of processed records
        """
        results = []
        for item in data:
            processed = self._transform(item)
            if self._validate(processed):
                results.append(processed)
        return results
    
    def _transform(self, item: dict) -> dict:
        """Transform a single record."""
        return {k.lower(): v for k, v in item.items()}
    
    def _validate(self, item: dict) -> bool:
        """Validate a record."""
        return bool(item)


def load_data(path: str) -> List[dict]:
    """
    Load data from file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        List of loaded records
    """
    import json
    with open(path) as f:
        return json.load(f)


def main():
    """Main entry point."""
    processor = DataProcessor({"verbose": True})
    data = [{"Name": "Alice"}, {"Name": "Bob"}]
    result = processor.process(data)
    print(f"Processed {len(result)} records")


if __name__ == "__main__":
    main()
'''
    
    # Create chunker
    chunker = SmartChunker(
        max_chunk_size=1500,  # Larger chunks for demo
        min_chunk_size=50,
        include_imports=True,
    )
    
    # Chunk the code
    chunks = chunker.chunk_python_code(sample_code, file_path="example.py")
    
    print(f"\n📦 Extracted {len(chunks)} chunks:\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk.chunk_type.value.upper()}: {chunk.qualified_name}")
        print(f"      Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"      Size: {chunk.char_count} chars")
        if chunk.docstring:
            print(f"      Doc: {chunk.docstring[:60]}...")
        print()
    
    return chunks


def demo_reranking():
    """Demonstrate reranking with CrossEncoder."""
    from src.memory import VectorStore, create_vector_store
    
    print("\n" + "=" * 60)
    print("🎯 DEMO: Reranking with CrossEncoder")
    print("=" * 60)
    
    # Sample documents
    documents = [
        "Python is a high-level interpreted programming language.",
        "Java is a compiled object-oriented language.",
        "Python supports both object-oriented and functional programming.",
        "JavaScript is mainly used for frontend web development.",
        "Machine learning in Python uses libraries like scikit-learn and TensorFlow.",
        "Python has simple and readable syntax.",
        "Rust is a systems-level programming language with safe memory management.",
    ]
    
    # Create VectorStore WITHOUT reranking
    print("\n📊 Results WITHOUT reranking:")
    store_no_rerank = VectorStore(use_reranker=False)
    store_no_rerank.add_documents(documents)
    
    query = "What are Python's features?"
    results = store_no_rerank.query(query, n_results=3)
    
    for doc, score, _ in results:
        print(f"  [{score:.3f}] {doc[:70]}...")
    
    # Create VectorStore WITH reranking
    print("\n🎯 Results WITH reranking:")
    store_with_rerank = VectorStore(use_reranker=True, reranker_model="fast")
    store_with_rerank.add_documents(documents)
    
    results_reranked = store_with_rerank.query(query, n_results=3)
    
    for doc, score, _ in results_reranked:
        print(f"  [{score:.3f}] {doc[:70]}...")
    
    print("\n💡 Note: Reranker scores are on a different scale (can be negative)")


def demo_integration():
    """Demonstrate complete integration: Chunking + VectorStore + Reranking."""
    from src.memory import SmartChunker, VectorStore
    
    print("\n" + "=" * 60)
    print("🔗 DEMO: Complete Integration")
    print("=" * 60)
    
    # Find Python files in the project
    src_dir = Path(__file__).parent.parent / "src"
    
    if not src_dir.exists():
        print(f"Directory {src_dir} not found, using current file")
        src_dir = Path(__file__).parent
    
    # Chunk the files
    chunker = SmartChunker(max_chunk_size=1000)
    all_chunks = []
    
    for py_file in list(src_dir.rglob("*.py"))[:5]:  # Limit to 5 files for demo
        try:
            chunks = chunker.chunk_file(str(py_file))
            all_chunks.extend(chunks)
            print(f"  📄 {py_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  ⚠️  {py_file.name}: {e}")
    
    if not all_chunks:
        print("No chunks extracted, skipping integration demo")
        return
    
    print(f"\n📦 Total: {len(all_chunks)} chunks")
    
    # Create VectorStore with reranking
    store = VectorStore(
        collection_name="code_search_demo",
        use_reranker=True,
    )
    
    # Add chunks
    store.add_code_chunks(all_chunks)
    
    # Sample queries
    queries = [
        "How is the vector store initialized?",
        "Where are errors handled?",
        "How does the training loop work?",
    ]
    
    print("\n🔍 Sample queries:")
    for query in queries:
        print(f"\n  Q: {query}")
        results = store.query(query, n_results=2)
        
        for doc, score, meta in results:
            name = meta.get("name", "unknown")
            source = Path(meta.get("source", "")).name
            print(f"    [{score:.2f}] {name} ({source})")


def main():
    """Run all demos."""
    print("\n" + "🚀 " * 20)
    print("    SMART CHUNKING + RERANKING DEMO")
    print("    Inspired by osgrep (github.com/Ryandonofrio3/osgrep)")
    print("🚀 " * 20)
    
    try:
        # Demo 1: Smart Chunking
        demo_smart_chunking()
        
        # Demo 2: Reranking
        demo_reranking()
        
        # Demo 3: Complete integration
        demo_integration()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\nInstall the required dependencies:")
        print("  pip install tree-sitter tree-sitter-python chromadb sentence-transformers")
    except Exception as e:
        logger.exception("Error during demo")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
