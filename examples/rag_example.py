"""
Example of RAG (Retrieval Augmented Generation) system usage.

This example shows how to:
1. Create a VectorStore with documents
2. Perform semantic queries
3. Use context for generation

Prerequisites:
    pip install chromadb sentence-transformers
"""

from src.memory import VectorStore
from src.agent import SEARCH_KNOWLEDGE_BASE, get_tools_prompt, SYSTEM_PROMPT_RAG


def main():
    print("=" * 60)
    print("RAG EXAMPLE - Retrieval Augmented Generation")
    print("=" * 60)
    
    # 1. Create VectorStore
    print("\n[1] Creating VectorStore...")
    store = VectorStore(
        collection_name="demo_kb",
        embedding_model="all-MiniLM-L6-v2",  # Fast model
        persist_directory=None,  # In-memory for demo
    )
    
    # 2. Add documents
    print("\n[2] Adding documents...")
    documents = [
        "Python is a high-level, interpreted, object-oriented programming language.",
        "PyTorch is an open source machine learning library developed by Meta AI.",
        "HuggingFace Transformers provides pre-trained models for NLP.",
        "QLoRA is a technique for efficient LLM fine-tuning with 4-bit quantization.",
        "Microsoft Agent Lightning is a framework for training AI agents with RL.",
        "ChromaDB is an open source vector database for AI applications.",
        "RAG (Retrieval Augmented Generation) combines information retrieval with generation.",
    ]
    
    store.add_documents(documents)
    print(f"   Added {store.count()} documents")
    
    # 3. Semantic queries
    print("\n[3] Semantic queries...")
    queries = [
        "What is Python?",
        "How does efficient fine-tuning work?",
        "What framework do I use for AI agents?",
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = store.query(query, n_results=2)
        for doc, score, _ in results:
            print(f"   → Score {score:.3f}: {doc[:80]}...")
    
    # 4. Context for RAG
    print("\n[4] Generating RAG context...")
    context = store.query_with_context("How do I train LLMs?", n_results=3)
    print(f"\n   Retrieved context:\n{context}")
    
    # 5. Tool definition
    print("\n[5] search_knowledge_base tool:")
    print(f"   {SEARCH_KNOWLEDGE_BASE.to_prompt_format()}")
    
    # 6. System prompt for RAG
    print("\n[6] RAG System Prompt:")
    print(SYSTEM_PROMPT_RAG[:300] + "...")
    
    print("\n" + "=" * 60)
    print("✅ RAG Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
