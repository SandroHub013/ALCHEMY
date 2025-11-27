"""
Esempio di uso del sistema RAG (Retrieval Augmented Generation).

Questo esempio mostra come:
1. Creare un VectorStore con documenti
2. Fare query semantiche
3. Usare il contesto per generazione

Prerequisiti:
    pip install chromadb sentence-transformers
"""

from src.memory import VectorStore
from src.agent import SEARCH_KNOWLEDGE_BASE, get_tools_prompt, SYSTEM_PROMPT_RAG


def main():
    print("=" * 60)
    print("ESEMPIO RAG - Retrieval Augmented Generation")
    print("=" * 60)
    
    # 1. Crea VectorStore
    print("\n[1] Creazione VectorStore...")
    store = VectorStore(
        collection_name="demo_kb",
        embedding_model="all-MiniLM-L6-v2",  # Modello veloce
        persist_directory=None,  # In-memory per demo
    )
    
    # 2. Aggiungi documenti
    print("\n[2] Aggiunta documenti...")
    documents = [
        "Python è un linguaggio di programmazione ad alto livello, interpretato e orientato agli oggetti.",
        "PyTorch è una libreria open source per machine learning sviluppata da Meta AI.",
        "Transformers di Hugging Face fornisce modelli pre-addestrati per NLP.",
        "QLoRA è una tecnica per fine-tuning efficiente di LLM con quantizzazione 4-bit.",
        "Agent Lightning di Microsoft è un framework per training di agenti AI con RL.",
        "ChromaDB è un database vettoriale open source per applicazioni AI.",
        "RAG (Retrieval Augmented Generation) combina recupero di informazioni con generazione.",
    ]
    
    store.add_documents(documents)
    print(f"   Aggiunti {store.count()} documenti")
    
    # 3. Query semantiche
    print("\n[3] Query semantiche...")
    queries = [
        "Cos'è Python?",
        "Come funziona il fine-tuning efficiente?",
        "Che framework uso per agenti AI?",
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = store.query(query, n_results=2)
        for doc, score, _ in results:
            print(f"   → Score {score:.3f}: {doc[:80]}...")
    
    # 4. Contesto per RAG
    print("\n[4] Generazione contesto RAG...")
    context = store.query_with_context("Come faccio training di LLM?", n_results=3)
    print(f"\n   Contesto recuperato:\n{context}")
    
    # 5. Tool definition
    print("\n[5] Tool search_knowledge_base:")
    print(f"   {SEARCH_KNOWLEDGE_BASE.to_prompt_format()}")
    
    # 6. System prompt per RAG
    print("\n[6] System Prompt RAG:")
    print(SYSTEM_PROMPT_RAG[:300] + "...")
    
    print("\n" + "=" * 60)
    print("✅ Demo RAG completata!")
    print("=" * 60)


if __name__ == "__main__":
    main()

