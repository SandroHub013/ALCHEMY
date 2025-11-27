#!/usr/bin/env python3
"""
Esempio di utilizzo Smart Chunking + Reranking.

Dimostra come usare le nuove funzionalità ispirate a osgrep:
1. Smart Chunking con tree-sitter per codice Python
2. Reranking con CrossEncoder per risultati più accurati
3. Integrazione con VectorStore esistente

Prerequisiti:
    pip install tree-sitter tree-sitter-python

Uso:
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
    """Dimostra il chunking intelligente del codice Python."""
    from src.memory import SmartChunker, ChunkType
    
    print("\n" + "=" * 60)
    print("🔧 DEMO: Smart Chunking con Tree-sitter")
    print("=" * 60)
    
    # Esempio di codice Python da chunkare
    sample_code = '''
"""Modulo di esempio per dimostrazione chunking."""

import os
import sys
from typing import List, Optional

# Costanti
MAX_ITEMS = 100
DEFAULT_TIMEOUT = 30


class DataProcessor:
    """
    Classe per processare dati.
    
    Gestisce la trasformazione e validazione dei dati
    prima del salvataggio.
    """
    
    def __init__(self, config: dict):
        """Inizializza il processor con configurazione."""
        self.config = config
        self.cache = {}
    
    def process(self, data: List[dict]) -> List[dict]:
        """
        Processa una lista di record.
        
        Args:
            data: Lista di dizionari da processare
            
        Returns:
            Lista di record processati
        """
        results = []
        for item in data:
            processed = self._transform(item)
            if self._validate(processed):
                results.append(processed)
        return results
    
    def _transform(self, item: dict) -> dict:
        """Trasforma un singolo record."""
        return {k.lower(): v for k, v in item.items()}
    
    def _validate(self, item: dict) -> bool:
        """Valida un record."""
        return bool(item)


def load_data(path: str) -> List[dict]:
    """
    Carica dati da file.
    
    Args:
        path: Percorso al file JSON
        
    Returns:
        Lista di record caricati
    """
    import json
    with open(path) as f:
        return json.load(f)


def main():
    """Entry point principale."""
    processor = DataProcessor({"verbose": True})
    data = [{"Name": "Alice"}, {"Name": "Bob"}]
    result = processor.process(data)
    print(f"Processati {len(result)} record")


if __name__ == "__main__":
    main()
'''
    
    # Crea chunker
    chunker = SmartChunker(
        max_chunk_size=1500,  # Chunk più grandi per demo
        min_chunk_size=50,
        include_imports=True,
    )
    
    # Chunka il codice
    chunks = chunker.chunk_python_code(sample_code, file_path="example.py")
    
    print(f"\n📦 Estratti {len(chunks)} chunks:\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk.chunk_type.value.upper()}: {chunk.qualified_name}")
        print(f"      Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"      Size: {chunk.char_count} chars")
        if chunk.docstring:
            print(f"      Doc: {chunk.docstring[:60]}...")
        print()
    
    return chunks


def demo_reranking():
    """Dimostra il reranking con CrossEncoder."""
    from src.memory import VectorStore, create_vector_store
    
    print("\n" + "=" * 60)
    print("🎯 DEMO: Reranking con CrossEncoder")
    print("=" * 60)
    
    # Documenti di esempio
    documents = [
        "Python è un linguaggio di programmazione interpretato ad alto livello.",
        "Java è un linguaggio compilato orientato agli oggetti.",
        "Python supporta sia la programmazione orientata agli oggetti che quella funzionale.",
        "JavaScript è usato principalmente per lo sviluppo web frontend.",
        "Il machine learning in Python usa librerie come scikit-learn e TensorFlow.",
        "Python ha una sintassi semplice e leggibile.",
        "Rust è un linguaggio di programmazione systems-level con gestione memoria sicura.",
    ]
    
    # Crea VectorStore SENZA reranking
    print("\n📊 Risultati SENZA reranking:")
    store_no_rerank = VectorStore(use_reranker=False)
    store_no_rerank.add_documents(documents)
    
    query = "Quali sono le caratteristiche di Python?"
    results = store_no_rerank.query(query, n_results=3)
    
    for doc, score, _ in results:
        print(f"  [{score:.3f}] {doc[:70]}...")
    
    # Crea VectorStore CON reranking
    print("\n🎯 Risultati CON reranking:")
    store_with_rerank = VectorStore(use_reranker=True, reranker_model="fast")
    store_with_rerank.add_documents(documents)
    
    results_reranked = store_with_rerank.query(query, n_results=3)
    
    for doc, score, _ in results_reranked:
        print(f"  [{score:.3f}] {doc[:70]}...")
    
    print("\n💡 Nota: Gli score del reranker sono su scala diversa (possono essere negativi)")


def demo_integration():
    """Dimostra l'integrazione completa: Chunking + VectorStore + Reranking."""
    from src.memory import SmartChunker, VectorStore
    
    print("\n" + "=" * 60)
    print("🔗 DEMO: Integrazione Completa")
    print("=" * 60)
    
    # Trova file Python nel progetto
    src_dir = Path(__file__).parent.parent / "src"
    
    if not src_dir.exists():
        print(f"Directory {src_dir} non trovata, uso file corrente")
        src_dir = Path(__file__).parent
    
    # Chunka i file
    chunker = SmartChunker(max_chunk_size=1000)
    all_chunks = []
    
    for py_file in list(src_dir.rglob("*.py"))[:5]:  # Limita a 5 file per demo
        try:
            chunks = chunker.chunk_file(str(py_file))
            all_chunks.extend(chunks)
            print(f"  📄 {py_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  ⚠️  {py_file.name}: {e}")
    
    if not all_chunks:
        print("Nessun chunk estratto, skip demo integrazione")
        return
    
    print(f"\n📦 Totale: {len(all_chunks)} chunks")
    
    # Crea VectorStore con reranking
    store = VectorStore(
        collection_name="code_search_demo",
        use_reranker=True,
    )
    
    # Aggiungi chunks
    store.add_code_chunks(all_chunks)
    
    # Query di esempio
    queries = [
        "Come si inizializza il vector store?",
        "Dove vengono gestiti gli errori?",
        "Come funziona il training loop?",
    ]
    
    print("\n🔍 Query di esempio:")
    for query in queries:
        print(f"\n  Q: {query}")
        results = store.query(query, n_results=2)
        
        for doc, score, meta in results:
            name = meta.get("name", "unknown")
            source = Path(meta.get("source", "")).name
            print(f"    [{score:.2f}] {name} ({source})")


def main():
    """Esegue tutte le demo."""
    print("\n" + "🚀 " * 20)
    print("    SMART CHUNKING + RERANKING DEMO")
    print("    Ispirato a osgrep (github.com/Ryandonofrio3/osgrep)")
    print("🚀 " * 20)
    
    try:
        # Demo 1: Smart Chunking
        demo_smart_chunking()
        
        # Demo 2: Reranking
        demo_reranking()
        
        # Demo 3: Integrazione completa
        demo_integration()
        
        print("\n" + "=" * 60)
        print("✅ Demo completata con successo!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ Dipendenza mancante: {e}")
        print("\nInstalla le dipendenze necessarie:")
        print("  pip install tree-sitter tree-sitter-python chromadb sentence-transformers")
    except Exception as e:
        logger.exception("Errore durante la demo")
        print(f"\n❌ Errore: {e}")


if __name__ == "__main__":
    main()

