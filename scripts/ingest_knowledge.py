#!/usr/bin/env python3
"""
Pipeline di Ingestione Dati per RAG.

Questo script carica documenti da una cartella, li processa con splitting
intelligente in base al tipo di file, e li salva in ChromaDB.

Uso:
    python scripts/ingest_knowledge.py --source_dir ./docs
    python scripts/ingest_knowledge.py --source_dir ./codebase --collection code_kb
    python scripts/ingest_knowledge.py --source_dir ./sops --collection sop_memory

Tipi di file supportati:
    - .py: PythonCodeTextSplitter (preserva funzioni/classi)
    - .md, .txt: RecursiveCharacterTextSplitter
    - .pdf: PyPDFLoader + RecursiveCharacterTextSplitter

Features:
    - Splitting intelligente per tipo di file
    - Metadata per citazione (file, riga, tipo)
    - Progress bar con tqdm
    - ChromaDB persistente su disco
"""

import argparse
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Progress bar
from tqdm import tqdm

# LangChain imports
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        Language,
    )
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        DirectoryLoader,
        UnstructuredMarkdownLoader,
    )
    from langchain.schema import Document
except ImportError as e:
    print(f"Errore import LangChain: {e}")
    print("Installa con: pip install langchain langchain-community langchain-text-splitters")
    sys.exit(1)

# ChromaDB e Embeddings
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Errore import: {e}")
    print("Installa con: pip install chromadb sentence-transformers")
    sys.exit(1)

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION = "knowledge_base"

# Estensioni supportate per tipo
FILE_TYPES = {
    "python": [".py"],
    "markdown": [".md"],
    "text": [".txt", ".rst", ".log"],
    "pdf": [".pdf"],
}


# =============================================================================
# SPLITTERS PER TIPO DI FILE
# =============================================================================

def get_python_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Crea uno splitter ottimizzato per codice Python.
    
    Usa separatori specifici per preservare:
    - Definizioni di classi
    - Definizioni di funzioni
    - Docstring
    - Import
    """
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_markdown_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Crea uno splitter ottimizzato per Markdown.
    
    Rispetta la struttura dei titoli e delle sezioni.
    """
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Crea uno splitter generico per testo.
    
    Usa separatori comuni: paragrafi, frasi, parole.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )


# =============================================================================
# DOCUMENT LOADERS
# =============================================================================

def load_python_file(file_path: Path) -> List[Document]:
    """Carica un file Python con metadata dettagliati."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
        
        # Crea documento con metadata
        doc = Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "file_type": "python",
                "file_name": file_path.name,
                "total_lines": len(lines),
                "ingested_at": datetime.now().isoformat(),
            }
        )
        return [doc]
    except Exception as e:
        logger.warning(f"Errore caricamento {file_path}: {e}")
        return []


def load_text_file(file_path: Path, file_type: str = "text") -> List[Document]:
    """Carica un file di testo generico."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        doc = Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "file_type": file_type,
                "file_name": file_path.name,
                "ingested_at": datetime.now().isoformat(),
            }
        )
        return [doc]
    except Exception as e:
        logger.warning(f"Errore caricamento {file_path}: {e}")
        return []


def load_pdf_file(file_path: Path) -> List[Document]:
    """Carica un file PDF."""
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        
        # Aggiungi metadata
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source": str(file_path),
                "file_type": "pdf",
                "file_name": file_path.name,
                "page_number": i + 1,
                "ingested_at": datetime.now().isoformat(),
            })
        
        return docs
    except Exception as e:
        logger.warning(f"Errore caricamento PDF {file_path}: {e}")
        return []


def load_markdown_file(file_path: Path) -> List[Document]:
    """Carica un file Markdown."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Estrai titolo se presente
        title = None
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        
        doc = Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "file_type": "markdown",
                "file_name": file_path.name,
                "title": title,
                "ingested_at": datetime.now().isoformat(),
            }
        )
        return [doc]
    except Exception as e:
        logger.warning(f"Errore caricamento {file_path}: {e}")
        return []


# =============================================================================
# PIPELINE DI INGESTIONE
# =============================================================================

class KnowledgeIngester:
    """
    Pipeline di ingestione documenti per RAG.
    
    Carica documenti, li splitta intelligentemente in base al tipo,
    genera embedding e li salva in ChromaDB.
    """
    
    def __init__(
        self,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Inizializza l'ingester.
        
        Args:
            chroma_path: Path per ChromaDB persistente
            collection_name: Nome della collezione
            embedding_model: Modello sentence-transformers
            chunk_size: Dimensione chunk
            chunk_overlap: Overlap tra chunk
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Inizializza embedding model
        logger.info(f"Caricamento modello embedding: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Inizializza ChromaDB
        os.makedirs(chroma_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Ottieni o crea collezione
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(f"ChromaDB inizializzato: {chroma_path}/{collection_name}")
        logger.info(f"Documenti esistenti: {self.collection.count()}")
        
        # Inizializza splitters
        self.python_splitter = get_python_splitter(chunk_size, chunk_overlap)
        self.markdown_splitter = get_markdown_splitter(chunk_size, chunk_overlap)
        self.text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    
    def _get_file_type(self, file_path: Path) -> Optional[str]:
        """Determina il tipo di file dall'estensione."""
        suffix = file_path.suffix.lower()
        
        for file_type, extensions in FILE_TYPES.items():
            if suffix in extensions:
                return file_type
        
        return None
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """Carica un file in base al tipo."""
        file_type = self._get_file_type(file_path)
        
        if file_type == "python":
            return load_python_file(file_path)
        elif file_type == "markdown":
            return load_markdown_file(file_path)
        elif file_type == "text":
            return load_text_file(file_path)
        elif file_type == "pdf":
            return load_pdf_file(file_path)
        else:
            return []
    
    def _split_document(self, doc: Document) -> List[Document]:
        """Splitta un documento in base al tipo."""
        file_type = doc.metadata.get("file_type", "text")
        
        if file_type == "python":
            chunks = self.python_splitter.split_documents([doc])
        elif file_type == "markdown":
            chunks = self.markdown_splitter.split_documents([doc])
        else:
            chunks = self.text_splitter.split_documents([doc])
        
        # Aggiungi metadata ai chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
            # Calcola riga di partenza approssimativa
            if "page_content" in doc.__dict__:
                original_content = doc.page_content
                chunk_start = original_content.find(chunk.page_content[:50])
                if chunk_start >= 0:
                    start_line = original_content[:chunk_start].count("\n") + 1
                    chunk.metadata["start_line"] = start_line
        
        return chunks
    
    def _generate_chunk_id(self, chunk: Document) -> str:
        """Genera un ID univoco per un chunk."""
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        source = chunk.metadata.get("source", "unknown")
        chunk_idx = chunk.metadata.get("chunk_index", 0)
        return f"{Path(source).stem}_{chunk_idx}_{content_hash}"
    
    def ingest_directory(
        self,
        source_dir: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Ingesta tutti i file da una directory.
        
        Args:
            source_dir: Directory sorgente
            recursive: Se cercare ricorsivamente
            file_extensions: Estensioni da cercare (None = tutte supportate)
            
        Returns:
            Statistiche di ingestione
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Directory non trovata: {source_dir}")
        
        # Determina estensioni da cercare
        if file_extensions is None:
            file_extensions = []
            for exts in FILE_TYPES.values():
                file_extensions.extend(exts)
        
        # Trova tutti i file
        files = []
        for ext in file_extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(source_path.glob(pattern))
        
        files = sorted(set(files))  # Rimuovi duplicati
        
        logger.info(f"Trovati {len(files)} file da processare")
        
        if not files:
            return {"files_processed": 0, "chunks_added": 0}
        
        # Processa file
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_added": 0,
            "errors": 0,
        }
        
        all_chunks = []
        
        for file_path in tqdm(files, desc="Caricamento file"):
            try:
                # Carica file
                docs = self._load_file(file_path)
                
                if not docs:
                    stats["files_skipped"] += 1
                    continue
                
                # Splitta documenti
                for doc in docs:
                    chunks = self._split_document(doc)
                    all_chunks.extend(chunks)
                
                stats["files_processed"] += 1
                
            except Exception as e:
                logger.warning(f"Errore processando {file_path}: {e}")
                stats["errors"] += 1
        
        logger.info(f"Generati {len(all_chunks)} chunk totali")
        
        if not all_chunks:
            return stats
        
        # Genera embedding e salva
        logger.info("Generazione embedding...")
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for chunk in tqdm(all_chunks, desc="Embedding"):
            chunk_id = self._generate_chunk_id(chunk)
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append(chunk.metadata)
        
        # Genera embedding in batch
        logger.info("Calcolo embedding in batch...")
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()
        
        # Salva in ChromaDB
        logger.info(f"Salvataggio {len(documents)} chunk in ChromaDB...")
        
        # ChromaDB ha un limite di batch, processiamo in chunk
        batch_size = 500
        for i in tqdm(range(0, len(documents), batch_size), desc="Salvataggio"):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_emb = embeddings[i:i + batch_size]
            
            # Usa upsert per evitare duplicati
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_emb,
            )
        
        stats["chunks_added"] = len(documents)
        
        logger.info(f"Ingestione completata! Totale documenti: {self.collection.count()}")
        
        return stats
    
    def ingest_file(self, file_path: str) -> int:
        """
        Ingesta un singolo file.
        
        Args:
            file_path: Path al file
            
        Returns:
            Numero di chunk aggiunti
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")
        
        # Carica e splitta
        docs = self._load_file(path)
        
        if not docs:
            return 0
        
        all_chunks = []
        for doc in docs:
            chunks = self._split_document(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return 0
        
        # Genera embedding e salva
        ids = [self._generate_chunk_id(c) for c in all_chunks]
        documents = [c.page_content for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        
        return len(documents)
    
    def query(
        self,
        query: str,
        n_results: int = 5,
        file_type: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Query sulla knowledge base.
        
        Args:
            query: Query testuale
            n_results: Numero di risultati
            file_type: Filtro opzionale per tipo file
            
        Returns:
            Lista di (documento, score, metadata)
        """
        # Genera embedding query
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Prepara filtro
        where = {"file_type": file_type} if file_type else None
        
        # Query
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        
        # Formatta risultati
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                output.append((doc, similarity, metadata))
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche sulla knowledge base."""
        count = self.collection.count()
        
        # Conta per tipo
        type_counts = {}
        if count > 0:
            # Campiona per ottenere distribuzione
            sample = self.collection.get(limit=min(count, 1000), include=["metadatas"])
            for meta in sample.get("metadatas", []):
                file_type = meta.get("file_type", "unknown")
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "chroma_path": self.chroma_path,
            "type_distribution": type_counts,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingestione documenti per RAG Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
    # Ingesta una cartella di documentazione
    python ingest_knowledge.py --source_dir ./docs
    
    # Ingesta codice sorgente
    python ingest_knowledge.py --source_dir ./src --collection code_kb
    
    # Ingesta solo file Python
    python ingest_knowledge.py --source_dir ./src --extensions .py
    
    # Query sulla knowledge base
    python ingest_knowledge.py --query "Come funziona il training?"
        """,
    )
    
    parser.add_argument(
        "--source_dir",
        type=str,
        help="Directory sorgente da ingestare",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Nome collezione ChromaDB (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--chroma_path",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help=f"Path ChromaDB (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Dimensione chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap chunk (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        help="Estensioni file da processare (es. .py .md)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Non cercare ricorsivamente",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Esegui una query invece di ingestare",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostra statistiche della knowledge base",
    )
    
    args = parser.parse_args()
    
    # Inizializza ingester
    ingester = KnowledgeIngester(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    
    # Modalità stats
    if args.stats:
        stats = ingester.get_stats()
        print("\n📊 Statistiche Knowledge Base:")
        print(f"   Collezione: {stats['collection_name']}")
        print(f"   Path: {stats['chroma_path']}")
        print(f"   Totale chunk: {stats['total_chunks']}")
        if stats['type_distribution']:
            print("   Distribuzione per tipo:")
            for t, c in stats['type_distribution'].items():
                print(f"     - {t}: {c}")
        return
    
    # Modalità query
    if args.query:
        print(f"\n🔍 Query: '{args.query}'")
        results = ingester.query(args.query, n_results=5)
        
        if not results:
            print("   Nessun risultato trovato.")
        else:
            for i, (doc, score, meta) in enumerate(results, 1):
                source = meta.get("file_name", "unknown")
                line = meta.get("start_line", "?")
                print(f"\n   [{i}] Score: {score:.3f} | {source}:{line}")
                print(f"       {doc[:200]}...")
        return
    
    # Modalità ingestione
    if args.source_dir:
        print(f"\n📥 Ingestione da: {args.source_dir}")
        
        stats = ingester.ingest_directory(
            source_dir=args.source_dir,
            recursive=not args.no_recursive,
            file_extensions=args.extensions,
        )
        
        print("\n✅ Ingestione completata!")
        print(f"   File processati: {stats['files_processed']}")
        print(f"   File saltati: {stats['files_skipped']}")
        print(f"   Chunk aggiunti: {stats['chunks_added']}")
        print(f"   Errori: {stats['errors']}")
        print(f"   Totale in DB: {ingester.collection.count()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

