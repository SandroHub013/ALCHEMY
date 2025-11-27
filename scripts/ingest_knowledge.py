#!/usr/bin/env python3
"""
Data Ingestion Pipeline for RAG.

This script loads documents from a folder, processes them with intelligent
splitting based on file type, and saves them to ChromaDB.

Usage:
    python scripts/ingest_knowledge.py --source_dir ./docs
    python scripts/ingest_knowledge.py --source_dir ./codebase --collection code_kb
    python scripts/ingest_knowledge.py --source_dir ./sops --collection sop_memory

Supported file types:
    - .py: PythonCodeTextSplitter (preserves functions/classes)
    - .md, .txt: RecursiveCharacterTextSplitter
    - .pdf: PyPDFLoader + RecursiveCharacterTextSplitter

Features:
    - Intelligent splitting by file type
    - Metadata for citation (file, line, type)
    - Progress bar with tqdm
    - Persistent ChromaDB on disk
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
    print(f"LangChain import error: {e}")
    print("Install with: pip install langchain langchain-community langchain-text-splitters")
    sys.exit(1)

# ChromaDB and Embeddings
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Import error: {e}")
    print("Install with: pip install chromadb sentence-transformers")
    sys.exit(1)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION = "knowledge_base"

# Supported extensions by type
FILE_TYPES = {
    "python": [".py"],
    "markdown": [".md"],
    "text": [".txt", ".rst", ".log"],
    "pdf": [".pdf"],
}


# =============================================================================
# SPLITTERS BY FILE TYPE
# =============================================================================

def get_python_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Create a splitter optimized for Python code.
    
    Uses specific separators to preserve:
    - Class definitions
    - Function definitions
    - Docstrings
    - Imports
    """
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_markdown_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Create a splitter optimized for Markdown.
    
    Respects header and section structure.
    """
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Create a generic text splitter.
    
    Uses common separators: paragraphs, sentences, words.
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
    """Load a Python file with detailed metadata."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
        
        # Create document with metadata
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
        logger.warning(f"Error loading {file_path}: {e}")
        return []


def load_text_file(file_path: Path, file_type: str = "text") -> List[Document]:
    """Load a generic text file."""
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
        logger.warning(f"Error loading {file_path}: {e}")
        return []


def load_pdf_file(file_path: Path) -> List[Document]:
    """Load a PDF file."""
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        
        # Add metadata
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
        logger.warning(f"Error loading PDF {file_path}: {e}")
        return []


def load_markdown_file(file_path: Path) -> List[Document]:
    """Load a Markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract title if present
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
        logger.warning(f"Error loading {file_path}: {e}")
        return []


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

class KnowledgeIngester:
    """
    Document ingestion pipeline for RAG.
    
    Loads documents, splits them intelligently based on type,
    generates embeddings and saves them to ChromaDB.
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
        Initialize the ingester.
        
        Args:
            chroma_path: Path for persistent ChromaDB
            collection_name: Collection name
            embedding_model: sentence-transformers model
            chunk_size: Chunk size
            chunk_overlap: Overlap between chunks
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        os.makedirs(chroma_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(f"ChromaDB initialized: {chroma_path}/{collection_name}")
        logger.info(f"Existing documents: {self.collection.count()}")
        
        # Initialize splitters
        self.python_splitter = get_python_splitter(chunk_size, chunk_overlap)
        self.markdown_splitter = get_markdown_splitter(chunk_size, chunk_overlap)
        self.text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    
    def _get_file_type(self, file_path: Path) -> Optional[str]:
        """Determine file type from extension."""
        suffix = file_path.suffix.lower()
        
        for file_type, extensions in FILE_TYPES.items():
            if suffix in extensions:
                return file_type
        
        return None
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """Load a file based on type."""
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
        """Split a document based on type."""
        file_type = doc.metadata.get("file_type", "text")
        
        if file_type == "python":
            chunks = self.python_splitter.split_documents([doc])
        elif file_type == "markdown":
            chunks = self.markdown_splitter.split_documents([doc])
        else:
            chunks = self.text_splitter.split_documents([doc])
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
            # Calculate approximate starting line
            if "page_content" in doc.__dict__:
                original_content = doc.page_content
                chunk_start = original_content.find(chunk.page_content[:50])
                if chunk_start >= 0:
                    start_line = original_content[:chunk_start].count("\n") + 1
                    chunk.metadata["start_line"] = start_line
        
        return chunks
    
    def _generate_chunk_id(self, chunk: Document) -> str:
        """Generate a unique ID for a chunk."""
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
        Ingest all files from a directory.
        
        Args:
            source_dir: Source directory
            recursive: Whether to search recursively
            file_extensions: Extensions to search for (None = all supported)
            
        Returns:
            Ingestion statistics
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Directory not found: {source_dir}")
        
        # Determine extensions to search for
        if file_extensions is None:
            file_extensions = []
            for exts in FILE_TYPES.values():
                file_extensions.extend(exts)
        
        # Find all files
        files = []
        for ext in file_extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(source_path.glob(pattern))
        
        files = sorted(set(files))  # Remove duplicates
        
        logger.info(f"Found {len(files)} files to process")
        
        if not files:
            return {"files_processed": 0, "chunks_added": 0}
        
        # Process files
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_added": 0,
            "errors": 0,
        }
        
        all_chunks = []
        
        for file_path in tqdm(files, desc="Loading files"):
            try:
                # Load file
                docs = self._load_file(file_path)
                
                if not docs:
                    stats["files_skipped"] += 1
                    continue
                
                # Split documents
                for doc in docs:
                    chunks = self._split_document(doc)
                    all_chunks.extend(chunks)
                
                stats["files_processed"] += 1
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                stats["errors"] += 1
        
        logger.info(f"Generated {len(all_chunks)} total chunks")
        
        if not all_chunks:
            return stats
        
        # Generate embeddings and save
        logger.info("Generating embeddings...")
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for chunk in tqdm(all_chunks, desc="Embedding"):
            chunk_id = self._generate_chunk_id(chunk)
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append(chunk.metadata)
        
        # Generate embeddings in batch
        logger.info("Computing embeddings in batch...")
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()
        
        # Save to ChromaDB
        logger.info(f"Saving {len(documents)} chunks to ChromaDB...")
        
        # ChromaDB has a batch limit, process in chunks
        batch_size = 500
        for i in tqdm(range(0, len(documents), batch_size), desc="Saving"):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_emb = embeddings[i:i + batch_size]
            
            # Use upsert to avoid duplicates
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_emb,
            )
        
        stats["chunks_added"] = len(documents)
        
        logger.info(f"Ingestion completed! Total documents: {self.collection.count()}")
        
        return stats
    
    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of chunks added
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load and split
        docs = self._load_file(path)
        
        if not docs:
            return 0
        
        all_chunks = []
        for doc in docs:
            chunks = self._split_document(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return 0
        
        # Generate embeddings and save
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
        Query the knowledge base.
        
        Args:
            query: Text query
            n_results: Number of results
            file_type: Optional filter by file type
            
        Returns:
            List of (document, score, metadata)
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Prepare filter
        where = {"file_type": file_type} if file_type else None
        
        # Query
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        
        # Format results
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                output.append((doc, similarity, metadata))
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """Return knowledge base statistics."""
        count = self.collection.count()
        
        # Count by type
        type_counts = {}
        if count > 0:
            # Sample to get distribution
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
        description="Document ingestion for RAG Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest a documentation folder
    python ingest_knowledge.py --source_dir ./docs
    
    # Ingest source code
    python ingest_knowledge.py --source_dir ./src --collection code_kb
    
    # Ingest only Python files
    python ingest_knowledge.py --source_dir ./src --extensions .py
    
    # Query the knowledge base
    python ingest_knowledge.py --query "How does training work?"
        """,
    )
    
    parser.add_argument(
        "--source_dir",
        type=str,
        help="Source directory to ingest",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--chroma_path",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help=f"ChromaDB path (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        help="File extensions to process (e.g. .py .md)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search recursively",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Execute a query instead of ingesting",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show knowledge base statistics",
    )
    
    args = parser.parse_args()
    
    # Initialize ingester
    ingester = KnowledgeIngester(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    
    # Stats mode
    if args.stats:
        stats = ingester.get_stats()
        print("\n📊 Knowledge Base Statistics:")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Path: {stats['chroma_path']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        if stats['type_distribution']:
            print("   Distribution by type:")
            for t, c in stats['type_distribution'].items():
                print(f"     - {t}: {c}")
        return
    
    # Query mode
    if args.query:
        print(f"\n🔍 Query: '{args.query}'")
        results = ingester.query(args.query, n_results=5)
        
        if not results:
            print("   No results found.")
        else:
            for i, (doc, score, meta) in enumerate(results, 1):
                source = meta.get("file_name", "unknown")
                line = meta.get("start_line", "?")
                print(f"\n   [{i}] Score: {score:.3f} | {source}:{line}")
                print(f"       {doc[:200]}...")
        return
    
    # Ingestion mode
    if args.source_dir:
        print(f"\n📥 Ingesting from: {args.source_dir}")
        
        stats = ingester.ingest_directory(
            source_dir=args.source_dir,
            recursive=not args.no_recursive,
            file_extensions=args.extensions,
        )
        
        print("\n✅ Ingestion completed!")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Files skipped: {stats['files_skipped']}")
        print(f"   Chunks added: {stats['chunks_added']}")
        print(f"   Errors: {stats['errors']}")
        print(f"   Total in DB: {ingester.collection.count()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
