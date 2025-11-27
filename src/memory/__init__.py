"""
Modulo per la gestione della memoria (RAG + Procedurale + Smart Chunking).

Componenti principali:
- VectorStore: Database vettoriale per RAG con ChromaDB
- Reranker: Reranking dei risultati con CrossEncoder
- SmartChunker: Chunking intelligente del codice con tree-sitter
- SOPManager: Gestione delle Standard Operating Procedures
"""

from .vector_store import (
    VectorStore,
    Document,
    Reranker,
    create_vector_store,
    load_documents_from_file,
)
from .procedural_memory import (
    SOP,
    SOPStep,
    SOPManager,
    StepStatus,
    get_system_prompt_with_sop,
    SYSTEM_PROMPT_WITH_SOP,
)
from .smart_chunker import (
    SmartChunker,
    CodeChunk,
    ChunkType,
    chunk_python_file,
    chunk_python_code,
    chunks_to_documents,
)

__all__ = [
    # Vector Store (RAG)
    "VectorStore",
    "Document",
    "Reranker",
    "create_vector_store",
    "load_documents_from_file",
    # Smart Chunking
    "SmartChunker",
    "CodeChunk",
    "ChunkType",
    "chunk_python_file",
    "chunk_python_code",
    "chunks_to_documents",
    # Procedural Memory (SOP)
    "SOP",
    "SOPStep",
    "SOPManager",
    "StepStatus",
    "get_system_prompt_with_sop",
    "SYSTEM_PROMPT_WITH_SOP",
]
