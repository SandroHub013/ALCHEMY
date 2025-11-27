"""
Vector Store per Retrieval Augmented Generation (RAG).

Implementa un wrapper semplice attorno a ChromaDB con embedding locali
generati da sentence-transformers.

Features:
- Embedding locali con sentence-transformers
- Reranking opzionale con CrossEncoder (ispirato a osgrep)
- Smart chunking per codice Python con tree-sitter

Uso:
    ```python
    from src.memory import VectorStore
    
    # Inizializza (con reranking opzionale)
    store = VectorStore(use_reranker=True)
    
    # Aggiungi documenti
    store.add_documents([
        "Python è un linguaggio di programmazione.",
        "Machine Learning usa algoritmi per imparare dai dati.",
    ])
    
    # Query (reranking automatico se abilitato)
    results = store.query("Cos'è Python?", n_results=3)
    for doc, score, metadata in results:
        print(f"Score: {score:.3f} - {doc}")
    ```
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging
import os
import hashlib

logger = logging.getLogger(__name__)

# Lazy imports per evitare errori se non installati
_chromadb = None
_sentence_transformers = None
_cross_encoder = None


def _get_chromadb():
    """Lazy import di ChromaDB."""
    global _chromadb
    if _chromadb is None:
        try:
            import chromadb
            _chromadb = chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB non installato. Installa con: pip install chromadb"
            )
    return _chromadb


def _get_sentence_transformers():
    """Lazy import di sentence-transformers."""
    global _sentence_transformers
    if _sentence_transformers is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformers = SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers non installato. "
                "Installa con: pip install sentence-transformers"
            )
    return _sentence_transformers


def _get_cross_encoder():
    """Lazy import di CrossEncoder per reranking."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers non installato o versione troppo vecchia. "
                "Installa/aggiorna con: pip install -U sentence-transformers"
            )
    return _cross_encoder


@dataclass
class Document:
    """Rappresenta un documento nel vector store."""
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    @classmethod
    def from_text(cls, text: str, metadata: Optional[Dict[str, Any]] = None) -> "Document":
        """Crea un documento da testo, generando un ID univoco."""
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        return cls(id=doc_id, text=text, metadata=metadata or {})


class Reranker:
    """
    Reranker basato su CrossEncoder per migliorare i risultati di ricerca.
    
    Ispirato a osgrep che usa ColBERT per reranking.
    CrossEncoder valuta la rilevanza di ogni coppia (query, documento)
    direttamente, producendo score più accurati del semplice cosine similarity.
    
    Attributes:
        model_name: Nome del modello CrossEncoder
        
    Modelli consigliati (dal più veloce al più accurato):
        - "cross-encoder/ms-marco-MiniLM-L-6-v2": Veloce, buona qualità
        - "cross-encoder/ms-marco-MiniLM-L-12-v2": Bilanciato
        - "BAAI/bge-reranker-base": Alta qualità, multilingue
    """
    
    RERANKER_MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "accurate": "BAAI/bge-reranker-base",
    }
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Inizializza il Reranker.
        
        Args:
            model_name: Nome del modello o shortcut ("fast", "balanced", "accurate")
        """
        # Risolvi shortcut
        self.model_name = self.RERANKER_MODELS.get(model_name, model_name)
        
        logger.info(f"Caricamento reranker: {self.model_name}")
        CrossEncoder = _get_cross_encoder()
        self.model = CrossEncoder(self.model_name)
        logger.info("Reranker caricato")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Riordina i documenti per rilevanza rispetto alla query.
        
        Args:
            query: Query di ricerca
            documents: Lista di documenti da riordinare
            top_k: Numero di risultati da restituire (None = tutti)
            
        Returns:
            Lista di tuple (indice_originale, score) ordinate per score decrescente
        """
        if not documents:
            return []
        
        # Crea coppie query-documento
        pairs = [(query, doc) for doc in documents]
        
        # Calcola score
        scores = self.model.predict(pairs)
        
        # Crea lista (indice, score) e ordina
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: -x[1])
        
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores
    
    def rerank_with_docs(
        self,
        query: str,
        documents: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Riordina risultati di query (con metadata) per rilevanza.
        
        Args:
            query: Query di ricerca
            documents: Lista di tuple (documento, score_originale, metadata)
            top_k: Numero di risultati da restituire
            
        Returns:
            Lista riordinata di tuple (documento, nuovo_score, metadata)
        """
        if not documents:
            return []
        
        # Estrai solo i testi
        texts = [doc for doc, _, _ in documents]
        
        # Rerank
        reranked = self.rerank(query, texts, top_k)
        
        # Ricostruisci con nuovi score
        result = []
        for original_idx, new_score in reranked:
            doc, _, metadata = documents[original_idx]
            result.append((doc, float(new_score), metadata))
        
        return result


class VectorStore:
    """
    Vector Store per RAG basato su ChromaDB.
    
    Usa sentence-transformers per generare embedding localmente,
    senza dipendenze da API esterne.
    
    Features:
        - Embedding locali con sentence-transformers
        - Reranking opzionale con CrossEncoder per risultati più accurati
        - Supporto per smart chunking di codice Python
    
    Attributes:
        collection_name: Nome della collezione ChromaDB
        embedding_model: Nome del modello sentence-transformers
        persist_directory: Directory per persistenza (None = in-memory)
        use_reranker: Se abilitare reranking dei risultati
        reranker_model: Modello per reranking
    """
    
    # Modelli consigliati (dal più veloce al più accurato)
    EMBEDDING_MODELS = {
        "fast": "all-MiniLM-L6-v2",           # 384 dim, veloce
        "balanced": "all-mpnet-base-v2",       # 768 dim, bilanciato
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingue
        "italian": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        use_reranker: bool = False,
        reranker_model: str = "fast",
    ):
        """
        Inizializza il VectorStore.
        
        Args:
            collection_name: Nome della collezione
            embedding_model: Modello per embedding (default: all-MiniLM-L6-v2)
            persist_directory: Directory per salvare i dati (None = solo RAM)
            use_reranker: Se abilitare reranking con CrossEncoder
            reranker_model: Modello reranker ("fast", "balanced", "accurate" o nome completo)
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.use_reranker = use_reranker
        
        # Inizializza embedding model
        logger.info(f"Caricamento modello embedding: {embedding_model}")
        SentenceTransformer = _get_sentence_transformers()
        self.embedding_model = SentenceTransformer(embedding_model)
        self._embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._embedding_dim}")
        
        # Inizializza reranker se richiesto
        self.reranker: Optional[Reranker] = None
        if use_reranker:
            self.reranker = Reranker(reranker_model)
        
        # Inizializza ChromaDB
        chromadb = _get_chromadb()
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"ChromaDB persistente in: {persist_directory}")
        else:
            self.client = chromadb.Client()
            logger.info("ChromaDB in-memory (dati persi al riavvio)")
        
        # Crea/ottieni collezione
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Usa cosine similarity
        )
        
        logger.info(
            f"VectorStore inizializzato - Collezione: {collection_name}, "
            f"Documenti: {self.collection.count()}, Reranker: {use_reranker}"
        )
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embedding per una lista di testi.
        
        Args:
            texts: Lista di testi
            
        Returns:
            Lista di embedding (ogni embedding è una lista di float)
        """
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.tolist()
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Aggiunge documenti al vector store.
        
        Args:
            texts: Lista di testi da aggiungere
            metadatas: Metadata opzionali per ogni documento
            ids: ID opzionali (generati automaticamente se non forniti)
            
        Returns:
            Lista degli ID dei documenti aggiunti
        """
        if not texts:
            return []
        
        # Genera ID se non forniti
        if ids is None:
            ids = [
                hashlib.md5(text.encode()).hexdigest()[:12]
                for text in texts
            ]
        
        # Prepara metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Genera embedding
        logger.info(f"Generazione embedding per {len(texts)} documenti...")
        embeddings = self._generate_embeddings(texts)
        
        # Aggiungi a ChromaDB
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.info(f"Aggiunti {len(texts)} documenti. Totale: {self.collection.count()}")
        return ids
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Aggiunge un singolo documento.
        
        Args:
            text: Testo del documento
            metadata: Metadata opzionale
            doc_id: ID opzionale
            
        Returns:
            ID del documento aggiunto
        """
        ids = self.add_documents(
            texts=[text],
            metadatas=[metadata] if metadata else None,
            ids=[doc_id] if doc_id else None,
        )
        return ids[0]
    
    def query(
        self,
        text: str,
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
        use_reranker: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Cerca i documenti più simili a una query.
        
        Se il reranker è abilitato, recupera più risultati iniziali e poi
        li riordina per rilevanza usando CrossEncoder.
        
        Args:
            text: Testo della query
            n_results: Numero di risultati da restituire
            where: Filtro opzionale sui metadata
            use_reranker: Override per usare/non usare reranker (None = usa default)
            rerank_top_k: Quanti risultati iniziali recuperare per reranking
                         (default: n_results * 3)
            
        Returns:
            Lista di tuple (documento, score, metadata)
            Score più alto = più rilevante
        """
        # Determina se usare reranker
        should_rerank = use_reranker if use_reranker is not None else self.use_reranker
        should_rerank = should_rerank and self.reranker is not None
        
        # Se reranking, recupera più risultati iniziali
        if should_rerank:
            initial_n = rerank_top_k or (n_results * 3)
        else:
            initial_n = n_results
        
        # Genera embedding della query
        query_embedding = self._generate_embeddings([text])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_n,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        
        # Formatta risultati
        # ChromaDB restituisce distanze, convertiamo in similarity (1 - distance per cosine)
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                # Cosine distance -> similarity
                similarity = 1 - distance
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                output.append((doc, similarity, metadata))
        
        # Applica reranking se abilitato
        if should_rerank and output and self.reranker:
            output = self.reranker.rerank_with_docs(text, output, top_k=n_results)
        
        return output[:n_results]
    
    def query_with_context(
        self,
        text: str,
        n_results: int = 3,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Cerca documenti e li formatta come contesto per RAG.
        
        Args:
            text: Query
            n_results: Numero di risultati
            separator: Separatore tra documenti
            
        Returns:
            Stringa formattata con i documenti recuperati
        """
        results = self.query(text, n_results=n_results)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, (doc, score, metadata) in enumerate(results, 1):
            source = metadata.get("source", "unknown")
            context_parts.append(f"[{i}] (score: {score:.2f}, source: {source})\n{doc}")
        
        return separator.join(context_parts)
    
    def delete(self, ids: List[str]) -> None:
        """Elimina documenti per ID."""
        self.collection.delete(ids=ids)
        logger.info(f"Eliminati {len(ids)} documenti")
    
    def clear(self) -> None:
        """Elimina tutti i documenti dalla collezione."""
        # ChromaDB non ha un metodo clear, ricreiamo la collezione
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collezione svuotata")
    
    def count(self) -> int:
        """Restituisce il numero di documenti."""
        return self.collection.count()
    
    def get_all(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Restituisce tutti i documenti.
        
        Returns:
            Lista di tuple (id, documento, metadata)
        """
        results = self.collection.get(include=["documents", "metadatas"])
        
        output = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                doc = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                output.append((doc_id, doc, metadata))
        
        return output
    
    def add_code_chunks(
        self,
        chunks: List[Any],  # List[CodeChunk] - avoid circular import
    ) -> List[str]:
        """
        Aggiunge code chunks dal SmartChunker.
        
        Args:
            chunks: Lista di CodeChunk dal modulo smart_chunker
            
        Returns:
            Lista degli ID dei documenti aggiunti
        """
        if not chunks:
            return []
        
        texts = []
        metadatas = []
        
        for chunk in chunks:
            # Usa to_embedding_text se disponibile, altrimenti content
            if hasattr(chunk, "to_embedding_text"):
                texts.append(chunk.to_embedding_text())
            else:
                texts.append(str(chunk.content))
            
            # Costruisci metadata
            metadata = {
                "source": getattr(chunk, "file_path", ""),
                "chunk_type": getattr(chunk, "chunk_type", "unknown"),
                "name": getattr(chunk, "qualified_name", getattr(chunk, "name", "")),
                "start_line": getattr(chunk, "start_line", 0),
                "end_line": getattr(chunk, "end_line", 0),
            }
            
            # Converti enum se necessario
            if hasattr(metadata["chunk_type"], "value"):
                metadata["chunk_type"] = metadata["chunk_type"].value
            
            metadatas.append(metadata)
        
        return self.add_documents(texts, metadatas)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_vector_store(
    persist_path: Optional[str] = "./vector_store",
    model: str = "fast",
    use_reranker: bool = False,
    reranker_model: str = "fast",
) -> VectorStore:
    """
    Factory function per creare un VectorStore con configurazione comune.
    
    Args:
        persist_path: Path per persistenza (None = in-memory)
        model: "fast", "balanced", "multilingual", o nome modello completo
        use_reranker: Se abilitare reranking per risultati più accurati
        reranker_model: "fast", "balanced", "accurate" o nome modello completo
        
    Returns:
        VectorStore configurato
        
    Example:
        ```python
        # VectorStore base
        store = create_vector_store()
        
        # Con reranking per risultati migliori
        store = create_vector_store(use_reranker=True)
        
        # Multilingue con reranker accurato
        store = create_vector_store(
            model="multilingual",
            use_reranker=True,
            reranker_model="accurate",
        )
        ```
    """
    # Risolvi shortcut modello
    embedding_model = VectorStore.EMBEDDING_MODELS.get(model, model)
    
    return VectorStore(
        collection_name="knowledge_base",
        embedding_model=embedding_model,
        persist_directory=persist_path,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
    )


def load_documents_from_file(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Carica e chunka un file di testo.
    
    Args:
        file_path: Path al file
        chunk_size: Dimensione massima chunk (caratteri)
        chunk_overlap: Overlap tra chunk
        
    Returns:
        Lista di chunk di testo
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Chunking semplice per paragrafi/frasi
    chunks = []
    paragraphs = text.split("\n\n")
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

