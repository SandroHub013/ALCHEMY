"""
Smart Chunker per codice sorgente.

Utilizza tree-sitter per splittare codice preservando i confini semantici
(funzioni, classi, metodi) invece di tagliare arbitrariamente per caratteri.

Ispirato a osgrep (https://github.com/Ryandonofrio3/osgrep).

Uso:
    ```python
    from src.memory.smart_chunker import SmartChunker, chunk_python_file
    
    # Chunka un file Python
    chunks = chunk_python_file("path/to/file.py")
    for chunk in chunks:
        print(f"[{chunk.chunk_type}] {chunk.name}: {len(chunk.content)} chars")
    
    # Usa il chunker generico
    chunker = SmartChunker()
    chunks = chunker.chunk_file("path/to/file.py")
    ```
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Lazy import per tree-sitter
_tree_sitter = None
_ts_python = None


class ChunkType(str, Enum):
    """Tipo di chunk estratto dal codice."""
    
    MODULE = "module"           # Intero modulo (fallback)
    CLASS = "class"             # Definizione classe
    FUNCTION = "function"       # Funzione top-level
    METHOD = "method"           # Metodo di classe
    DOCSTRING = "docstring"     # Docstring modulo/classe
    IMPORT = "import"           # Blocco import
    COMMENT = "comment"         # Commento significativo
    OTHER = "other"             # Altro codice


@dataclass
class CodeChunk:
    """
    Rappresenta un chunk di codice estratto.
    
    Attributes:
        content: Il codice sorgente del chunk
        chunk_type: Tipo di chunk (function, class, etc.)
        name: Nome dell'elemento (nome funzione/classe)
        start_line: Linea di inizio (1-indexed)
        end_line: Linea di fine (1-indexed)
        file_path: Path del file sorgente
        parent: Nome del parent (es. classe per un metodo)
        docstring: Docstring estratta se presente
        metadata: Metadati aggiuntivi
    """
    
    content: str
    chunk_type: ChunkType
    name: str = ""
    start_line: int = 0
    end_line: int = 0
    file_path: str = ""
    parent: Optional[str] = None
    docstring: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def qualified_name(self) -> str:
        """Nome qualificato (es. ClassName.method_name)."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name
    
    @property
    def char_count(self) -> int:
        """Numero di caratteri nel chunk."""
        return len(self.content)
    
    @property
    def line_count(self) -> int:
        """Numero di linee nel chunk."""
        return self.content.count("\n") + 1
    
    def to_dict(self) -> dict:
        """Converte il chunk in dizionario per serializzazione."""
        return {
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "file_path": self.file_path,
            "parent": self.parent,
            "docstring": self.docstring,
            "char_count": self.char_count,
            "line_count": self.line_count,
            "metadata": self.metadata,
        }
    
    def to_embedding_text(self) -> str:
        """
        Genera testo ottimizzato per embedding.
        
        Include contesto (tipo, nome, docstring) per migliorare la ricerca semantica.
        """
        parts = []
        
        # Header con contesto
        if self.chunk_type == ChunkType.CLASS:
            parts.append(f"# Class: {self.name}")
        elif self.chunk_type == ChunkType.FUNCTION:
            parts.append(f"# Function: {self.name}")
        elif self.chunk_type == ChunkType.METHOD:
            parts.append(f"# Method: {self.qualified_name}")
        
        # Docstring se presente
        if self.docstring:
            parts.append(f"# Description: {self.docstring[:200]}")
        
        # File path per contesto
        if self.file_path:
            parts.append(f"# File: {Path(self.file_path).name}")
        
        # Contenuto
        parts.append(self.content)
        
        return "\n".join(parts)


def _get_tree_sitter():
    """Lazy import di tree-sitter."""
    global _tree_sitter
    if _tree_sitter is None:
        try:
            import tree_sitter
            _tree_sitter = tree_sitter
        except ImportError:
            raise ImportError(
                "tree-sitter non installato. Installa con: pip install tree-sitter"
            )
    return _tree_sitter


def _get_tree_sitter_python():
    """Lazy import di tree-sitter-python."""
    global _ts_python
    if _ts_python is None:
        try:
            import tree_sitter_python
            _ts_python = tree_sitter_python
        except ImportError:
            raise ImportError(
                "tree-sitter-python non installato. "
                "Installa con: pip install tree-sitter-python"
            )
    return _ts_python


class SmartChunker:
    """
    Chunker intelligente che preserva la struttura semantica del codice.
    
    Usa tree-sitter per parsing AST e estrae chunk basati su:
    - Definizioni di funzioni
    - Definizioni di classi (con metodi interni)
    - Blocchi import
    - Docstring modulo
    
    Attributes:
        max_chunk_size: Dimensione massima chunk (caratteri). Chunk più grandi vengono splittati.
        min_chunk_size: Dimensione minima chunk. Chunk più piccoli vengono aggregati.
        include_imports: Se includere blocchi import come chunk separati.
        include_docstrings: Se estrarre docstring come chunk separati.
    """
    
    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".pyw": "python",
    }
    
    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        include_imports: bool = True,
        include_docstrings: bool = True,
    ):
        """
        Inizializza lo SmartChunker.
        
        Args:
            max_chunk_size: Dimensione massima in caratteri per chunk
            min_chunk_size: Dimensione minima in caratteri per chunk
            include_imports: Includere blocchi import
            include_docstrings: Estrarre docstring separatamente
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_imports = include_imports
        self.include_docstrings = include_docstrings
        
        # Parser cache
        self._parsers: dict[str, object] = {}
    
    def _get_parser(self, language: str):
        """Ottiene o crea un parser per il linguaggio specificato."""
        if language in self._parsers:
            return self._parsers[language]
        
        tree_sitter = _get_tree_sitter()
        
        if language == "python":
            ts_python = _get_tree_sitter_python()
            lang = tree_sitter.Language(ts_python.language())
            parser = tree_sitter.Parser(lang)
            self._parsers[language] = parser
            return parser
        
        raise ValueError(f"Linguaggio non supportato: {language}")
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Rileva il linguaggio dal path del file."""
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext)
    
    def _extract_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """Estrae la docstring da un nodo (funzione/classe)."""
        # In Python, la docstring è il primo statement se è una stringa
        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break
        
        if body is None:
            return None
        
        for child in body.children:
            if child.type == "expression_statement":
                expr = child.children[0] if child.children else None
                if expr and expr.type == "string":
                    docstring = source_bytes[expr.start_byte:expr.end_byte].decode("utf-8")
                    # Rimuovi triple quotes
                    docstring = docstring.strip('"""').strip("'''").strip()
                    return docstring
            elif child.type not in ("comment", "newline"):
                # Non è una docstring se c'è altro codice prima
                break
        
        return None
    
    def _get_node_name(self, node, source_bytes: bytes) -> str:
        """Estrae il nome di un nodo (funzione/classe)."""
        for child in node.children:
            if child.type == "identifier":
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
        return ""
    
    def _node_to_chunk(
        self,
        node,
        source_bytes: bytes,
        chunk_type: ChunkType,
        file_path: str,
        parent: Optional[str] = None,
    ) -> CodeChunk:
        """Converte un nodo tree-sitter in CodeChunk."""
        content = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        name = self._get_node_name(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        
        return CodeChunk(
            content=content,
            chunk_type=chunk_type,
            name=name,
            start_line=node.start_point[0] + 1,  # 1-indexed
            end_line=node.end_point[0] + 1,
            file_path=file_path,
            parent=parent,
            docstring=docstring,
        )
    
    def _split_large_chunk(self, chunk: CodeChunk) -> list[CodeChunk]:
        """
        Splitta un chunk troppo grande in parti più piccole.
        
        Cerca di splittare su linee vuote o dopo statement completi.
        """
        if chunk.char_count <= self.max_chunk_size:
            return [chunk]
        
        content = chunk.content
        chunks = []
        current_start = 0
        current_line = chunk.start_line
        
        # Split su doppi newline (paragrafi/blocchi)
        parts = re.split(r'\n\s*\n', content)
        current_content = ""
        
        for part in parts:
            if len(current_content) + len(part) > self.max_chunk_size and current_content:
                # Salva chunk corrente
                line_count = current_content.count("\n") + 1
                chunks.append(CodeChunk(
                    content=current_content.strip(),
                    chunk_type=chunk.chunk_type,
                    name=f"{chunk.name}_part{len(chunks)+1}",
                    start_line=current_line,
                    end_line=current_line + line_count - 1,
                    file_path=chunk.file_path,
                    parent=chunk.parent,
                    metadata={"is_partial": True, "original_name": chunk.name},
                ))
                current_line += line_count
                current_content = part
            else:
                if current_content:
                    current_content += "\n\n" + part
                else:
                    current_content = part
        
        # Ultimo chunk
        if current_content.strip():
            chunks.append(CodeChunk(
                content=current_content.strip(),
                chunk_type=chunk.chunk_type,
                name=f"{chunk.name}_part{len(chunks)+1}" if chunks else chunk.name,
                start_line=current_line,
                end_line=chunk.end_line,
                file_path=chunk.file_path,
                parent=chunk.parent,
                metadata={"is_partial": bool(chunks), "original_name": chunk.name},
            ))
        
        return chunks
    
    def chunk_python_code(
        self,
        code: str,
        file_path: str = "<string>",
    ) -> list[CodeChunk]:
        """
        Chunka codice Python usando tree-sitter.
        
        Args:
            code: Codice sorgente Python
            file_path: Path del file (per metadata)
            
        Returns:
            Lista di CodeChunk estratti
        """
        parser = self._get_parser("python")
        source_bytes = code.encode("utf-8")
        tree = parser.parse(source_bytes)
        root = tree.root_node
        
        chunks: list[CodeChunk] = []
        import_lines: list[str] = []
        import_start_line = 0
        
        def process_node(node, parent_class: Optional[str] = None):
            nonlocal import_lines, import_start_line
            
            if node.type == "import_statement" or node.type == "import_from_statement":
                # Raccogli import
                if self.include_imports:
                    if not import_lines:
                        import_start_line = node.start_point[0] + 1
                    import_lines.append(
                        source_bytes[node.start_byte:node.end_byte].decode("utf-8")
                    )
                return
            
            if node.type == "function_definition":
                chunk_type = ChunkType.METHOD if parent_class else ChunkType.FUNCTION
                chunk = self._node_to_chunk(
                    node, source_bytes, chunk_type, file_path, parent_class
                )
                chunks.extend(self._split_large_chunk(chunk))
                return
            
            if node.type == "class_definition":
                # Estrai la classe come chunk
                chunk = self._node_to_chunk(
                    node, source_bytes, ChunkType.CLASS, file_path
                )
                class_name = chunk.name
                
                # Se la classe è piccola, tienila intera
                if chunk.char_count <= self.max_chunk_size:
                    chunks.append(chunk)
                else:
                    # Altrimenti estrai i metodi separatamente
                    # Prima aggiungi header classe (fino al primo metodo)
                    class_header = self._extract_class_header(node, source_bytes)
                    if class_header:
                        chunks.append(CodeChunk(
                            content=class_header,
                            chunk_type=ChunkType.CLASS,
                            name=class_name,
                            start_line=chunk.start_line,
                            end_line=chunk.start_line + class_header.count("\n"),
                            file_path=file_path,
                            docstring=chunk.docstring,
                            metadata={"is_header": True},
                        ))
                    
                    # Poi processa i metodi
                    for child in node.children:
                        if child.type == "block":
                            for block_child in child.children:
                                process_node(block_child, class_name)
                return
            
            # Processa figli per altri tipi
            for child in node.children:
                process_node(child, parent_class)
        
        # Processa l'albero
        for child in root.children:
            process_node(child)
        
        # Aggiungi blocco import se raccolto
        if import_lines and self.include_imports:
            import_content = "\n".join(import_lines)
            if len(import_content) >= self.min_chunk_size:
                chunks.insert(0, CodeChunk(
                    content=import_content,
                    chunk_type=ChunkType.IMPORT,
                    name="imports",
                    start_line=import_start_line,
                    end_line=import_start_line + len(import_lines) - 1,
                    file_path=file_path,
                ))
        
        # Filtra chunk troppo piccoli (aggrega con precedente se possibile)
        chunks = self._aggregate_small_chunks(chunks)
        
        return chunks
    
    def _extract_class_header(self, node, source_bytes: bytes) -> str:
        """Estrae l'header di una classe (definizione + docstring, senza metodi)."""
        header_lines = []
        content = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        lines = content.split("\n")
        
        in_docstring = False
        docstring_delimiter = None
        
        for line in lines:
            stripped = line.strip()
            
            # Gestisci docstring multilinea
            if in_docstring:
                header_lines.append(line)
                if docstring_delimiter and docstring_delimiter in stripped:
                    in_docstring = False
                continue
            
            # Inizio docstring
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_delimiter = stripped[:3]
                header_lines.append(line)
                if stripped.count(docstring_delimiter) < 2:
                    in_docstring = True
                continue
            
            # Linea class o decoratori
            if stripped.startswith("class ") or stripped.startswith("@"):
                header_lines.append(line)
                continue
            
            # Attributi di classe (senza def)
            if not stripped.startswith("def ") and not stripped.startswith("async def"):
                # Potrebbe essere un attributo o linea vuota
                if stripped == "" or "=" in stripped or stripped.startswith("#"):
                    header_lines.append(line)
                    continue
            
            # Trovato un metodo, stop
            break
        
        return "\n".join(header_lines)
    
    def _aggregate_small_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Aggrega chunk troppo piccoli con i precedenti."""
        if not chunks:
            return chunks
        
        result = []
        pending: Optional[CodeChunk] = None
        
        for chunk in chunks:
            if chunk.char_count < self.min_chunk_size:
                if pending:
                    # Aggrega con pending
                    pending = CodeChunk(
                        content=pending.content + "\n\n" + chunk.content,
                        chunk_type=pending.chunk_type,
                        name=pending.name,
                        start_line=pending.start_line,
                        end_line=chunk.end_line,
                        file_path=pending.file_path,
                        parent=pending.parent,
                        docstring=pending.docstring,
                        metadata={**pending.metadata, "aggregated": True},
                    )
                else:
                    pending = chunk
            else:
                if pending:
                    # Aggiungi pending se abbastanza grande, altrimenti aggrega
                    if pending.char_count >= self.min_chunk_size:
                        result.append(pending)
                        result.append(chunk)
                    else:
                        # Aggrega pending con questo chunk
                        result.append(CodeChunk(
                            content=pending.content + "\n\n" + chunk.content,
                            chunk_type=chunk.chunk_type,
                            name=chunk.name,
                            start_line=pending.start_line,
                            end_line=chunk.end_line,
                            file_path=chunk.file_path,
                            parent=chunk.parent,
                            docstring=chunk.docstring,
                            metadata={**chunk.metadata, "aggregated": True},
                        ))
                    pending = None
                else:
                    result.append(chunk)
        
        # Gestisci ultimo pending
        if pending:
            if result and pending.char_count < self.min_chunk_size:
                # Aggrega con ultimo
                last = result[-1]
                result[-1] = CodeChunk(
                    content=last.content + "\n\n" + pending.content,
                    chunk_type=last.chunk_type,
                    name=last.name,
                    start_line=last.start_line,
                    end_line=pending.end_line,
                    file_path=last.file_path,
                    parent=last.parent,
                    docstring=last.docstring,
                    metadata={**last.metadata, "aggregated": True},
                )
            else:
                result.append(pending)
        
        return result
    
    def chunk_file(self, file_path: str) -> list[CodeChunk]:
        """
        Chunka un file sorgente.
        
        Args:
            file_path: Path al file da processare
            
        Returns:
            Lista di CodeChunk estratti
            
        Raises:
            ValueError: Se il linguaggio non è supportato
            FileNotFoundError: Se il file non esiste
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")
        
        language = self._detect_language(file_path)
        if language is None:
            raise ValueError(
                f"Linguaggio non supportato per estensione: {path.suffix}\n"
                f"Estensioni supportate: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        code = path.read_text(encoding="utf-8")
        
        if language == "python":
            return self.chunk_python_code(code, str(path))
        
        raise ValueError(f"Linguaggio non implementato: {language}")
    
    def chunk_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[list[str]] = None,
    ) -> list[CodeChunk]:
        """
        Chunka tutti i file supportati in una directory.
        
        Args:
            directory: Directory da processare
            recursive: Se cercare ricorsivamente
            exclude_patterns: Pattern da escludere (es. ["test_*", "__pycache__"])
            
        Returns:
            Lista di tutti i CodeChunk estratti
        """
        exclude_patterns = exclude_patterns or ["__pycache__", ".git", ".venv", "venv"]
        path = Path(directory)
        
        if not path.is_dir():
            raise ValueError(f"Non è una directory: {directory}")
        
        all_chunks = []
        pattern = "**/*" if recursive else "*"
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in path.glob(f"{pattern}{ext}"):
                # Check esclusioni
                skip = False
                for exclude in exclude_patterns:
                    if exclude in str(file_path):
                        skip = True
                        break
                
                if skip:
                    continue
                
                try:
                    chunks = self.chunk_file(str(file_path))
                    all_chunks.extend(chunks)
                    logger.debug(f"Chunkato {file_path}: {len(chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Errore chunking {file_path}: {e}")
        
        logger.info(f"Chunkati {len(all_chunks)} chunks da {directory}")
        return all_chunks


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def chunk_python_file(
    file_path: str,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 100,
) -> list[CodeChunk]:
    """
    Convenience function per chunkare un file Python.
    
    Args:
        file_path: Path al file Python
        max_chunk_size: Dimensione massima chunk
        min_chunk_size: Dimensione minima chunk
        
    Returns:
        Lista di CodeChunk
    """
    chunker = SmartChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
    )
    return chunker.chunk_file(file_path)


def chunk_python_code(
    code: str,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 100,
) -> list[CodeChunk]:
    """
    Convenience function per chunkare codice Python da stringa.
    
    Args:
        code: Codice sorgente Python
        max_chunk_size: Dimensione massima chunk
        min_chunk_size: Dimensione minima chunk
        
    Returns:
        Lista di CodeChunk
    """
    chunker = SmartChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
    )
    return chunker.chunk_python_code(code)


def chunks_to_documents(chunks: list[CodeChunk]) -> list[tuple[str, dict]]:
    """
    Converte CodeChunk in formato compatibile con VectorStore.
    
    Args:
        chunks: Lista di CodeChunk
        
    Returns:
        Lista di tuple (text, metadata) per add_documents
    """
    return [
        (
            chunk.to_embedding_text(),
            {
                "source": chunk.file_path,
                "chunk_type": chunk.chunk_type.value,
                "name": chunk.qualified_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
        )
        for chunk in chunks
    ]

