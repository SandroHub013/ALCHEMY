"""
Smart Chunker for source code.

Uses tree-sitter to split code preserving semantic boundaries
(functions, classes, methods) instead of arbitrarily cutting by characters.

Inspired by osgrep (https://github.com/Ryandonofrio3/osgrep).

Usage:
    ```python
    from src.memory.smart_chunker import SmartChunker, chunk_python_file
    
    # Chunk a Python file
    chunks = chunk_python_file("path/to/file.py")
    for chunk in chunks:
        print(f"[{chunk.chunk_type}] {chunk.name}: {len(chunk.content)} chars")
    
    # Use the generic chunker
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

# Lazy import for tree-sitter
_tree_sitter = None
_ts_python = None


class ChunkType(str, Enum):
    """Type of chunk extracted from code."""
    
    MODULE = "module"           # Entire module (fallback)
    CLASS = "class"             # Class definition
    FUNCTION = "function"       # Top-level function
    METHOD = "method"           # Class method
    DOCSTRING = "docstring"     # Module/class docstring
    IMPORT = "import"           # Import block
    COMMENT = "comment"         # Significant comment
    OTHER = "other"             # Other code


@dataclass
class CodeChunk:
    """
    Represents an extracted code chunk.
    
    Attributes:
        content: The source code of the chunk
        chunk_type: Type of chunk (function, class, etc.)
        name: Name of the element (function/class name)
        start_line: Start line (1-indexed)
        end_line: End line (1-indexed)
        file_path: Path of the source file
        parent: Name of the parent (e.g., class for a method)
        docstring: Extracted docstring if present
        metadata: Additional metadata
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
        """Qualified name (e.g., ClassName.method_name)."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name
    
    @property
    def char_count(self) -> int:
        """Number of characters in the chunk."""
        return len(self.content)
    
    @property
    def line_count(self) -> int:
        """Number of lines in the chunk."""
        return self.content.count("\n") + 1
    
    def to_dict(self) -> dict:
        """Convert the chunk to dictionary for serialization."""
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
        Generate text optimized for embedding.
        
        Includes context (type, name, docstring) to improve semantic search.
        """
        parts = []
        
        # Header with context
        if self.chunk_type == ChunkType.CLASS:
            parts.append(f"# Class: {self.name}")
        elif self.chunk_type == ChunkType.FUNCTION:
            parts.append(f"# Function: {self.name}")
        elif self.chunk_type == ChunkType.METHOD:
            parts.append(f"# Method: {self.qualified_name}")
        
        # Docstring if present
        if self.docstring:
            parts.append(f"# Description: {self.docstring[:200]}")
        
        # File path for context
        if self.file_path:
            parts.append(f"# File: {Path(self.file_path).name}")
        
        # Content
        parts.append(self.content)
        
        return "\n".join(parts)


def _get_tree_sitter():
    """Lazy import of tree-sitter."""
    global _tree_sitter
    if _tree_sitter is None:
        try:
            import tree_sitter
            _tree_sitter = tree_sitter
        except ImportError:
            raise ImportError(
                "tree-sitter not installed. Install with: pip install tree-sitter"
            )
    return _tree_sitter


def _get_tree_sitter_python():
    """Lazy import of tree-sitter-python."""
    global _ts_python
    if _ts_python is None:
        try:
            import tree_sitter_python
            _ts_python = tree_sitter_python
        except ImportError:
            raise ImportError(
                "tree-sitter-python not installed. "
                "Install with: pip install tree-sitter-python"
            )
    return _ts_python


class SmartChunker:
    """
    Intelligent chunker that preserves the semantic structure of code.
    
    Uses tree-sitter for AST parsing and extracts chunks based on:
    - Function definitions
    - Class definitions (with internal methods)
    - Import blocks
    - Module docstring
    
    Attributes:
        max_chunk_size: Maximum chunk size (characters). Larger chunks are split.
        min_chunk_size: Minimum chunk size. Smaller chunks are aggregated.
        include_imports: Whether to include import blocks as separate chunks.
        include_docstrings: Whether to extract docstrings as separate chunks.
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
        Initialize the SmartChunker.
        
        Args:
            max_chunk_size: Maximum size in characters per chunk
            min_chunk_size: Minimum size in characters per chunk
            include_imports: Include import blocks
            include_docstrings: Extract docstrings separately
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_imports = include_imports
        self.include_docstrings = include_docstrings
        
        # Parser cache
        self._parsers: dict[str, object] = {}
    
    def _get_parser(self, language: str):
        """Get or create a parser for the specified language."""
        if language in self._parsers:
            return self._parsers[language]
        
        tree_sitter = _get_tree_sitter()
        
        if language == "python":
            ts_python = _get_tree_sitter_python()
            lang = tree_sitter.Language(ts_python.language())
            parser = tree_sitter.Parser(lang)
            self._parsers[language] = parser
            return parser
        
        raise ValueError(f"Unsupported language: {language}")
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect the language from the file path."""
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext)
    
    def _extract_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """Extract the docstring from a node (function/class)."""
        # In Python, the docstring is the first statement if it's a string
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
                    # Remove triple quotes
                    docstring = docstring.strip('"""').strip("'''").strip()
                    return docstring
            elif child.type not in ("comment", "newline"):
                # Not a docstring if there's other code before
                break
        
        return None
    
    def _get_node_name(self, node, source_bytes: bytes) -> str:
        """Extract the name of a node (function/class)."""
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
        """Convert a tree-sitter node to CodeChunk."""
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
        Split a chunk that's too large into smaller parts.
        
        Tries to split on empty lines or after complete statements.
        """
        if chunk.char_count <= self.max_chunk_size:
            return [chunk]
        
        content = chunk.content
        chunks = []
        current_start = 0
        current_line = chunk.start_line
        
        # Split on double newlines (paragraphs/blocks)
        parts = re.split(r'\n\s*\n', content)
        current_content = ""
        
        for part in parts:
            if len(current_content) + len(part) > self.max_chunk_size and current_content:
                # Save current chunk
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
        
        # Last chunk
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
        Chunk Python code using tree-sitter.
        
        Args:
            code: Python source code
            file_path: File path (for metadata)
            
        Returns:
            List of extracted CodeChunks
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
                # Collect imports
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
                # Extract the class as a chunk
                chunk = self._node_to_chunk(
                    node, source_bytes, ChunkType.CLASS, file_path
                )
                class_name = chunk.name
                
                # If the class is small, keep it whole
                if chunk.char_count <= self.max_chunk_size:
                    chunks.append(chunk)
                else:
                    # Otherwise extract methods separately
                    # First add class header (up to the first method)
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
                    
                    # Then process the methods
                    for child in node.children:
                        if child.type == "block":
                            for block_child in child.children:
                                process_node(block_child, class_name)
                return
            
            # Process children for other types
            for child in node.children:
                process_node(child, parent_class)
        
        # Process the tree
        for child in root.children:
            process_node(child)
        
        # Add import block if collected
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
        
        # Filter chunks that are too small (aggregate with previous if possible)
        chunks = self._aggregate_small_chunks(chunks)
        
        return chunks
    
    def _extract_class_header(self, node, source_bytes: bytes) -> str:
        """Extract the header of a class (definition + docstring, without methods)."""
        header_lines = []
        content = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        lines = content.split("\n")
        
        in_docstring = False
        docstring_delimiter = None
        
        for line in lines:
            stripped = line.strip()
            
            # Handle multiline docstring
            if in_docstring:
                header_lines.append(line)
                if docstring_delimiter and docstring_delimiter in stripped:
                    in_docstring = False
                continue
            
            # Docstring start
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_delimiter = stripped[:3]
                header_lines.append(line)
                if stripped.count(docstring_delimiter) < 2:
                    in_docstring = True
                continue
            
            # Class line or decorators
            if stripped.startswith("class ") or stripped.startswith("@"):
                header_lines.append(line)
                continue
            
            # Class attributes (without def)
            if not stripped.startswith("def ") and not stripped.startswith("async def"):
                # Could be an attribute or empty line
                if stripped == "" or "=" in stripped or stripped.startswith("#"):
                    header_lines.append(line)
                    continue
            
            # Found a method, stop
            break
        
        return "\n".join(header_lines)
    
    def _aggregate_small_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Aggregate chunks that are too small with the previous ones."""
        if not chunks:
            return chunks
        
        result = []
        pending: Optional[CodeChunk] = None
        
        for chunk in chunks:
            if chunk.char_count < self.min_chunk_size:
                if pending:
                    # Aggregate with pending
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
                    # Add pending if large enough, otherwise aggregate
                    if pending.char_count >= self.min_chunk_size:
                        result.append(pending)
                        result.append(chunk)
                    else:
                        # Aggregate pending with this chunk
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
        
        # Handle last pending
        if pending:
            if result and pending.char_count < self.min_chunk_size:
                # Aggregate with last
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
        Chunk a source file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of extracted CodeChunks
            
        Raises:
            ValueError: If the language is not supported
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        language = self._detect_language(file_path)
        if language is None:
            raise ValueError(
                f"Unsupported language for extension: {path.suffix}\n"
                f"Supported extensions: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        code = path.read_text(encoding="utf-8")
        
        if language == "python":
            return self.chunk_python_code(code, str(path))
        
        raise ValueError(f"Language not implemented: {language}")
    
    def chunk_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[list[str]] = None,
    ) -> list[CodeChunk]:
        """
        Chunk all supported files in a directory.
        
        Args:
            directory: Directory to process
            recursive: Whether to search recursively
            exclude_patterns: Patterns to exclude (e.g., ["test_*", "__pycache__"])
            
        Returns:
            List of all extracted CodeChunks
        """
        exclude_patterns = exclude_patterns or ["__pycache__", ".git", ".venv", "venv"]
        path = Path(directory)
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        all_chunks = []
        pattern = "**/*" if recursive else "*"
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in path.glob(f"{pattern}{ext}"):
                # Check exclusions
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
                    logger.debug(f"Chunked {file_path}: {len(chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Error chunking {file_path}: {e}")
        
        logger.info(f"Chunked {len(all_chunks)} chunks from {directory}")
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
    Convenience function to chunk a Python file.
    
    Args:
        file_path: Path to the Python file
        max_chunk_size: Maximum chunk size
        min_chunk_size: Minimum chunk size
        
    Returns:
        List of CodeChunks
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
    Convenience function to chunk Python code from a string.
    
    Args:
        code: Python source code
        max_chunk_size: Maximum chunk size
        min_chunk_size: Minimum chunk size
        
    Returns:
        List of CodeChunks
    """
    chunker = SmartChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
    )
    return chunker.chunk_python_code(code)


def chunks_to_documents(chunks: list[CodeChunk]) -> list[tuple[str, dict]]:
    """
    Convert CodeChunks to format compatible with VectorStore.
    
    Args:
        chunks: List of CodeChunks
        
    Returns:
        List of tuples (text, metadata) for add_documents
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
