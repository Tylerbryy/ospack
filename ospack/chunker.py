"""Tree-sitter based code chunking."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_sitter_languages import get_parser

from .log import get_logger

logger = get_logger(__name__)

# Map file extensions to tree-sitter language names
EXTENSION_TO_LANGUAGE = {
    # JavaScript/TypeScript
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    # Python
    "py": "python",
    # Systems languages
    "rs": "rust",
    "go": "go",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "hpp": "cpp",
    "cc": "cpp",
    # JVM
    "java": "java",
    "kt": "kotlin",
    "scala": "scala",
    # Other
    "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "cs": "c_sharp",
}

# Node types to extract as chunks (language-agnostic where possible)
CHUNK_NODE_TYPES = {
    # Functions
    "function_declaration",
    "function_definition",
    "method_definition",
    "method_declaration",
    "function_item",  # Rust
    "func_literal",  # Go
    # Classes
    "class_declaration",
    "class_definition",
    "impl_item",  # Rust
    "struct_item",  # Rust
    "interface_declaration",
    "type_alias_declaration",
    # Modules
    "module",
    "mod_item",  # Rust
}


@dataclass
class Chunk:
    """A semantic chunk of code."""

    content: str
    start_line: int
    end_line: int
    node_type: str
    name: str | None = None


class Chunker:
    """Tree-sitter based code chunker."""

    def __init__(self, max_chunk_lines: int = 100, min_chunk_lines: int = 3):
        self.max_chunk_lines = max_chunk_lines
        self.min_chunk_lines = min_chunk_lines
        self._parsers: dict[str, Any] = {}

    def _create_file_chunk(self, file_path: str, content: str) -> Chunk:
        """Create a fallback chunk representing the entire file."""
        lines = content.splitlines()
        return Chunk(
            content=content,
            start_line=1,
            end_line=len(lines) or 1,
            node_type="file",
            name=Path(file_path).name,
        )

    def _get_parser(self, language: str):
        """Get or create a parser for the given language."""
        if language not in self._parsers:
            self._parsers[language] = get_parser(language)
        return self._parsers[language]

    def _get_name(self, node) -> str | None:
        """Extract the name from a node if possible."""
        # Look for name/identifier child nodes
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier"):
                return child.text.decode("utf-8")
            # For function declarations, look for nested identifier
            if child.type in ("function_declarator", "declarator"):
                for subchild in child.children:
                    if subchild.type == "identifier":
                        return subchild.text.decode("utf-8")
        return None

    def _extract_chunks(self, node, source_bytes: bytes, depth: int = 0) -> Iterator[Chunk]:
        """Recursively extract chunks from AST nodes."""
        # Check if this node is a chunk boundary
        if node.type in CHUNK_NODE_TYPES:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            line_count = end_line - start_line + 1

            # If chunk is within size limits, yield it
            if self.min_chunk_lines <= line_count <= self.max_chunk_lines:
                content = source_bytes[node.start_byte : node.end_byte].decode("utf-8")
                yield Chunk(
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    node_type=node.type,
                    name=self._get_name(node),
                )
                return  # Don't recurse into children

            # If too large, recurse into children
            if line_count > self.max_chunk_lines:
                for child in node.children:
                    yield from self._extract_chunks(child, source_bytes, depth + 1)
                return

        # Not a chunk boundary, recurse into children
        for child in node.children:
            yield from self._extract_chunks(child, source_bytes, depth + 1)

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        """Chunk a file into semantic units."""
        ext = Path(file_path).suffix.lstrip(".")
        language = EXTENSION_TO_LANGUAGE.get(ext)

        if not language:
            # Fallback: treat entire file as one chunk for unknown languages
            return [self._create_file_chunk(file_path, content)]

        try:
            parser = self._get_parser(language)
            source_bytes = content.encode("utf-8")
            tree = parser.parse(source_bytes)

            chunks = list(self._extract_chunks(tree.root_node, source_bytes))

            # If no semantic chunks found, fall back to file chunk
            if not chunks:
                return [self._create_file_chunk(file_path, content)]

            return chunks

        except Exception:
            # Fallback on parse errors
            logger.debug("Parse error for %s, using file chunk", file_path, exc_info=True)
            return [self._create_file_chunk(file_path, content)]


# Global singleton
_chunker: Chunker | None = None


def get_chunker() -> Chunker:
    """Get or create the global chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = Chunker()
    return _chunker
