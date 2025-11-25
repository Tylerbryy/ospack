"""Code chunking using langchain-text-splitters.

Uses RecursiveCharacterTextSplitter with language-specific separators
for 26+ programming languages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .log import get_logger

logger = get_logger(__name__)


# Map file extensions to langchain Language enum
EXT_TO_LANGUAGE: dict[str, Language] = {
    # Python
    ".py": Language.PYTHON,
    ".pyw": Language.PYTHON,
    ".pyi": Language.PYTHON,
    # JavaScript/TypeScript
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    # Systems languages
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".hxx": Language.CPP,
    # JVM languages
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    # Other languages
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".swift": Language.SWIFT,
    ".cs": Language.CSHARP,
    ".lua": Language.LUA,
    ".hs": Language.HASKELL,
    ".pl": Language.PERL,
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    # Markup/Config
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".tex": Language.LATEX,
    ".rst": Language.RST,
    ".sol": Language.SOL,
    ".proto": Language.PROTO,
    # Scripts
    ".ps1": Language.POWERSHELL,
    ".vb": Language.VISUALBASIC6,
    ".cob": Language.COBOL,
}


@dataclass
class Chunk:
    """A semantic chunk of code."""
    content: str
    start_line: int
    end_line: int
    type: Literal["function", "class", "block", "other"]
    context: list[str] = field(default_factory=list)
    chunk_index: int = -1
    is_anchor: bool = False

    @property
    def name(self) -> str | None:
        """Extract name from context if available."""
        for ctx in reversed(self.context):
            if ctx.startswith(("Function:", "Class:", "Method:", "Symbol:")):
                return ctx.split(":", 1)[1].strip()
        return None

    @property
    def node_type(self) -> str:
        """Return type as string for compatibility."""
        return self.type


class LangChainChunker:
    """Chunker using langchain-text-splitters.

    Uses RecursiveCharacterTextSplitter with language-specific separators
    for known languages, falls back to generic splitting for unknown.
    """

    # Chunk size tuned for embedding models (Jina handles 8192 tokens)
    CHUNK_SIZE = 2000  # chars (~500 tokens)
    CHUNK_OVERLAP = 200

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        """Chunk a file into semantic units."""
        if not content.strip():
            return []

        ext = Path(file_path).suffix.lower()
        lang = EXT_TO_LANGUAGE.get(ext)

        try:
            if lang:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang,
                    chunk_size=self.CHUNK_SIZE,
                    chunk_overlap=self.CHUNK_OVERLAP,
                )
            else:
                # Unknown language - use default separators
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.CHUNK_SIZE,
                    chunk_overlap=self.CHUNK_OVERLAP,
                )
        except ValueError:
            # Language not supported by langchain
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP,
            )

        # Split the content
        docs = splitter.create_documents([content])

        # Convert to Chunk objects
        return self._docs_to_chunks(docs, file_path, content)

    def _docs_to_chunks(
        self, docs: list, file_path: str, original_content: str
    ) -> list[Chunk]:
        """Convert langchain Documents to Chunk objects."""
        chunks: list[Chunk] = []
        lines = original_content.split("\n")

        for doc in docs:
            chunk_text = doc.page_content

            # Find line numbers by locating chunk in original content
            start_line, end_line = self._find_line_numbers(
                chunk_text, original_content, lines
            )

            chunks.append(Chunk(
                content=chunk_text,
                start_line=start_line,
                end_line=end_line,
                type="block",
                context=[f"File: {file_path}"],
            ))

        return chunks

    def _find_line_numbers(
        self, chunk_text: str, original: str, lines: list[str]
    ) -> tuple[int, int]:
        """Find the line numbers for a chunk in the original content."""
        # Find the chunk's position in the original
        try:
            start_pos = original.find(chunk_text)
            if start_pos == -1:
                # Chunk might have been modified, try first line
                first_line = chunk_text.split("\n")[0]
                start_pos = original.find(first_line)
                if start_pos == -1:
                    return 0, len(lines)

            # Count newlines before start to get start line
            start_line = original[:start_pos].count("\n")
            end_line = start_line + chunk_text.count("\n") + 1

            return start_line, min(end_line, len(lines))
        except Exception:
            return 0, len(lines)


# --- Compatibility Layer ---

class Chunker(LangChainChunker):
    """Alias for backward compatibility."""
    pass


# Global singleton
_chunker: Chunker | None = None


def get_chunker() -> Chunker:
    """Get or create the global chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = Chunker()
    return _chunker
