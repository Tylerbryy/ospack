"""Core packing logic - combines imports and semantic search."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .indexer import get_indexer
from .log import get_logger
from .resolver import get_resolver

logger = get_logger(__name__)

# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4


@dataclass
class PackedChunk:
    """A code chunk included in the pack."""

    path: Path
    content: str
    start_line: int
    end_line: int
    name: str  # Function/class name
    reason: str
    score: float = 1.0


@dataclass
class PackedFile:
    """A file included in the pack."""

    path: Path
    content: str
    reason: str  # Why this file was included
    score: float = 1.0  # Relevance score (1.0 for imports, semantic score for search)


@dataclass
class PackResult:
    """Result of packing operation."""

    files: list[PackedFile] = field(default_factory=list)
    chunks: list[PackedChunk] = field(default_factory=list)
    total_lines: int = 0
    total_chars: int = 0
    total_tokens: int = 0


class Packer:
    """Packs relevant code context."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.indexer = get_indexer(str(self.root_dir))
        self.resolver = get_resolver(str(self.root_dir))

    def pack(
        self,
        focus: str | None = None,
        query: str | None = None,
        max_files: int = 10,
        max_chunks: int = 20,
        max_tokens: int | None = None,
        min_score: float = 0.0,
        depth: int = 2,
        rerank: bool = True,
        hybrid: bool = True,
        chunk_mode: bool = False,
    ) -> PackResult:
        """
        Pack relevant code context.

        Args:
            focus: Entry file for import resolution
            query: Semantic search query
            max_files: Maximum number of files to include
            max_chunks: Maximum chunks when in chunk_mode
            max_tokens: Token budget (approximate)
            min_score: Minimum relevance score to include
            depth: Depth for import resolution
            rerank: Use cross-encoder reranking for better quality
            hybrid: Use hybrid BM25+dense search
            chunk_mode: Return chunks instead of full files

        Returns:
            PackResult with included files/chunks
        """
        result = PackResult()
        included_paths: set[Path] = set()
        included_chunks: set[str] = set()  # Track chunk IDs

        def _within_budget(is_first: bool = False) -> bool:
            """Check if we're within token budget. Always allow at least one result."""
            if max_tokens is None:
                return True
            if is_first:
                return True  # Always allow at least one result
            return result.total_tokens < max_tokens

        def _add_to_token_count(content: str) -> int:
            tokens = len(content) // CHARS_PER_TOKEN
            result.total_tokens += tokens
            return tokens

        # 1. If focus file specified, resolve imports
        if focus:
            focus_path = Path(focus)
            if not focus_path.is_absolute():
                focus_path = self.root_dir / focus_path
            focus_path = focus_path.resolve()

            if focus_path.exists():
                # Add focus file first
                self._add_file(result, focus_path, "focus file", included_paths)
                _add_to_token_count(result.files[-1].content if result.files else "")

                # Resolve imports
                graph = self.resolver.get_dependency_graph(focus_path, max_depth=depth)
                for dep_path in graph.get(focus_path, []):
                    if len(result.files) >= max_files or not _within_budget():
                        break
                    self._add_file(result, dep_path, "import", included_paths)
                    if result.files:
                        _add_to_token_count(result.files[-1].content)

        # 2. If query specified, do semantic search
        if query:
            # Build index if needed
            self.indexer.build_index()

            # Search for relevant chunks (hybrid + rerank)
            search_limit = max_chunks * 2 if chunk_mode else max_files * 2
            search_results = self.indexer.search(
                query,
                limit=search_limit,
                rerank=rerank,
                hybrid=hybrid,
            )

            for sr in search_results:
                # Check score threshold
                score = sr.get("rerank_score") or sr.get("rrf_score") or sr.get("score", 0)
                if score < min_score:
                    continue

                # Check budget (always allow first result)
                is_first = len(result.chunks) == 0 and len(result.files) == 0
                if not _within_budget(is_first=is_first):
                    break

                if chunk_mode:
                    # Chunk mode: return individual chunks
                    chunk_id = sr.get("id", f"{sr['file_path']}:{sr['start_line']}")
                    if chunk_id in included_chunks:
                        continue
                    if len(result.chunks) >= max_chunks:
                        break

                    included_chunks.add(chunk_id)
                    chunk = PackedChunk(
                        path=Path(sr["file_path"]),
                        content=sr["content"],
                        start_line=sr["start_line"],
                        end_line=sr["end_line"],
                        name=sr.get("name", ""),
                        reason=f"semantic match (score: {score:.2f})",
                        score=score,
                    )
                    result.chunks.append(chunk)
                    result.total_lines += sr["end_line"] - sr["start_line"] + 1
                    result.total_chars += len(sr["content"])
                    _add_to_token_count(sr["content"])
                else:
                    # File mode: expand to full files
                    file_path = Path(sr["file_path"])
                    if file_path in included_paths:
                        continue
                    if len(result.files) >= max_files:
                        break

                    self._add_file(
                        result,
                        file_path,
                        f"semantic match: {sr.get('name', 'chunk')}",
                        included_paths,
                        score=score,
                    )
                    if result.files:
                        _add_to_token_count(result.files[-1].content)

        return result

    def _add_file(
        self,
        result: PackResult,
        file_path: Path,
        reason: str,
        included: set[Path],
        score: float = 1.0,
    ):
        """Add a file to the pack result."""
        if file_path in included:
            return
        if not file_path.exists():
            return

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            included.add(file_path)

            result.files.append(
                PackedFile(
                    path=file_path,
                    content=content,
                    reason=reason,
                    score=score,
                )
            )
            result.total_lines += len(content.splitlines())
            result.total_chars += len(content)

        except Exception:
            logger.warning("Could not read %s", file_path, exc_info=True)


def format_output(
    result: PackResult,
    format: Literal["xml", "compact", "chunks"] = "xml",
    root_dir: Path | None = None,
) -> str:
    """Format pack result for output."""
    if format == "xml":
        return _format_xml(result, root_dir)
    elif format == "chunks":
        return _format_chunks(result, root_dir)
    else:
        return _format_compact(result, root_dir)


def _format_xml(result: PackResult, root_dir: Path | None) -> str:
    """Format as XML (Claude-friendly)."""
    lines = ["<context>"]

    for pf in result.files:
        rel_path = pf.path.relative_to(root_dir) if root_dir else pf.path
        lines.append(f'  <file path="{rel_path}" reason="{pf.reason}">')
        lines.append("    <content>")
        # Escape XML special chars
        content = pf.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(content)
        lines.append("    </content>")
        lines.append("  </file>")

    lines.append("</context>")
    lines.append(f"\n<!-- {len(result.files)} files, {result.total_lines} lines -->")
    return "\n".join(lines)


def _format_compact(result: PackResult, root_dir: Path | None) -> str:
    """Format as compact markdown."""
    lines = []

    for pf in result.files:
        rel_path = pf.path.relative_to(root_dir) if root_dir else pf.path
        ext = pf.path.suffix.lstrip(".")
        lines.append(f"## {rel_path}")
        lines.append(f"```{ext}")
        lines.append(pf.content)
        lines.append("```")
        lines.append("")

    lines.append(f"<!-- {len(result.files)} files, {result.total_lines} lines -->")
    return "\n".join(lines)


def _format_chunks(result: PackResult, root_dir: Path | None) -> str:
    """Format as individual chunks with line numbers - optimized for agent context."""
    lines = ["<context>"]

    for chunk in result.chunks:
        rel_path = chunk.path.relative_to(root_dir) if root_dir else chunk.path
        ext = chunk.path.suffix.lstrip(".")
        name_attr = f' name="{chunk.name}"' if chunk.name else ""

        lines.append(
            f'  <chunk path="{rel_path}" lines="{chunk.start_line}-{chunk.end_line}"'
            f'{name_attr} score="{chunk.score:.2f}">'
        )
        lines.append(f"```{ext}")
        lines.append(chunk.content)
        lines.append("```")
        lines.append("  </chunk>")

    lines.append("</context>")
    lines.append(
        f"\n<!-- {len(result.chunks)} chunks, {result.total_lines} lines, "
        f"~{result.total_tokens} tokens -->"
    )
    return "\n".join(lines)
