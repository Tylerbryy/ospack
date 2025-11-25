"""Core packing logic - combines imports and semantic search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from .indexer import get_indexer
from .log import get_logger
from .resolver import get_resolver


class Verbosity(str, Enum):
    """Output verbosity levels."""

    QUIET = "quiet"  # Minimal: paths + scores only
    NORMAL = "normal"  # Default: code + metadata
    VERBOSE = "verbose"  # Full: code + all metadata + debug info


logger = get_logger(__name__)

# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4

# Score type definitions and their typical ranges
SCORE_TYPES = {
    "rerank": {"min": -15.0, "max": 5.0, "default_threshold": -10.0},
    "rrf": {"min": 0.0, "max": 0.1, "default_threshold": 0.001},
    "dense": {"min": 0.0, "max": 1.0, "default_threshold": 0.3},
}


def extract_score(result: dict) -> tuple[float, str]:
    """Extract score and score_type from a search result.

    Returns:
        Tuple of (score_value, score_type) where score_type is one of:
        "rerank", "rrf", or "dense"
    """
    if result.get("rerank_score") is not None:
        return result["rerank_score"], "rerank"
    elif result.get("rrf_score") is not None:
        return result["rrf_score"], "rrf"
    return result.get("score", 0), "dense"


@dataclass
class TruncationInfo:
    """Information about result truncation for agent guidance."""

    truncated: bool
    total_available: int
    returned: int
    reason: str | None = None  # "token_budget", "max_files", "max_chunks"
    suggestion: str | None = None


@dataclass
class PaginationInfo:
    """Information about pagination for large result sets."""

    offset: int
    limit: int
    total: int
    has_more: bool

    @property
    def next_offset(self) -> int | None:
        """Get the offset for the next page, or None if no more results."""
        if self.has_more:
            return self.offset + self.limit
        return None


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
    score_type: str = "dense"  # "rerank", "rrf", or "dense"


@dataclass
class PackedFile:
    """A file included in the pack."""

    path: Path
    content: str
    reason: str  # Why this file was included
    score: float = 1.0  # Relevance score (1.0 for imports, semantic score for search)
    score_type: str = "dense"  # "rerank", "rrf", or "dense"


@dataclass
class PackResult:
    """Result of packing operation."""

    files: list[PackedFile] = field(default_factory=list)
    chunks: list[PackedChunk] = field(default_factory=list)
    total_lines: int = 0
    total_chars: int = 0
    total_tokens: int = 0
    truncation: TruncationInfo | None = None
    pagination: PaginationInfo | None = None
    score_type: str = "dense"  # Primary score type used in results


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
        min_score: float | None = None,
        depth: int = 2,
        rerank: bool = True,
        hybrid: bool = True,
        chunk_mode: bool = False,
        offset: int = 0,
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
            offset: Skip first N results (for pagination)

        Returns:
            PackResult with included files/chunks
        """
        result = PackResult()
        included_paths: set[Path] = set()
        included_chunks: set[str] = set()  # Track chunk IDs

        # Track truncation reasons
        truncation_reason: str | None = None
        total_available = 0

        # Determine score type based on search mode
        if rerank:
            score_type = "rerank"
        elif hybrid:
            score_type = "rrf"
        else:
            score_type = "dense"
        result.score_type = score_type

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
                deps = graph.get(focus_path, [])
                total_available += len(deps) + 1  # +1 for focus file

                for dep_path in deps:
                    if len(result.files) >= max_files:
                        truncation_reason = "max_files"
                        break
                    if not _within_budget():
                        truncation_reason = "token_budget"
                        break
                    self._add_file(result, dep_path, "import", included_paths)
                    if result.files:
                        _add_to_token_count(result.files[-1].content)

        # 2. If query specified, do semantic search
        if query:
            # Build index if needed
            self.indexer.build_index()

            # Search for relevant chunks (hybrid + rerank)
            # Fetch extra to account for offset
            limit = max_chunks if chunk_mode else max_files
            search_limit = (offset + limit) * 2
            search_results = self.indexer.search(
                query,
                limit=search_limit,
                rerank=rerank,
                hybrid=hybrid,
            )

            total_search_results = len(search_results)
            total_available += total_search_results

            # Apply offset (skip first N results)
            if offset > 0:
                search_results = search_results[offset:]

            for sr in search_results:
                score, item_score_type = extract_score(sr)

                # Check score threshold (only if min_score is set)
                if min_score is not None and score < min_score:
                    continue

                # Check budget (always allow first result)
                is_first = len(result.chunks) == 0 and len(result.files) == 0
                if not _within_budget(is_first=is_first):
                    truncation_reason = "token_budget"
                    break

                if chunk_mode:
                    # Chunk mode: return individual chunks
                    chunk_id = sr.get("id", f"{sr['file_path']}:{sr['start_line']}")
                    if chunk_id in included_chunks:
                        continue
                    if len(result.chunks) >= max_chunks:
                        truncation_reason = "max_chunks"
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
                        score_type=item_score_type,
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
                        truncation_reason = "max_files"
                        break

                    self._add_file(
                        result,
                        file_path,
                        f"semantic match: {sr.get('name', 'chunk')}",
                        included_paths,
                        score=score,
                        score_type=item_score_type,
                    )
                    if result.files:
                        _add_to_token_count(result.files[-1].content)

        # Build truncation info
        returned = len(result.chunks) if chunk_mode else len(result.files)
        if truncation_reason or returned < total_available:
            suggestion = None
            if truncation_reason == "token_budget":
                suggestion = f"Use --max-tokens {(max_tokens or 4000) * 2} for more results"
            elif truncation_reason == "max_files":
                suggestion = f"Use --max-files {max_files * 2} for more results"
            elif truncation_reason == "max_chunks":
                suggestion = f"Use --max-chunks {max_chunks * 2} for more results"

            result.truncation = TruncationInfo(
                truncated=truncation_reason is not None,
                total_available=total_available,
                returned=returned,
                reason=truncation_reason,
                suggestion=suggestion,
            )

        # Build pagination info (only for query mode)
        if query:
            limit = max_chunks if chunk_mode else max_files
            has_more = (offset + returned) < total_available
            result.pagination = PaginationInfo(
                offset=offset,
                limit=limit,
                total=total_available,
                has_more=has_more,
            )

        return result

    def _add_file(
        self,
        result: PackResult,
        file_path: Path,
        reason: str,
        included: set[Path],
        score: float = 1.0,
        score_type: str = "dense",
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
                    score_type=score_type,
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
    verbosity: Verbosity = Verbosity.NORMAL,
) -> str:
    """Format pack result for output.

    Verbosity levels:
    - QUIET: Minimal output (paths + scores only, no code)
    - NORMAL: Default (code + metadata)
    - VERBOSE: Full output (code + all metadata + debug info)
    """
    if verbosity == Verbosity.QUIET:
        return _format_quiet(result, root_dir)
    elif format == "xml":
        return _format_xml(result, root_dir, verbosity)
    elif format == "chunks":
        return _format_chunks(result, root_dir, verbosity)
    else:
        return _format_compact(result, root_dir, verbosity)


def _format_quiet(result: PackResult, root_dir: Path | None) -> str:
    """Format as minimal output - paths and scores only."""
    lines = []

    # Files
    if result.files:
        lines.append("# Files")
        for pf in result.files:
            rel_path = pf.path.relative_to(root_dir) if root_dir else pf.path
            score_info = f" ({pf.score:.2f})" if pf.score != 1.0 else ""
            lines.append(f"- {rel_path}{score_info}")

    # Chunks
    if result.chunks:
        lines.append("# Chunks")
        for chunk in result.chunks:
            rel_path = chunk.path.relative_to(root_dir) if root_dir else chunk.path
            name_info = f" [{chunk.name}]" if chunk.name else ""
            lines.append(
                f"- {rel_path}:{chunk.start_line}-{chunk.end_line}"
                f"{name_info} ({chunk.score:.2f})"
            )

    # Summary line
    count = len(result.files) or len(result.chunks)
    kind = "files" if result.files else "chunks"
    summary = f"# {count} {kind}, ~{result.total_tokens} tokens"
    if result.pagination and result.pagination.has_more:
        summary += f" (more: --offset {result.pagination.next_offset})"
    lines.append(summary)

    return "\n".join(lines)


def _format_xml(
    result: PackResult, root_dir: Path | None, verbosity: Verbosity = Verbosity.NORMAL
) -> str:
    """Format as XML (Claude-friendly)."""
    lines = ["<context>"]

    # Add metadata (always include in normal/verbose)
    meta_attrs = (
        f'files="{len(result.files)}" lines="{result.total_lines}" '
        f'tokens="~{result.total_tokens}" score_type="{result.score_type}"'
    )
    if verbosity == Verbosity.VERBOSE and result.pagination:
        meta_attrs += f' offset="{result.pagination.offset}" total="{result.pagination.total}"'
    lines.append(f"  <meta {meta_attrs}/>")

    for pf in result.files:
        rel_path = pf.path.relative_to(root_dir) if root_dir else pf.path
        file_attrs = f'path="{rel_path}"'
        if verbosity == Verbosity.VERBOSE:
            file_attrs += (
                f' reason="{pf.reason}" score="{pf.score:.2f}" score_type="{pf.score_type}"'
            )
        elif pf.score != 1.0:
            file_attrs += f' score="{pf.score:.2f}"'
        lines.append(f"  <file {file_attrs}>")
        # Escape XML special chars
        content = pf.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(content)
        lines.append("  </file>")

    lines.append("</context>")

    # Add truncation info as comment
    summary = (
        f"<!-- {len(result.files)} files, {result.total_lines} lines, ~{result.total_tokens} tokens"
    )
    if result.truncation and result.truncation.truncated:
        summary += f" | TRUNCATED: {result.truncation.reason}"
        summary += f" ({result.truncation.returned}/{result.truncation.total_available} shown)"
        if result.truncation.suggestion:
            summary += f" | {result.truncation.suggestion}"
    if result.pagination and result.pagination.has_more:
        summary += f" | next_offset={result.pagination.next_offset}"
    summary += " -->"
    lines.append(f"\n{summary}")
    return "\n".join(lines)


def _format_compact(
    result: PackResult, root_dir: Path | None, verbosity: Verbosity = Verbosity.NORMAL
) -> str:
    """Format as compact markdown."""
    lines = []

    for pf in result.files:
        rel_path = pf.path.relative_to(root_dir) if root_dir else pf.path
        ext = pf.path.suffix.lstrip(".")
        if verbosity == Verbosity.VERBOSE:
            score_info = f" (score: {pf.score:.2f}, {pf.score_type}, {pf.reason})"
        elif pf.score != 1.0:
            score_info = f" ({pf.score:.2f})"
        else:
            score_info = ""
        lines.append(f"## {rel_path}{score_info}")
        lines.append(f"```{ext}")
        lines.append(pf.content)
        lines.append("```")
        lines.append("")

    # Summary with truncation info
    summary = (
        f"<!-- {len(result.files)} files, {result.total_lines} lines, ~{result.total_tokens} tokens"
    )
    if result.truncation and result.truncation.truncated:
        summary += f" | TRUNCATED: {result.truncation.reason}"
        summary += f" ({result.truncation.returned}/{result.truncation.total_available} shown)"
        if result.truncation.suggestion:
            summary += f" | {result.truncation.suggestion}"
    if result.pagination and result.pagination.has_more:
        summary += f" | next_offset={result.pagination.next_offset}"
    summary += " -->"
    lines.append(summary)
    return "\n".join(lines)


def _format_chunks(
    result: PackResult, root_dir: Path | None, verbosity: Verbosity = Verbosity.NORMAL
) -> str:
    """Format as individual chunks with line numbers - optimized for agent context."""
    lines = ["<context>"]

    # Add metadata
    meta_attrs = (
        f'chunks="{len(result.chunks)}" lines="{result.total_lines}" '
        f'tokens="~{result.total_tokens}" score_type="{result.score_type}"'
    )
    if verbosity == Verbosity.VERBOSE and result.pagination:
        meta_attrs += f' offset="{result.pagination.offset}" total="{result.pagination.total}"'
    lines.append(f"  <meta {meta_attrs}/>")

    for chunk in result.chunks:
        rel_path = chunk.path.relative_to(root_dir) if root_dir else chunk.path
        ext = chunk.path.suffix.lstrip(".")
        name_attr = f' name="{chunk.name}"' if chunk.name else ""

        chunk_attrs = f'path="{rel_path}" lines="{chunk.start_line}-{chunk.end_line}"{name_attr}'
        if verbosity == Verbosity.VERBOSE:
            chunk_attrs += f' score="{chunk.score:.2f}" score_type="{chunk.score_type}"'
        else:
            chunk_attrs += f' score="{chunk.score:.2f}"'

        lines.append(f"  <chunk {chunk_attrs}>")
        lines.append(f"```{ext}")
        lines.append(chunk.content)
        lines.append("```")
        lines.append("  </chunk>")

    lines.append("</context>")

    # Summary with truncation info
    summary = (
        f"<!-- {len(result.chunks)} chunks, {result.total_lines} lines, "
        f"~{result.total_tokens} tokens"
    )
    if result.truncation and result.truncation.truncated:
        summary += f" | TRUNCATED: {result.truncation.reason}"
        summary += f" ({result.truncation.returned}/{result.truncation.total_available} shown)"
        if result.truncation.suggestion:
            summary += f" | {result.truncation.suggestion}"
    if result.pagination and result.pagination.has_more:
        summary += f" | next_offset={result.pagination.next_offset}"
    summary += " -->"
    lines.append(f"\n{summary}")
    return "\n".join(lines)
