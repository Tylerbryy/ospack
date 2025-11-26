"""Core packing logic - combines imports and semantic search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import tiktoken

from .indexer import get_indexer
from .log import get_logger
from .resolver import get_resolver

if TYPE_CHECKING:
    from tree_sitter import Node


class Verbosity(str, Enum):
    """Output verbosity levels."""

    QUIET = "quiet"  # Minimal: paths + scores only
    NORMAL = "normal"  # Default: code + metadata
    VERBOSE = "verbose"  # Full: code + all metadata + debug info


logger = get_logger(__name__)

# Tiktoken encoder for accurate token counting (cl100k_base = GPT-4/Claude tokenizer)
_encoder: tiktoken.Encoding | None = None

OVERHEAD_PER_FILE = 50  # XML tags, newlines, etc.

# Score type definitions and their typical ranges
SCORE_TYPES = {
    "rerank": {"min": -15.0, "max": 5.0, "default_threshold": -10.0},
    "rrf": {"min": 0.0, "max": 0.1, "default_threshold": 0.001},
    "dense": {"min": 0.0, "max": 1.0, "default_threshold": 0.3},
}

# Skeletonization: Node types that have bodies which can be collapsed
SKELETONIZABLE_TYPES = {
    "function_definition",  # Python
    "class_definition",  # Python
    "function_declaration",  # JS/TS/Go/Java
    "class_declaration",  # JS/TS/Java
    "method_definition",  # JS/TS
    "method_declaration",  # Java
    "arrow_function",  # JS/TS
    "function_item",  # Rust
    "impl_item",  # Rust
    "struct_item",  # Rust
}

# Node types that represent function/method bodies
BODY_TYPES = {
    "block",  # Python, JS/TS, Java, Go
    "statement_block",  # JS/TS
    "compound_statement",  # C/C++
    "function_body",  # Various
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


def _estimate_tokens(content: str) -> int:
    """Count tokens using tiktoken (cl100k_base encoding)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return len(_encoder.encode(content)) + OVERHEAD_PER_FILE


class Skeletonizer:
    """Compress code files by showing only signatures for non-focus functions.

    This reduces token usage while preserving structural context - the LLM can
    see what classes/methods exist without the implementation details.
    """

    # Map extensions to tree-sitter language module names
    EXT_TO_LANG = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
    }

    LANG_TO_MODULE = {
        "python": ("tree_sitter_python", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"),
        "go": ("tree_sitter_go", "language"),
        "rust": ("tree_sitter_rust", "language"),
        "java": ("tree_sitter_java", "language"),
        "c": ("tree_sitter_c", "language"),
        "cpp": ("tree_sitter_cpp", "language"),
    }

    def __init__(self):
        self._parsers: dict[str, any] = {}

    def _get_parser(self, lang: str):
        """Get or create a parser for the given language."""
        if lang in self._parsers:
            return self._parsers[lang]

        try:
            import importlib

            from tree_sitter import Language, Parser

            if lang not in self.LANG_TO_MODULE:
                return None

            module_name, func_name = self.LANG_TO_MODULE[lang]
            lang_module = importlib.import_module(module_name)
            lang_func = getattr(lang_module, func_name)
            language = Language(lang_func())
            parser = Parser(language)
            self._parsers[lang] = parser
            return parser
        except Exception as e:
            logger.debug("Failed to get parser for %s: %s", lang, e)
            return None

    def _get_node_name(self, node: Node) -> str | None:
        """Extract the name from a definition node."""
        for field in ("name", "declarator"):
            name_node = node.child_by_field_name(field)
            if name_node:
                while name_node.type in ("function_declarator", "pointer_declarator"):
                    inner = name_node.child_by_field_name("declarator")
                    if inner:
                        name_node = inner
                    else:
                        break

                if name_node.type in ("identifier", "type_identifier"):
                    return name_node.text.decode("utf-8") if name_node.text else None
                elif name_node.text:
                    return name_node.text.decode("utf-8")

        # Handle decorated_definition
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    return self._get_node_name(child)

        # Fallback
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8") if child.text else None

        return None

    def _find_body_node(self, node: Node) -> Node | None:
        """Find the body node within a definition."""
        # Try common field names
        for field in ("body", "consequence", "block"):
            body = node.child_by_field_name(field)
            if body:
                return body

        # Fallback: look for block-like children
        for child in node.children:
            if child.type in BODY_TYPES:
                return child

        return None

    def _extract_signature(self, node: Node, content: str, lang: str) -> str:
        """Extract just the signature (declaration line) of a definition."""
        node_text = content[node.start_byte : node.end_byte]
        lines = node_text.split("\n")

        if not lines:
            return ""

        # For Python decorated definitions, include decorators
        if node.type == "decorated_definition":
            sig_lines = []
            for line in lines:
                stripped = line.strip()
                sig_lines.append(line)
                if stripped.startswith(("def ", "class ", "async def ")):
                    # Include the colon
                    if ":" in stripped:
                        break
                    # Multi-line signature
                    continue
            return "\n".join(sig_lines)

        # For most languages, signature is until the opening brace or colon
        sig = lines[0]

        # Handle multi-line signatures (parameters spanning lines)
        if sig.count("(") > sig.count(")"):
            paren_depth = sig.count("(") - sig.count(")")
            for line in lines[1:10]:  # Max 10 lines for signature
                sig += "\n" + line
                paren_depth += line.count("(") - line.count(")")
                if paren_depth <= 0:
                    break

        return sig

    def _collapse_body(self, node: Node, content: str, lang: str, indent: str) -> str:
        """Replace a function body with '...' while preserving the signature."""
        sig = self._extract_signature(node, content, lang)

        # Determine the placeholder based on language
        if lang == "python":
            placeholder = f"{indent}    ..."
        elif lang in ("go", "rust", "java", "c", "cpp"):
            placeholder = f"{indent}    // ..."
        else:  # JavaScript/TypeScript
            placeholder = f"{indent}  // ..."

        # Combine signature and placeholder
        if lang == "python":
            # Python: signature ends with colon
            if sig.rstrip().endswith(":"):
                return f"{sig}\n{placeholder}"
            return f"{sig}:\n{placeholder}"
        else:
            # C-like languages: add braces if not present
            if "{" in sig:
                # Signature already has opening brace
                sig_before_brace = sig.rsplit("{", 1)[0]
                return f"{sig_before_brace}{{\n{placeholder}\n{indent}}}"
            else:
                return f"{sig} {{\n{placeholder}\n{indent}}}"

    def skeletonize(
        self,
        file_path: Path,
        content: str,
        focus_symbols: set[str] | None = None,
        focus_lines: set[int] | None = None,
    ) -> str:
        """Skeletonize a file, collapsing non-focus function bodies.

        Args:
            file_path: Path to the file
            content: File content
            focus_symbols: Set of symbol names to keep full (e.g., {"my_func", "MyClass"})
            focus_lines: Set of line numbers to keep full (any function containing these lines)

        Returns:
            Skeletonized content with collapsed function bodies
        """
        lang = self.EXT_TO_LANG.get(file_path.suffix.lower())
        if not lang:
            return content

        parser = self._get_parser(lang)
        if not parser:
            return content

        try:
            tree = parser.parse(bytes(content, "utf-8"))
        except Exception:
            return content

        focus_symbols = focus_symbols or set()
        focus_lines = focus_lines or set()

        # Collect all modifications to make
        # List of (start_byte, end_byte, replacement)
        modifications: list[tuple[int, int, str]] = []

        def should_keep_full(node: Node) -> bool:
            """Check if a node should keep its full body."""
            name = self._get_node_name(node)
            if name and name in focus_symbols:
                return True

            # Check if any focus line is within this node
            node_start_line = node.start_point[0] + 1
            node_end_line = node.end_point[0] + 1
            for line in focus_lines:
                if node_start_line <= line <= node_end_line:
                    return True

            return False

        def collect_modifications(node: Node, depth: int = 0):
            """Recursively collect nodes to collapse."""
            if node.type in SKELETONIZABLE_TYPES:
                if not should_keep_full(node):
                    # Calculate indentation
                    line_start = content.rfind("\n", 0, node.start_byte) + 1
                    indent = content[line_start : node.start_byte]

                    collapsed = self._collapse_body(node, content, lang, indent)
                    modifications.append((node.start_byte, node.end_byte, collapsed))
                    return  # Don't recurse into collapsed nodes

            # Recurse into children
            for child in node.children:
                collect_modifications(child, depth + 1)

        collect_modifications(tree.root_node)

        if not modifications:
            return content

        # Apply modifications in reverse order to preserve byte offsets
        modifications.sort(key=lambda x: x[0], reverse=True)
        result = content
        for start, end, replacement in modifications:
            result = result[:start] + replacement + result[end:]

        return result


# Global skeletonizer instance
_skeletonizer: Skeletonizer | None = None


def get_skeletonizer() -> Skeletonizer:
    """Get or create the global Skeletonizer instance."""
    global _skeletonizer
    if _skeletonizer is None:
        _skeletonizer = Skeletonizer()
    return _skeletonizer


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
        skeletonize: bool = False,
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
            skeletonize: Collapse non-focus function bodies to signatures only

        Returns:
            PackResult with included files/chunks
        """
        result = PackResult()

        # Unified coverage registry to prevent duplicates across Focus and Search
        # Maps Path -> set of covered line ranges (start, end)
        # If full file is included, range is (0, float('inf'))
        coverage: dict[Path, set[tuple[int, float]]] = {}

        def is_fully_covered(path: Path) -> bool:
            ranges = coverage.get(path, set())
            return (0, float("inf")) in ranges

        def add_file_coverage(path: Path):
            coverage[path] = {(0, float("inf"))}

        def add_chunk_coverage(path: Path, start: int, end: int) -> bool:
            """Return True if this chunk adds NEW info."""
            if is_fully_covered(path):
                return False

            # Simple overlap check
            ranges = coverage.setdefault(path, set())
            if (start, end) in ranges:
                return False

            ranges.add((start, end))
            return True

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

        # Track focus symbols for skeletonization
        focus_symbols: set[str] = set()
        focus_lines: set[int] = set()

        # 1. If focus file specified, resolve imports
        if focus:
            focus_path = Path(focus)
            if not focus_path.is_absolute():
                focus_path = (self.root_dir / focus_path).resolve()
            else:
                focus_path = focus_path.resolve()
                # Validate path is within root_dir
                try:
                    focus_path.relative_to(self.root_dir)
                except ValueError:
                    # Path is outside root_dir, treat as relative
                    logger.warning("Focus path %s is outside root_dir, treating as relative", focus)
                    focus_path = (self.root_dir / focus).resolve()

            if focus_path.exists() and focus_path.is_file():
                # Add focus file first (always full content)
                self._add_file(result, focus_path, "focus file", coverage, skeletonize=False)

                # BFS import resolution
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
                    self._add_file(
                        result, dep_path, "import", coverage,
                        skeletonize=skeletonize, focus_symbols=focus_symbols
                    )

        # 2. If query specified, do semantic search
        if query and result.total_tokens < (max_tokens or float("inf")):
            # Build index if needed
            self.indexer.build_index()

            # Search for relevant chunks (hybrid + rerank)
            # Fetch extra to account for offset and deduplication
            limit = max_chunks if chunk_mode else max_files
            search_limit = (offset + limit) * 3
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

                path = Path(sr["file_path"])

                # Skip if already fully covered (deduplication)
                if is_fully_covered(path):
                    continue

                # Check budget (always allow first result)
                is_first = len(result.chunks) == 0 and len(result.files) == 0
                if not _within_budget(is_first=is_first):
                    truncation_reason = "token_budget"
                    break

                if chunk_mode:
                    # Chunk mode: return individual chunks
                    start = sr["start_line"]
                    end = sr["end_line"]

                    if not add_chunk_coverage(path, start, end):
                        continue  # Already have this chunk

                    if len(result.chunks) >= max_chunks:
                        truncation_reason = "max_chunks"
                        break

                    content = sr["content"]
                    cost = _estimate_tokens(content)

                    if result.total_tokens + cost > (max_tokens or float("inf")):
                        truncation_reason = "token_budget"
                        break

                    chunk = PackedChunk(
                        path=path,
                        content=content,
                        start_line=start,
                        end_line=end,
                        name=sr.get("name", ""),
                        reason=f"semantic match (score: {score:.2f})",
                        score=score,
                        score_type=item_score_type,
                    )
                    result.chunks.append(chunk)
                    result.total_lines += end - start + 1
                    result.total_chars += len(content)
                    result.total_tokens += cost
                else:
                    # File mode: expand to full files
                    if len(result.files) >= max_files:
                        truncation_reason = "max_files"
                        break

                    # Track matched symbols for skeletonization
                    matched_name = sr.get('name', '')
                    if matched_name:
                        focus_symbols.add(matched_name)
                    # Track matched lines
                    focus_lines.update(range(sr.get('start_line', 0), sr.get('end_line', 0) + 1))

                    self._add_file(
                        result,
                        path,
                        f"semantic match: {matched_name or 'chunk'}",
                        coverage,
                        score=score,
                        score_type=item_score_type,
                        skeletonize=skeletonize,
                        focus_symbols=focus_symbols,
                        focus_lines=focus_lines,
                    )

        # 3. Post-processing: Merge adjacent chunks
        if chunk_mode and result.chunks:
            result.chunks = self._merge_chunks(result.chunks)
            # Recalculate totals after merging
            result.total_lines = sum(c.end_line - c.start_line + 1 for c in result.chunks)
            result.total_chars = sum(len(c.content) for c in result.chunks)
            result.total_tokens = sum(_estimate_tokens(c.content) for c in result.chunks)

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
        coverage: dict[Path, set[tuple[int, float]]],
        score: float = 1.0,
        score_type: str = "dense",
        skeletonize: bool = False,
        focus_symbols: set[str] | None = None,
        focus_lines: set[int] | None = None,
    ):
        """Add a file to the pack result.

        Args:
            result: PackResult to add to
            file_path: Path to the file
            reason: Why this file was included
            coverage: Coverage tracking dict
            score: Relevance score
            score_type: Type of score
            skeletonize: Whether to collapse non-focus function bodies
            focus_symbols: Symbols to keep full when skeletonizing
            focus_lines: Lines to keep full when skeletonizing
        """
        # Check if already fully covered
        if (0, float("inf")) in coverage.get(file_path, set()):
            return
        if not file_path.exists():
            return

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Apply skeletonization if requested
            if skeletonize:
                skeletonizer = get_skeletonizer()
                content = skeletonizer.skeletonize(
                    file_path, content,
                    focus_symbols=focus_symbols,
                    focus_lines=focus_lines
                )
                reason = f"{reason} [skeletonized]"

            cost = _estimate_tokens(content)

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
            result.total_tokens += cost

            # Mark fully covered
            coverage.setdefault(file_path, set()).add((0, float("inf")))

        except Exception:
            logger.warning("Could not read %s", file_path, exc_info=True)

    def _merge_chunks(self, chunks: list[PackedChunk]) -> list[PackedChunk]:
        """Merge adjacent or overlapping chunks from the same file."""
        if not chunks:
            return []

        # Sort by file path, then start line
        chunks.sort(key=lambda x: (x.path, x.start_line))

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # If same file and overlap or adjacent (within 5 lines)
            if (
                current.path == next_chunk.path
                and next_chunk.start_line <= current.end_line + 5
            ):
                # Merge by re-reading the file slice
                try:
                    full_content = current.path.read_text(encoding="utf-8", errors="ignore")
                    lines = full_content.splitlines()

                    new_end = max(current.end_line, next_chunk.end_line)
                    # Python list slicing is 0-indexed, lines are 1-indexed
                    merged_content = "\n".join(lines[current.start_line - 1 : new_end])

                    current = PackedChunk(
                        path=current.path,
                        content=merged_content,
                        start_line=current.start_line,
                        end_line=new_end,
                        name=current.name,  # Keep first name
                        reason=current.reason,
                        score=max(current.score, next_chunk.score),
                        score_type=current.score_type,
                    )
                except Exception:
                    # Fallback: just append strings if file read fails
                    current.content += "\n" + next_chunk.content
                    current.end_line = max(current.end_line, next_chunk.end_line)
                    current.score = max(current.score, next_chunk.score)
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged


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
