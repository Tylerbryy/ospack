"""High-level workflow tools for AI agents."""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .chunker import get_chunker
from .indexer import EXCLUDE_PATTERNS, is_text_file, get_indexer
from .log import get_logger
from .packer import extract_score
from .resolver import get_resolver

logger = get_logger(__name__)

MAX_WORKERS = min(os.cpu_count() or 4, 8)


@dataclass
class Implementation:
    """A code implementation match."""

    file_path: Path
    name: str
    content: str
    start_line: int
    end_line: int
    score: float
    score_type: str


@dataclass
class FindResult:
    """Result of find_implementation."""

    implementations: list[Implementation]
    query: str
    total_found: int


@dataclass
class CodeContext:
    """Context around a piece of code."""

    file_path: Path
    content: str
    start_line: int
    end_line: int
    name: str | None = None


@dataclass
class ExplainResult:
    """Result of explain_code."""

    target: CodeContext
    imports: list[Path]  # Files this code imports
    imported_by: list[Path]  # Files that import this code
    related_chunks: list[CodeContext]  # Semantically related code


@dataclass
class DependencyNode:
    """A node in the dependency graph."""

    file_path: Path
    imports: list[Path]
    imported_by: list[Path]


@dataclass
class DiscoverResult:
    """Result of discover_related."""

    entry_point: Path
    dependency_graph: dict[str, DependencyNode]
    semantic_relatives: list[CodeContext]
    total_files: int


@dataclass
class ImpactResult:
    """Result of analyze_impact."""

    target: Path
    function: str | None
    directly_affected: list[Path]  # Files that import the target
    transitively_affected: list[Path]  # Files affected through dependency chain
    total_affected: int


# --- Helper for Parallelism ---
def _resolve_deps_worker(args: tuple[str, str]) -> tuple[str, list[str]]:
    """Worker to resolve imports for a single file (runs in subprocess)."""
    root_dir, file_path = args
    try:
        resolver = get_resolver(root_dir)
        imports = resolver.resolve_imports(file_path)
        return str(file_path), [str(p) for p in imports]
    except Exception:
        return str(file_path), []


class Workflows:
    """High-level workflow tools for code exploration."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.indexer = get_indexer(str(self.root_dir))
        self.resolver = get_resolver(str(self.root_dir))
        self.chunker = get_chunker()  # Precise parsing tool
        self._reverse_deps: dict[Path, list[Path]] | None = None

    def find_implementation(
        self,
        concept: str,
        max_results: int = 5,
        rerank: bool = True,
    ) -> FindResult:
        """
        Find where a concept/function is implemented using Hybrid Search.

        Args:
            concept: Natural language description (e.g., "user authentication")
            max_results: Maximum implementations to return
            rerank: Use cross-encoder reranking

        Returns:
            FindResult with ranked implementations
        """
        # Build index if needed
        self.indexer.build_index()

        # Search for implementations
        results = self.indexer.search(
            concept,
            limit=max_results * 2,
            rerank=rerank,
            hybrid=True,
        )

        implementations = []
        for r in results[:max_results]:
            score, score_type = extract_score(r)

            implementations.append(
                Implementation(
                    file_path=Path(r["file_path"]),
                    name=r.get("name", ""),
                    content=r["content"],
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                    score=score,
                    score_type=score_type,
                )
            )

        return FindResult(
            implementations=implementations,
            query=concept,
            total_found=len(results),
        )

    def explain_code(
        self,
        file: str,
        function: str | None = None,
        include_imports: bool = True,
        include_importers: bool = True,
        include_related: bool = True,
        max_related: int = 3,
    ) -> ExplainResult:
        """
        Get explanation of code with context.
        Uses Chunker for PRECISE definition finding, not fuzzy search.

        Args:
            file: Path to the file
            function: Optional function/class name to focus on
            include_imports: Include files this code imports
            include_importers: Include files that import this code
            include_related: Include semantically related chunks
            max_related: Max related chunks to include

        Returns:
            ExplainResult with code and surrounding context
        """
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = self.root_dir / file_path
        file_path = file_path.resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file}")

        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # 1. Precise Target Identification using Chunker
        target: CodeContext
        if function:
            # Re-parse file to find exact node
            chunks = self.chunker.chunk(str(file_path), content)

            # Look for exact name match (case-insensitive)
            match = next(
                (c for c in chunks if c.name and c.name.lower() == function.lower()),
                None,
            )

            if match:
                target = CodeContext(
                    file_path=file_path,
                    content=match.content,
                    start_line=match.start_line,
                    end_line=match.end_line,
                    name=match.name,
                )
            else:
                # Fallback: Function requested but not found, return whole file
                target = CodeContext(
                    file_path=file_path,
                    content=content,
                    start_line=1,
                    end_line=len(content.splitlines()),
                    name=function,
                )
        else:
            target = CodeContext(
                file_path=file_path,
                content=content,
                start_line=1,
                end_line=len(content.splitlines()),
            )

        # 2. Get imports
        imports: list[Path] = []
        if include_imports:
            imports = list(self.resolver.resolve_imports(file_path))

        # 3. Get reverse dependencies (files that import this)
        imported_by: list[Path] = []
        if include_importers:
            imported_by = self._get_reverse_deps(file_path)

        # 4. Get semantically related chunks (use search for *related* logic only)
        related_chunks: list[CodeContext] = []
        if include_related and function:
            self.indexer.build_index()
            # Contextual query combining function name and file name
            results = self.indexer.search(
                f"{function} {file_path.stem}",
                limit=max_related + 5,
                rerank=True,
                hybrid=True,
            )
            for r in results:
                # Skip the target file
                if Path(r["file_path"]) == file_path:
                    continue
                if len(related_chunks) >= max_related:
                    break
                related_chunks.append(
                    CodeContext(
                        file_path=Path(r["file_path"]),
                        content=r["content"],
                        start_line=r["start_line"],
                        end_line=r["end_line"],
                        name=r.get("name"),
                    )
                )

        return ExplainResult(
            target=target,
            imports=imports,
            imported_by=imported_by,
            related_chunks=related_chunks,
        )

    def discover_related(
        self,
        entry_point: str,
        depth: int = 2,
        include_semantic: bool = True,
        max_semantic: int = 5,
    ) -> DiscoverResult:
        """
        Find all related code from an entry point.

        Args:
            entry_point: Path to entry file
            depth: Import resolution depth
            include_semantic: Include semantically related code
            max_semantic: Max semantic matches to include

        Returns:
            DiscoverResult with dependency graph and related code
        """
        entry_path = Path(entry_point)
        if not entry_path.is_absolute():
            entry_path = self.root_dir / entry_path
        entry_path = entry_path.resolve()

        # Build dependency graph
        dep_graph = self.resolver.get_dependency_graph(entry_path, max_depth=depth)

        # Build reverse deps for all files in graph
        graph_with_reverse: dict[str, DependencyNode] = {}
        all_files = set(dep_graph.keys())

        for _, imports in dep_graph.items():
            all_files.update(imports)

        for file_path in all_files:
            imports = dep_graph.get(file_path, [])
            imported_by = [f for f, deps in dep_graph.items() if file_path in deps]
            rel_path = str(file_path.relative_to(self.root_dir))
            graph_with_reverse[rel_path] = DependencyNode(
                file_path=file_path,
                imports=imports,
                imported_by=imported_by,
            )

        # Get semantic relatives
        semantic_relatives: list[CodeContext] = []
        if include_semantic:
            # Use entry point name as query
            self.indexer.build_index()
            results = self.indexer.search(
                entry_path.stem,  # filename without extension
                limit=max_semantic + len(all_files),
                rerank=True,
                hybrid=True,
            )
            for r in results:
                # Skip files already in dependency graph
                if Path(r["file_path"]) in all_files:
                    continue
                if len(semantic_relatives) >= max_semantic:
                    break
                semantic_relatives.append(
                    CodeContext(
                        file_path=Path(r["file_path"]),
                        content=r["content"],
                        start_line=r["start_line"],
                        end_line=r["end_line"],
                        name=r.get("name"),
                    )
                )

        return DiscoverResult(
            entry_point=entry_path,
            dependency_graph=graph_with_reverse,
            semantic_relatives=semantic_relatives,
            total_files=len(all_files),
        )

    def analyze_impact(
        self,
        file: str,
        function: str | None = None,
        max_depth: int = 3,
        fuzzy_matching: bool = True,
    ) -> ImpactResult:
        """
        Analyze what would be affected by changes to a file/function.
        Uses iterative BFS to avoid recursion limits.
        Optionally uses "fuzzy matching" for DI frameworks.

        Args:
            file: Path to the file being changed
            function: Optional specific function being changed
            max_depth: Max depth for transitive analysis
            fuzzy_matching: Use symbol-based matching for DI frameworks

        Returns:
            ImpactResult with affected files
        """
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = self.root_dir / file_path
        file_path = file_path.resolve()

        # 1. Build Strict Graph (Cached)
        self._build_reverse_deps()

        # 2. Identify "Magic" References (Fuzzy Fallback for DI frameworks)
        magic_dependents: set[Path] = set()

        if fuzzy_matching:
            try:
                # What "Symbols" does this file export?
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = self.chunker.chunk(str(file_path), content)

                # Filter short names to avoid false positives
                exported_symbols = [c.name for c in chunks if c.name and len(c.name) > 4]

                if exported_symbols:
                    # Who mentions these symbols?
                    self.indexer.build_index()
                    for sym in exported_symbols[:5]:  # Limit to top 5 symbols
                        results = self.indexer.search(
                            sym, limit=20, rerank=False, hybrid=True
                        )
                        for r in results:
                            potential_dep = Path(r["file_path"])
                            # If another file mentions our class name, it's a "Magic" dependent
                            if potential_dep != file_path:
                                magic_dependents.add(potential_dep)
            except Exception:
                pass  # Graceful degradation

        # 3. Merge Strict and Magic dependents
        directly_affected = set(self._reverse_deps.get(file_path, []))
        directly_affected.update(magic_dependents)

        # 4. Iterative BFS Traversal (avoids RecursionError)
        transitively_affected: list[Path] = []
        queue = deque([(p, 1) for p in directly_affected])  # (path, depth)
        visited = set(directly_affected)
        visited.add(file_path)

        while queue:
            current_path, current_depth = queue.popleft()

            # Record as transitive (if not direct)
            if current_path not in directly_affected:
                transitively_affected.append(current_path)

            # Stop if max depth reached
            if current_depth >= max_depth:
                continue

            # Find next dependents (use strict deps for traversal)
            dependents = self._reverse_deps.get(current_path, [])
            for dep in dependents:
                if dep not in visited:
                    visited.add(dep)
                    queue.append((dep, current_depth + 1))

        return ImpactResult(
            target=file_path,
            function=function,
            directly_affected=list(directly_affected),
            transitively_affected=transitively_affected,
            total_affected=len(directly_affected) + len(transitively_affected),
        )

    def _build_reverse_deps(self):
        """Build reverse dependency graph using Parallel Processing."""
        if self._reverse_deps is not None:
            return

        self._reverse_deps = {}

        # 1. Identify Source Files
        from pathspec import PathSpec
        from pathspec.patterns import GitWildMatchPattern

        exclude_spec = PathSpec.from_lines(GitWildMatchPattern, EXCLUDE_PATTERNS)

        tasks: list[tuple[str, str]] = []
        for path in self.root_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(self.root_dir))
            if exclude_spec.match_file(rel):
                continue
            if not is_text_file(path):
                continue
            tasks.append((str(self.root_dir), str(path)))

        if not tasks:
            return

        # 2. Parallel Resolution
        forward_deps: dict[Path, list[Path]] = {}

        # For small repos, sequential is faster than process spawn overhead
        if len(tasks) < 50:
            for root_dir, file_path in tasks:
                imports = self.resolver.resolve_imports(file_path)
                forward_deps[Path(file_path)] = list(imports)
        else:
            # Parallel for larger repos
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(_resolve_deps_worker, t) for t in tasks]
                for future in as_completed(futures):
                    f_path_str, imports_str = future.result()
                    f_path = Path(f_path_str)
                    forward_deps[f_path] = [Path(p) for p in imports_str]

        # 3. Invert Graph (Fast, In-Memory)
        for file_path, imports in forward_deps.items():
            for imported in imports:
                if imported not in self._reverse_deps:
                    self._reverse_deps[imported] = []
                if file_path not in self._reverse_deps[imported]:
                    self._reverse_deps[imported].append(file_path)

        logger.debug("Built reverse deps for %d files", len(forward_deps))

    def _get_reverse_deps(self, file_path: Path) -> list[Path]:
        """Get files that import the given file."""
        self._build_reverse_deps()
        return self._reverse_deps.get(file_path, [])


def get_workflows(root_dir: str) -> Workflows:
    """Get a Workflows instance for the given root directory."""
    return Workflows(root_dir)


# Output formatters for CLI
def format_find_result(
    result: FindResult,
    root_dir: Path,
    format: Literal["xml", "compact"] = "xml",
) -> str:
    """Format find_implementation result."""
    if format == "xml":
        lines = [f'<implementations query="{result.query}" total="{result.total_found}">']
        for impl in result.implementations:
            rel_path = impl.file_path.relative_to(root_dir)
            name_attr = f' name="{impl.name}"' if impl.name else ""
            lines.append(
                f'  <impl path="{rel_path}" lines="{impl.start_line}-{impl.end_line}"'
                f'{name_attr} score="{impl.score:.2f}" score_type="{impl.score_type}">'
            )
            ext = impl.file_path.suffix.lstrip(".")
            lines.append(f"```{ext}")
            lines.append(impl.content)
            lines.append("```")
            lines.append("  </impl>")
        lines.append("</implementations>")
        return "\n".join(lines)
    else:
        lines = [f"# Implementations for: {result.query}", ""]
        for impl in result.implementations:
            rel_path = impl.file_path.relative_to(root_dir)
            ext = impl.file_path.suffix.lstrip(".")
            name_info = f" ({impl.name})" if impl.name else ""
            lines.append(
                f"## {rel_path}:{impl.start_line}-{impl.end_line}{name_info} "
                f"[score: {impl.score:.2f}]"
            )
            lines.append(f"```{ext}")
            lines.append(impl.content)
            lines.append("```")
            lines.append("")
        lines.append(f"<!-- {len(result.implementations)} of {result.total_found} shown -->")
        return "\n".join(lines)


def format_explain_result(
    result: ExplainResult,
    root_dir: Path,
    format: Literal["xml", "compact"] = "xml",
) -> str:
    """Format explain_code result."""
    rel_target = result.target.file_path.relative_to(root_dir)

    if format == "xml":
        lines = ["<explanation>"]

        # Target code
        name_attr = f' name="{result.target.name}"' if result.target.name else ""
        lines.append(
            f'  <target path="{rel_target}" '
            f'lines="{result.target.start_line}-{result.target.end_line}"{name_attr}>'
        )
        ext = result.target.file_path.suffix.lstrip(".")
        lines.append(f"```{ext}")
        lines.append(result.target.content)
        lines.append("```")
        lines.append("  </target>")

        # Imports
        if result.imports:
            lines.append("  <imports>")
            for imp in result.imports:
                rel_imp = imp.relative_to(root_dir)
                lines.append(f'    <file path="{rel_imp}"/>')
            lines.append("  </imports>")

        # Imported by
        if result.imported_by:
            lines.append("  <imported_by>")
            for imp in result.imported_by:
                rel_imp = imp.relative_to(root_dir)
                lines.append(f'    <file path="{rel_imp}"/>')
            lines.append("  </imported_by>")

        # Related
        if result.related_chunks:
            lines.append("  <related>")
            for ctx in result.related_chunks:
                rel_ctx = ctx.file_path.relative_to(root_dir)
                name_attr = f' name="{ctx.name}"' if ctx.name else ""
                lines.append(
                    f'    <chunk path="{rel_ctx}" '
                    f'lines="{ctx.start_line}-{ctx.end_line}"{name_attr}>'
                )
                ext = ctx.file_path.suffix.lstrip(".")
                lines.append(f"```{ext}")
                lines.append(ctx.content)
                lines.append("```")
                lines.append("    </chunk>")
            lines.append("  </related>")

        lines.append("</explanation>")
        return "\n".join(lines)
    else:
        lines = [f"# Explanation: {rel_target}", ""]

        # Target
        name_info = f" ({result.target.name})" if result.target.name else ""
        lines.append(f"## Target{name_info}")
        ext = result.target.file_path.suffix.lstrip(".")
        lines.append(f"```{ext}")
        lines.append(result.target.content)
        lines.append("```")
        lines.append("")

        # Imports
        if result.imports:
            lines.append("## Imports")
            for imp in result.imports:
                lines.append(f"- {imp.relative_to(root_dir)}")
            lines.append("")

        # Imported by
        if result.imported_by:
            lines.append("## Imported By")
            for imp in result.imported_by:
                lines.append(f"- {imp.relative_to(root_dir)}")
            lines.append("")

        # Related
        if result.related_chunks:
            lines.append("## Related Code")
            for ctx in result.related_chunks:
                rel_ctx = ctx.file_path.relative_to(root_dir)
                name_info = f" ({ctx.name})" if ctx.name else ""
                lines.append(f"### {rel_ctx}:{ctx.start_line}-{ctx.end_line}{name_info}")
                ext = ctx.file_path.suffix.lstrip(".")
                lines.append(f"```{ext}")
                lines.append(ctx.content)
                lines.append("```")
                lines.append("")

        return "\n".join(lines)


def format_impact_result(
    result: ImpactResult,
    root_dir: Path,
    format: Literal["xml", "compact"] = "xml",
) -> str:
    """Format analyze_impact result."""
    rel_target = result.target.relative_to(root_dir)

    if format == "xml":
        func_attr = f' function="{result.function}"' if result.function else ""
        lines = [
            f'<impact target="{rel_target}"{func_attr} '
            f'total_affected="{result.total_affected}">'
        ]

        if result.directly_affected:
            lines.append("  <directly_affected>")
            for f in result.directly_affected:
                lines.append(f'    <file path="{f.relative_to(root_dir)}"/>')
            lines.append("  </directly_affected>")

        if result.transitively_affected:
            lines.append("  <transitively_affected>")
            for f in result.transitively_affected:
                lines.append(f'    <file path="{f.relative_to(root_dir)}"/>')
            lines.append("  </transitively_affected>")

        lines.append("</impact>")
        return "\n".join(lines)
    else:
        func_info = f" ({result.function})" if result.function else ""
        lines = [
            f"# Impact Analysis: {rel_target}{func_info}",
            f"Total affected: {result.total_affected}",
            "",
        ]

        if result.directly_affected:
            lines.append("## Directly Affected")
            for f in result.directly_affected:
                lines.append(f"- {f.relative_to(root_dir)}")
            lines.append("")

        if result.transitively_affected:
            lines.append("## Transitively Affected")
            for f in result.transitively_affected:
                lines.append(f"- {f.relative_to(root_dir)}")
            lines.append("")

        return "\n".join(lines)
