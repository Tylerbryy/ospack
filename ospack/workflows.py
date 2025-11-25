"""High-level workflow tools for AI agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .indexer import get_indexer
from .packer import extract_score
from .resolver import get_resolver


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


class Workflows:
    """High-level workflow tools for code exploration."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.indexer = get_indexer(str(self.root_dir))
        self.resolver = get_resolver(str(self.root_dir))
        self._reverse_deps: dict[Path, list[Path]] | None = None

    def find_implementation(
        self,
        concept: str,
        max_results: int = 5,
        rerank: bool = True,
    ) -> FindResult:
        """
        Find where a concept/function is implemented.

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

        # Find the target chunk
        target: CodeContext
        if function:
            # Search for the specific function in the index
            self.indexer.build_index()
            results = self.indexer.search(
                f"{function} {file_path.name}",
                limit=10,
                rerank=False,
                hybrid=True,
            )
            # Find matching chunk
            for r in results:
                if (
                    Path(r["file_path"]) == file_path
                    and r.get("name", "").lower() == function.lower()
                ):
                    target = CodeContext(
                        file_path=file_path,
                        content=r["content"],
                        start_line=r["start_line"],
                        end_line=r["end_line"],
                        name=r.get("name"),
                    )
                    break
            else:
                # Fall back to full file
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

        # Get imports
        imports: list[Path] = []
        if include_imports:
            imports = self.resolver.resolve_imports(file_path, content)

        # Get reverse dependencies (files that import this)
        imported_by: list[Path] = []
        if include_importers:
            imported_by = self._get_reverse_deps(file_path)

        # Get semantically related chunks
        related_chunks: list[CodeContext] = []
        if include_related and function:
            self.indexer.build_index()
            results = self.indexer.search(
                function,
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
    ) -> ImpactResult:
        """
        Analyze what would be affected by changes to a file/function.

        Args:
            file: Path to the file being changed
            function: Optional specific function being changed
            max_depth: Max depth for transitive analysis

        Returns:
            ImpactResult with affected files
        """
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = self.root_dir / file_path
        file_path = file_path.resolve()

        # Build reverse dependency map
        self._build_reverse_deps()

        # Direct dependents
        directly_affected = self._reverse_deps.get(file_path, [])

        # Transitive dependents
        transitively_affected: list[Path] = []
        visited = set(directly_affected)
        visited.add(file_path)

        def traverse(paths: list[Path], depth: int):
            if depth >= max_depth:
                return
            for p in paths:
                dependents = self._reverse_deps.get(p, [])
                for dep in dependents:
                    if dep not in visited:
                        visited.add(dep)
                        transitively_affected.append(dep)
                traverse(dependents, depth + 1)

        traverse(directly_affected, 1)

        return ImpactResult(
            target=file_path,
            function=function,
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            total_affected=len(directly_affected) + len(transitively_affected),
        )

    def _build_reverse_deps(self):
        """Build reverse dependency map for all source files."""
        if self._reverse_deps is not None:
            return

        self._reverse_deps = {}

        # Get all source files (reuse indexer's file discovery)
        from pathspec import PathSpec
        from pathspec.patterns import GitWildMatchPattern

        from .indexer import EXCLUDE_PATTERNS, INCLUDE_PATTERNS

        include_spec = PathSpec.from_lines(GitWildMatchPattern, INCLUDE_PATTERNS)
        exclude_spec = PathSpec.from_lines(GitWildMatchPattern, EXCLUDE_PATTERNS)

        source_files = []
        for path in self.root_dir.rglob("*"):
            if not path.is_file():
                continue
            rel_path = str(path.relative_to(self.root_dir))
            if include_spec.match_file(rel_path) and not exclude_spec.match_file(rel_path):
                source_files.append(path)

        # Build forward deps then invert
        for file_path in source_files:
            imports = self.resolver.resolve_imports(file_path)
            for imported in imports:
                if imported not in self._reverse_deps:
                    self._reverse_deps[imported] = []
                if file_path not in self._reverse_deps[imported]:
                    self._reverse_deps[imported].append(file_path)

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
