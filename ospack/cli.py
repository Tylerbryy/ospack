"""CLI interface for ospack."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .errors import ErrorCode, ErrorResponse
from .packer import Packer, Verbosity, format_output

console = Console()


def _format_error(err: ErrorResponse, format: str = "compact") -> str:
    """Format error response for output."""
    if format == "xml" or format == "chunks":
        return err.format_xml()
    return err.format_compact()


@click.group()
@click.version_option()
def main():
    """ospack - Semantic context packer for AI-assisted coding."""
    pass


@main.command()
@click.option(
    "--focus", "-f",
    help="Entry point file for import graph traversal. Resolves imports transitively "
         "up to --import-depth levels. Use when you have a specific file and want its dependencies. "
         "Example: --focus src/auth.py"
)
@click.option(
    "--query", "-q",
    help="Natural language query for semantic search. Finds code matching the concept, "
         "not just literal keywords. Use when searching for functionality by description. "
         "Example: --query 'user authentication with JWT'"
)
@click.option(
    "--max-files", "-m", default=10, type=int,
    help="Maximum number of files to return. Lower values (3-5) for focused context, "
         "higher (15-20) for broad exploration. Default: 10"
)
@click.option(
    "--max-chunks", "-c", default=20, type=int,
    help="Maximum chunks when using --format chunks. Each chunk is a function/class. Default: 20"
)
@click.option(
    "--max-tokens", "-t", type=int,
    help="Approximate token budget for output. Useful for fitting in LLM context windows. "
         "Example: --max-tokens 8000 for ~2K tokens of headroom in a 10K context"
)
@click.option(
    "--min-score", "-s", type=float,
    help="Filter results below this relevance score (0.0-1.0). Use 0.3-0.5 for strict filtering, "
         "lower for broader results. Not recommended unless results are too noisy."
)
@click.option(
    "--import-depth", "-d", default=2, type=int,
    help="How many levels of imports to follow from --focus file. 1=direct imports only, "
         "2=imports of imports, 3+=deep dependency tree. Default: 2"
)
@click.option(
    "--format", "-o",
    type=click.Choice(["xml", "compact", "chunks"]),
    default="xml",
    help="Output format. 'xml' for Claude-optimized structured output (default), "
         "'compact' for human-readable markdown, 'chunks' for granular function-level results"
)
@click.option(
    "--root", "-r", default=".",
    help="Repository root directory (absolute path recommended). All file paths are relative to this. "
         "Default: current directory"
)
@click.option("--rerank/--no-rerank", default=True, help="Use cross-encoder reranking for better relevance ordering")
@click.option("--hybrid/--no-hybrid", default=True, help="Combine keyword (BM25) and semantic search")
@click.option("--offset", default=0, type=int, help="Skip first N results (for pagination)")
@click.option("--quiet", "-Q", is_flag=True, help="Paths and scores only, no code content")
@click.option("--verbose", "-v", is_flag=True, help="Include all metadata and debug info")
def pack(
    focus: str | None,
    query: str | None,
    max_files: int,
    max_chunks: int,
    max_tokens: int | None,
    min_score: float,
    import_depth: int,
    format: str,
    root: str,
    rerank: bool,
    hybrid: bool,
    offset: int,
    quiet: bool,
    verbose: bool,
):
    """Pack relevant code context for AI assistants.

    This is the main command for gathering code context. It combines two strategies:

    1. IMPORT RESOLUTION (--focus): Starting from a file, follow import statements
       to find dependencies. Use when you have a specific file and need its context.

    2. SEMANTIC SEARCH (--query): Find code matching a natural language description.
       Use when searching for functionality by concept rather than file location.

    You can use both together: --focus finds dependencies, --query adds related code.

    OUTPUT FORMATS:
      xml     - Structured XML optimized for Claude (default)
      compact - Human-readable markdown
      chunks  - Granular function/class level (best for large codebases)

    EXAMPLES:
      ospack pack --focus src/auth.py --import-depth 2
      ospack pack --query "database connection pooling" --max-files 5
      ospack pack --focus src/api.py --query "error handling" --format chunks
    """
    if not focus and not query:
        err = ErrorResponse.create(
            code=ErrorCode.MISSING_REQUIRED,
            error="Must specify --focus and/or --query",
            context={"missing": ["focus", "query"]},
        )
        console.print(_format_error(err, format))
        raise SystemExit(1)

    root_path = Path(root).resolve()
    if not root_path.exists():
        err = ErrorResponse.create(
            code=ErrorCode.INVALID_PATH,
            error=f"Root directory not found: {root}",
            context={"path": str(root), "resolved": str(root_path)},
        )
        console.print(_format_error(err, format))
        raise SystemExit(1)

    packer = Packer(str(root_path))

    # Enable chunk mode when format is "chunks"
    chunk_mode = format == "chunks"

    with console.status("Packing context..."):
        result = packer.pack(
            focus=focus,
            query=query,
            max_files=max_files,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
            min_score=min_score,
            depth=import_depth,
            rerank=rerank,
            hybrid=hybrid,
            chunk_mode=chunk_mode,
            offset=offset,
        )

    if not result.files and not result.chunks:
        err = ErrorResponse.create(
            code=ErrorCode.NO_RESULTS,
            error="No results found matching criteria",
            context={
                "query": query,
                "focus": focus,
                "min_score": min_score,
            },
        )
        console.print(_format_error(err, format))
        return

    # Determine verbosity (quiet takes precedence over verbose)
    verbosity = Verbosity.NORMAL
    if quiet:
        verbosity = Verbosity.QUIET
    elif verbose:
        verbosity = Verbosity.VERBOSE

    output = format_output(result, format=format, root_dir=root_path, verbosity=verbosity)
    console.print(output)


@main.command()
@click.option(
    "--root", "-r", default=".",
    help="Repository root directory (absolute path recommended)"
)
@click.option("--force", is_flag=True, help="Delete existing index and rebuild from scratch")
def index(root: str, force: bool):
    """Build or rebuild the semantic search index.

    Creates a vector index of all code in the repository for semantic search.
    The index is stored in ~/.ospack/lancedb/{repo-hash}/ and automatically
    updates when files change.

    Run this once after cloning a repo, or use --force to rebuild after
    major refactoring. Incremental updates happen automatically during pack/search.

    WHEN TO USE:
      - First time setup: ospack index --root /path/to/repo
      - After git pull with many changes: ospack index (auto-updates)
      - Index seems corrupted: ospack index --force

    WHEN NOT TO USE:
      - Before every pack command (index auto-updates)
      - Small file changes (handled incrementally)
    """
    from .indexer import get_indexer

    root_path = Path(root).resolve()
    if not root_path.exists():
        console.print(f"[red]Error:[/red] Root directory not found: {root}")
        raise SystemExit(1)

    indexer = get_indexer(str(root_path))

    with console.status("Building index..."):
        count = indexer.build_index(force=force)

    console.print(f"[green]Index built with {count} chunks[/green]")


@main.command()
@click.argument("query")
@click.option("--root", "-r", default=".", help="Repository root directory (absolute path recommended)")
@click.option("--limit", "-l", default=10, type=int, help="Maximum results to return (default: 10)")
def search(query: str, root: str, limit: int):
    """Quick semantic search without full context packing.

    Returns a table of matching code chunks with file paths, line numbers,
    and relevance scores. Use this for quick lookups; use 'pack --query'
    when you need the actual code content for an LLM.

    WHEN TO USE:
      - Quick exploration: "where is authentication handled?"
      - Finding file locations before using --focus
      - Verifying what the index contains

    WHEN NOT TO USE:
      - Need actual code content (use: ospack pack --query)
      - Need to copy results to an LLM (use: ospack pack --query --format xml)

    EXAMPLE:
      ospack search "database connection pooling" --limit 5
    """
    from .indexer import get_indexer

    root_path = Path(root).resolve()
    indexer = get_indexer(str(root_path))

    # Ensure index exists
    indexer.build_index()

    results = indexer.search(query, limit=limit)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Search results for: {query}")
    table.add_column("File", style="cyan")
    table.add_column("Lines", style="green")
    table.add_column("Name", style="yellow")
    table.add_column("Score", style="magenta")

    for r in results:
        rel_path = Path(r["file_path"]).relative_to(root_path)
        # Use best available score: rerank > rrf > base score
        score = r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0)
        table.add_row(
            str(rel_path),
            f"{r['start_line']}-{r['end_line']}",
            r["name"] or "-",
            f"{score:.3f}",
        )

    console.print(table)


@main.command()
@click.option("--root", "-r", default=".", help="Repository root directory (absolute path recommended)")
def info(root: str):
    """Show ospack configuration and index status.

    Displays the compute device (CPU/GPU), index location, and chunk count.
    Use this to verify ospack is configured correctly and the index exists.

    USEFUL FOR:
      - Checking if GPU acceleration is active
      - Finding where the index is stored
      - Verifying how many chunks are indexed
    """
    from .embedder import get_device
    from .indexer import get_indexer

    root_path = Path(root).resolve()

    console.print("[bold]ospack info[/bold]")
    console.print(f"Root: {root_path}")
    console.print(f"Device: {get_device()}")

    indexer = get_indexer(str(root_path))
    console.print(f"Index path: {indexer.storage_dir}")

    if indexer._table:
        try:
            count = len(indexer._table.to_pandas())
            console.print(f"Indexed chunks: {count}")
        except Exception:
            console.print("Index not loaded")


# ============================================================================
# High-Level Workflow Commands
# ============================================================================


@main.command()
@click.argument("concept")
@click.option("--root", "-r", default=".", help="Repository root directory")
@click.option("--max-results", "-n", default=5, type=int, help="Max implementations to show")
@click.option("--format", "-o", type=click.Choice(["xml", "compact"]), default="xml")
@click.option("--rerank/--no-rerank", default=True, help="Use cross-encoder reranking")
def find(concept: str, root: str, max_results: int, format: str, rerank: bool):
    """Find implementations of a concept.

    Example: ospack find "user authentication"
    """
    from .workflows import format_find_result, get_workflows

    root_path = Path(root).resolve()
    if not root_path.exists():
        err = ErrorResponse.create(
            code=ErrorCode.INVALID_PATH,
            error=f"Root directory not found: {root}",
        )
        console.print(_format_error(err, format))
        raise SystemExit(1)

    workflows = get_workflows(str(root_path))

    with console.status(f"Finding implementations of '{concept}'..."):
        result = workflows.find_implementation(
            concept=concept,
            max_results=max_results,
            rerank=rerank,
        )

    if not result.implementations:
        err = ErrorResponse.create(
            code=ErrorCode.NO_RESULTS,
            error=f"No implementations found for: {concept}",
            context={"query": concept},
        )
        console.print(_format_error(err, format))
        return

    output = format_find_result(result, root_path, format=format)
    console.print(output)


@main.command()
@click.argument("file")
@click.option("--function", "-f", help="Specific function/class to explain")
@click.option("--root", "-r", default=".", help="Repository root directory")
@click.option("--format", "-o", type=click.Choice(["xml", "compact"]), default="xml")
@click.option("--imports/--no-imports", default=True, help="Include imports")
@click.option("--importers/--no-importers", default=True, help="Include files that import this")
@click.option("--related/--no-related", default=True, help="Include semantically related code")
def explain(
    file: str,
    function: str | None,
    root: str,
    format: str,
    imports: bool,
    importers: bool,
    related: bool,
):
    """Explain code with context.

    Example: ospack explain src/auth.py --function login
    """
    from .workflows import format_explain_result, get_workflows

    root_path = Path(root).resolve()
    if not root_path.exists():
        err = ErrorResponse.create(
            code=ErrorCode.INVALID_PATH,
            error=f"Root directory not found: {root}",
        )
        console.print(_format_error(err, format))
        raise SystemExit(1)

    workflows = get_workflows(str(root_path))

    try:
        with console.status(f"Analyzing {file}..."):
            result = workflows.explain_code(
                file=file,
                function=function,
                include_imports=imports,
                include_importers=importers,
                include_related=related,
            )
    except FileNotFoundError as e:
        err = ErrorResponse.create(
            code=ErrorCode.FILE_NOT_FOUND,
            error=str(e),
            context={"file": file},
        )
        console.print(_format_error(err, format))
        raise SystemExit(1) from None

    output = format_explain_result(result, root_path, format=format)
    console.print(output)


@main.command()
@click.argument("entry_point")
@click.option("--root", "-r", default=".", help="Repository root directory")
@click.option("--depth", "-d", default=2, type=int, help="Import resolution depth")
@click.option("--semantic/--no-semantic", default=True, help="Include semantic relatives")
@click.option("--format", "-o", type=click.Choice(["xml", "compact"]), default="compact")
def discover(entry_point: str, root: str, depth: int, semantic: bool, format: str):
    """Discover related code from an entry point.

    Example: ospack discover src/main.py --depth 3
    """
    from .workflows import get_workflows

    root_path = Path(root).resolve()
    if not root_path.exists():
        err = ErrorResponse.create(
            code=ErrorCode.INVALID_PATH,
            error=f"Root directory not found: {root}",
        )
        console.print(_format_error(err, format))
        raise SystemExit(1)

    workflows = get_workflows(str(root_path))

    with console.status(f"Discovering related code from {entry_point}..."):
        result = workflows.discover_related(
            entry_point=entry_point,
            depth=depth,
            include_semantic=semantic,
        )

    # Custom formatting for discover (dependency graph is unique)
    rel_entry = result.entry_point.relative_to(root_path)

    if format == "xml":
        lines = [f'<discovery entry="{rel_entry}" total_files="{result.total_files}">']
        lines.append("  <dependency_graph>")
        for rel_path, node in result.dependency_graph.items():
            imports_list = ",".join(str(p.relative_to(root_path)) for p in node.imports)
            importers_list = ",".join(str(p.relative_to(root_path)) for p in node.imported_by)
            lines.append(
                f'    <file path="{rel_path}" imports="{imports_list}" '
                f'imported_by="{importers_list}"/>'
            )
        lines.append("  </dependency_graph>")
        if result.semantic_relatives:
            lines.append("  <semantic_relatives>")
            for ctx in result.semantic_relatives:
                rel_ctx = ctx.file_path.relative_to(root_path)
                name_attr = f' name="{ctx.name}"' if ctx.name else ""
                lines.append(
                    f'    <chunk path="{rel_ctx}" '
                    f'lines="{ctx.start_line}-{ctx.end_line}"{name_attr}/>'
                )
            lines.append("  </semantic_relatives>")
        lines.append("</discovery>")
        console.print("\n".join(lines))
    else:
        lines = [
            f"# Discovery: {rel_entry}",
            f"Total files in graph: {result.total_files}",
            "",
            "## Dependency Graph",
        ]
        for rel_path, node in result.dependency_graph.items():
            lines.append(f"### {rel_path}")
            if node.imports:
                lines.append("  Imports:")
                for imp in node.imports:
                    lines.append(f"    - {imp.relative_to(root_path)}")
            if node.imported_by:
                lines.append("  Imported by:")
                for imp in node.imported_by:
                    lines.append(f"    - {imp.relative_to(root_path)}")
            lines.append("")

        if result.semantic_relatives:
            lines.append("## Semantic Relatives")
            for ctx in result.semantic_relatives:
                rel_ctx = ctx.file_path.relative_to(root_path)
                name_info = f" ({ctx.name})" if ctx.name else ""
                lines.append(f"- {rel_ctx}:{ctx.start_line}-{ctx.end_line}{name_info}")

        console.print("\n".join(lines))


@main.command()
@click.argument("file")
@click.option("--function", "-f", help="Specific function being changed")
@click.option("--root", "-r", default=".", help="Repository root directory")
@click.option("--depth", "-d", default=3, type=int, help="Max depth for transitive analysis")
@click.option("--format", "-o", type=click.Choice(["xml", "compact"]), default="xml")
def impact(file: str, function: str | None, root: str, depth: int, format: str):
    """Analyze impact of changes to a file.

    Example: ospack impact src/utils.py --function helper
    """
    from .workflows import format_impact_result, get_workflows

    root_path = Path(root).resolve()
    if not root_path.exists():
        err = ErrorResponse.create(
            code=ErrorCode.INVALID_PATH,
            error=f"Root directory not found: {root}",
        )
        console.print(_format_error(err, format))
        raise SystemExit(1)

    workflows = get_workflows(str(root_path))

    with console.status(f"Analyzing impact of changes to {file}..."):
        result = workflows.analyze_impact(
            file=file,
            function=function,
            max_depth=depth,
        )

    output = format_impact_result(result, root_path, format=format)
    console.print(output)


# ============================================================================
# MCP Server Command
# ============================================================================


@main.command()
def mcp():
    """Start MCP server for AI agent integration.

    Runs ospack as a Model Context Protocol (MCP) server over stdio.
    AI agents like Claude can then call ospack tools directly.

    INSTALL:
      claude mcp add ospack -- ospack mcp

    TOOLS EXPOSED:
      ospack_pack   - Pack code context (focus + semantic search)
      ospack_search - Lightweight semantic search
      ospack_index  - Build/update search index
    """
    from .mcp_server import mcp as mcp_server
    mcp_server.run()


if __name__ == "__main__":
    main()
