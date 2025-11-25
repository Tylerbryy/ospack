"""CLI interface for ospack."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .packer import Packer, format_output

console = Console()


@click.group()
@click.version_option()
def main():
    """ospack - Semantic context packer for AI-assisted coding."""
    pass


@main.command()
@click.option("--focus", "-f", help="Focus file for import resolution")
@click.option("--query", "-q", help="Semantic search query")
@click.option("--max-files", "-m", default=10, type=int, help="Maximum files to include")
@click.option("--max-chunks", "-c", default=20, type=int, help="Maximum chunks in chunk mode")
@click.option("--max-tokens", "-t", type=int, help="Token budget (approximate)")
@click.option("--min-score", "-s", default=0.0, type=float, help="Minimum relevance score")
@click.option("--depth", "-d", default=2, type=int, help="Import resolution depth")
@click.option(
    "--format",
    "-o",
    type=click.Choice(["xml", "compact", "chunks"]),
    default="xml",
    help="Output format (chunks = optimized for agents)",
)
@click.option("--root", "-r", default=".", help="Repository root directory")
@click.option("--rerank/--no-rerank", default=True, help="Use cross-encoder reranking")
@click.option("--hybrid/--no-hybrid", default=True, help="Use hybrid BM25+dense search")
def pack(
    focus: str | None,
    query: str | None,
    max_files: int,
    max_chunks: int,
    max_tokens: int | None,
    min_score: float,
    depth: int,
    format: str,
    root: str,
    rerank: bool,
    hybrid: bool,
):
    """Pack relevant code context for AI assistants.

    Uses hybrid search (BM25 + semantic) with cross-encoder reranking by default.

    Use --format chunks for optimized agent context (returns code chunks instead of full files).
    """
    if not focus and not query:
        console.print("[red]Error:[/red] Must specify --focus and/or --query")
        raise SystemExit(1)

    root_path = Path(root).resolve()
    if not root_path.exists():
        console.print(f"[red]Error:[/red] Root directory not found: {root}")
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
            depth=depth,
            rerank=rerank,
            hybrid=hybrid,
            chunk_mode=chunk_mode,
        )

    if not result.files and not result.chunks:
        console.print("[yellow]No results found matching criteria[/yellow]")
        return

    output = format_output(result, format=format, root_dir=root_path)
    console.print(output)


@main.command()
@click.option("--root", "-r", default=".", help="Repository root directory")
@click.option("--force", is_flag=True, help="Force rebuild even if index exists")
def index(root: str, force: bool):
    """Build or rebuild the semantic index."""
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
@click.option("--root", "-r", default=".", help="Repository root directory")
@click.option("--limit", "-l", default=10, type=int, help="Number of results")
def search(query: str, root: str, limit: int):
    """Search the semantic index."""
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
@click.option("--root", "-r", default=".", help="Repository root directory")
def info(root: str):
    """Show information about the current index."""
    from .embedder import get_device
    from .indexer import get_indexer

    root_path = Path(root).resolve()

    console.print("[bold]ospack info[/bold]")
    console.print(f"Root: {root_path}")
    console.print(f"Device: {get_device()}")

    indexer = get_indexer(str(root_path))
    console.print(f"Index path: {indexer.db_path}")

    if indexer._table:
        try:
            count = len(indexer._table.to_pandas())
            console.print(f"Indexed chunks: {count}")
        except Exception:
            console.print("Index not loaded")


if __name__ == "__main__":
    main()
