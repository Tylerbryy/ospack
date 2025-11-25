"""MCP Server for ospack - exposes semantic code packing to AI agents.

This module provides an MCP (Model Context Protocol) server that exposes
ospack's functionality as tools that AI agents can call directly.

Usage:
    # Run as standalone server (stdio transport)
    python -m ospack.mcp_server

    # Add to Claude Code
    claude mcp add ospack -- python -m ospack.mcp_server
"""

from __future__ import annotations

import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .indexer import get_indexer
from .packer import Packer, format_output

mcp = FastMCP(
    "ospack",
    instructions=(
        "Semantic code context packer for AI assistants. "
        "Use ospack_pack to gather relevant code context, ospack_search "
        "to find code by concept, ospack_index to build the search index."
    ),
)


@mcp.tool()
def ospack_pack(
    root: str,
    focus: str | None = None,
    query: str | None = None,
    max_files: int = 10,
    import_depth: int = 2,
    format: str = "compact",
) -> str:
    """Pack relevant code context for AI assistants.

    Combines two strategies to gather the most relevant code:

    1. IMPORT RESOLUTION (focus): Starting from a file, follow import
       statements to find its dependencies. Use when you have a specific
       file and need to understand what it uses.

    2. SEMANTIC SEARCH (query): Find code matching a natural language
       description. Use when searching for functionality by concept.

    You can use both together: focus finds dependencies, query adds related code.

    WHEN TO USE:
    - Need context about specific functionality
    - Investigating how features work
    - Understanding code dependencies

    WHEN NOT TO USE:
    - Just reading a single known file (use file read instead)
    - Making simple edits (use direct file edit)

    Args:
        root: Repository root directory (required)
        focus: Entry point file for import resolution (relative to root)
        query: Natural language search query
        max_files: Maximum files to include (default: 10)
        import_depth: Levels of imports to follow (default: 2)
        format: Output format - "compact" or "xml" (default: compact)

    Returns:
        Packed code context as formatted string
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return f"Error: Root directory does not exist: {root}"

    packer = Packer(str(root_path))

    result = packer.pack(
        focus=focus,
        query=query,
        max_files=max_files,
        max_chunks=15,
        depth=import_depth,
        rerank=True,
        hybrid=True,
        chunk_mode=True,  # Return chunks, not full files - lightweight
    )

    return format_output(result, format="chunks", root_dir=root_path)


@mcp.tool()
def ospack_search(
    root: str,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """Search codebase semantically using natural language.

    Finds code chunks that match the conceptual meaning of your query,
    even if they don't contain the exact keywords. Uses AI embeddings
    to understand code semantics.

    WHEN TO USE:
    - Exploring unfamiliar codebase
    - Finding where functionality is implemented
    - Discovering related code

    WHEN NOT TO USE:
    - Know exact file/function name (use file read)
    - Need full context with imports (use ospack_pack)

    Args:
        root: Repository root directory
        query: Natural language description of what you're looking for
        limit: Maximum results to return (default: 10)

    Returns:
        List of matches with file_path, content, score, start_line, end_line, name
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return [{"error": f"Root directory does not exist: {root}"}]

    indexer = get_indexer(str(root_path))

    # Build index if needed
    indexer.build_index()

    # Search with hybrid + reranking for best quality
    results = indexer.search(
        query=query,
        limit=limit,
        rerank=True,
        hybrid=True,
    )

    # Clean up results for MCP output
    clean_results = []
    for r in results:
        # Make file_path relative to root for readability
        file_path = Path(r["file_path"])
        try:
            rel_path = str(file_path.relative_to(root_path))
        except ValueError:
            rel_path = str(file_path)

        clean_results.append({
            "file_path": rel_path,
            "name": r.get("name", ""),
            "start_line": r.get("start_line", 0),
            "end_line": r.get("end_line", 0),
            "score": round(r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0), 3),
            "content": r.get("content", "")[:500],  # Truncate for overview
        })

    return clean_results


@mcp.tool()
def ospack_index(
    root: str,
    force: bool = False,
) -> dict:
    """Build or update the semantic search index for a repository.

    Creates embeddings for all code chunks to enable semantic search.
    Runs incrementally - only processes changed files unless force=True.

    The index is stored in ~/.ospack/index/{repo-hash}/ and persists
    between sessions.

    WHEN TO USE:
    - First time using ospack on a repository
    - After significant code changes
    - If search results seem stale

    WHEN NOT TO USE:
    - Index already exists and code hasn't changed
    - Just want to search (index builds automatically if needed)

    Args:
        root: Repository root directory
        force: Rebuild from scratch (default: False)

    Returns:
        Stats dict with chunks_indexed, time_taken
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return {"error": f"Root directory does not exist: {root}"}

    indexer = get_indexer(str(root_path))

    start_time = time.time()
    chunks_indexed = indexer.build_index(force=force)
    elapsed = time.time() - start_time

    return {
        "chunks_indexed": chunks_indexed,
        "time_taken": round(elapsed, 2),
        "index_path": str(indexer.storage_dir),
        "status": "rebuilt" if force else ("updated" if chunks_indexed > 0 else "up_to_date"),
    }


def main():
    """Run the MCP server with stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
