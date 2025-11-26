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

import re
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .indexer import get_indexer
from .mapper import generate_repo_map
from .packer import Packer, format_output, _estimate_tokens
from .workflows import get_workflows

mcp = FastMCP(
    "ospack",
    instructions=(
        "Semantic code context packer for AI assistants. "
        "Use ospack_map for a birds-eye view of repo structure, ospack_pack to gather relevant code context, "
        "ospack_search to find code by concept, ospack_index to build the search index, "
        "ospack_probe to find missing symbols in packed context, "
        "ospack_impact to find files affected by changes (reverse dependency analysis), "
        "and ospack_audit to check token costs before packing (dry-run)."
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
    focus_only: bool = False,
    skeleton: bool = True,
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
        focus_only: Skip semantic search, only use import resolution (FAST for large repos)
        skeleton: Collapse imported file bodies to signatures only (default: True, saves tokens)

    Returns:
        Packed code context as formatted string
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return f"Error: Root directory does not exist: {root}"

    # Skip query when focus_only is set (avoids expensive index building)
    effective_query = None if focus_only else query

    packer = Packer(str(root_path))

    result = packer.pack(
        focus=focus,
        query=effective_query,
        max_files=max_files,
        max_chunks=15,
        depth=import_depth,
        rerank=True,
        hybrid=True,
        chunk_mode=not focus_only,  # Full files when focus_only, chunks otherwise
        skeletonize=skeleton,  # Collapse non-focus function bodies to save tokens
    )

    output_format = format if focus_only else "chunks"
    return format_output(result, format=output_format, root_dir=root_path)


@mcp.tool()
def ospack_map(
    root: str,
    include_signatures: bool = True,
    max_sigs: int | None = None,
) -> str:
    """Generate a structural map of the repository.

    Creates a compressed tree-view showing the directory structure with
    class names and function signatures - no implementation details.
    Methods are indented under their parent classes for visual hierarchy.

    This gives you a "birds-eye view" of the codebase before diving into
    specific files. Use this FIRST when exploring an unfamiliar codebase
    to understand where files live and what they contain.

    WHEN TO USE:
    - First thing when starting work on an unfamiliar repo
    - To understand overall project structure
    - To find where specific functionality might live
    - Before using ospack_pack to know what files to focus on

    Args:
        root: Repository root directory (required)
        include_signatures: Include function/class signatures (default: True)
        max_sigs: Maximum signatures per file to prevent context overflow (default: None = unlimited)

    Returns:
        Tree-formatted string showing directory structure with signatures
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return f"Error: Root directory does not exist: {root}"

    return generate_repo_map(
        root_path,
        format="tree",
        include_signatures=include_signatures,
        max_sigs=max_sigs,
    )


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


# Common built-in symbols to ignore when detecting missing symbols
BUILTIN_SYMBOLS = {
    # Python builtins
    "print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "range", "enumerate", "zip", "map", "filter", "sorted", "reversed", "any", "all",
    "min", "max", "sum", "abs", "round", "open", "isinstance", "hasattr", "getattr",
    "setattr", "type", "super", "property", "staticmethod", "classmethod", "None",
    "True", "False", "self", "cls", "args", "kwargs",
    # JavaScript/TypeScript builtins
    "console", "window", "document", "Promise", "Array", "Object", "String", "Number",
    "Boolean", "Math", "JSON", "Date", "Error", "Map", "Set", "undefined", "null",
    "this", "async", "await", "export", "import", "require", "module", "default",
    # Common variable names (often false positives)
    "i", "j", "k", "x", "y", "n", "err", "error", "result", "data", "value", "key",
    "item", "items", "name", "path", "file", "content", "msg", "message", "text",
}

# Patterns to extract symbol references from code
SYMBOL_PATTERNS = [
    # Function calls: func(...)
    re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('),
    # Class instantiation: new Foo(...)
    re.compile(r'\bnew\s+([A-Z][A-Za-z0-9_]*)\s*\('),
    # Type annotations: : Type, -> Type
    re.compile(r'[:\-]\s*>?\s*([A-Z][A-Za-z0-9_]*)'),
    # Generic types: List[Foo], Dict[str, Bar]
    re.compile(r'\[([A-Z][A-Za-z0-9_]*)'),
    # Attribute access on known patterns: Foo.bar
    re.compile(r'\b([A-Z][A-Za-z0-9_]+)\.[a-z_]'),
    # Import statements (to identify what's imported but maybe not defined)
    re.compile(r'from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import'),
    re.compile(r'import\s+([A-Za-z_][A-Za-z0-9_]*)'),
]


def _extract_symbols_from_content(content: str) -> set[str]:
    """Extract symbol references from code content."""
    symbols = set()
    for pattern in SYMBOL_PATTERNS:
        for match in pattern.finditer(content):
            symbol = match.group(1)
            # Filter out builtins and very short names
            if symbol not in BUILTIN_SYMBOLS and len(symbol) > 2:
                symbols.add(symbol)
    return symbols


def _extract_definitions_from_content(content: str) -> set[str]:
    """Extract symbol definitions from code content."""
    definitions = set()

    # Python definitions
    for match in re.finditer(r'\b(?:def|class|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)', content):
        definitions.add(match.group(1))

    # JavaScript/TypeScript definitions
    for match in re.finditer(r'\b(?:function|class|interface|type|enum)\s+([A-Za-z_][A-Za-z0-9_]*)', content):
        definitions.add(match.group(1))

    # Variable/const declarations (rough approximation)
    for match in re.finditer(r'\b(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=', content):
        definitions.add(match.group(1))

    # Python assignments at module level (rough)
    for match in re.finditer(r'^([A-Z][A-Za-z0-9_]*)\s*=', content, re.MULTILINE):
        definitions.add(match.group(1))

    return definitions


@mcp.tool()
def ospack_probe(
    root: str,
    content: str,
    limit: int = 10,
) -> dict:
    """Analyze packed context for missing symbols and suggest follow-up queries.

    This tool enables "Chain-of-Thought Retrieval" - instead of hoping one-shot
    retrieval gets everything right, you can iteratively discover and fetch
    missing dependencies.

    HOW IT WORKS:
    1. Extracts all symbol references from the provided code content
    2. Identifies which symbols are used but not defined in the content
    3. Searches the codebase to find where these missing symbols are defined
    4. Returns suggestions for follow-up ospack_pack or ospack_search calls

    WHEN TO USE:
    - After calling ospack_pack, to find symbols that were referenced but not included
    - When you notice undefined references in the context you received
    - To iteratively build complete context for complex features

    WORKFLOW EXAMPLE:
    1. Call ospack_pack(focus="auth.py") -> get auth module context
    2. Call ospack_probe(content=<packed_content>) -> find missing "User", "Token" classes
    3. Call ospack_pack(query="User class definition") -> fetch missing pieces
    4. Repeat until you have complete context

    Args:
        root: Repository root directory
        content: The packed code content to analyze for missing symbols
        limit: Maximum number of missing symbol suggestions to return

    Returns:
        Dict with:
        - missing_symbols: List of symbols used but not defined
        - suggestions: List of suggested follow-up queries
        - defined_symbols: List of symbols defined in the content (for reference)
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return {"error": f"Root directory does not exist: {root}"}

    # Extract symbols referenced and defined in the content
    referenced = _extract_symbols_from_content(content)
    defined = _extract_definitions_from_content(content)

    # Find missing symbols (referenced but not defined)
    missing = referenced - defined - BUILTIN_SYMBOLS

    if not missing:
        return {
            "missing_symbols": [],
            "suggestions": [],
            "defined_symbols": list(defined)[:20],
            "message": "No missing symbols detected. Context appears complete."
        }

    suggestions = []

    # Use semantic search (BM25 + dense) to find potential matches
    indexer = get_indexer(str(root_path))
    if indexer._table is not None or indexer.bm25_path.exists():
        for symbol in list(missing)[:limit]:
            results = indexer.search(f"{symbol} definition", limit=1, rerank=False, hybrid=True)
            if results:
                r = results[0]
                file_path = Path(r["file_path"])
                try:
                    rel_path = str(file_path.relative_to(root_path))
                except ValueError:
                    rel_path = str(file_path)
                suggestions.append({
                    "symbol": symbol,
                    "file": rel_path,
                    "name": r.get("name", ""),
                    "suggestion": f"ospack_pack(focus='{rel_path}')"
                })
            else:
                suggestions.append({
                    "symbol": symbol,
                    "suggestion": f"ospack_search(query='{symbol} definition')"
                })
    else:
        # No index available, suggest search for all
        for symbol in list(missing)[:limit]:
            suggestions.append({
                "symbol": symbol,
                "suggestion": f"ospack_search(query='{symbol} definition')"
            })

    return {
        "missing_symbols": list(missing)[:limit],
        "suggestions": suggestions,
        "defined_symbols": list(defined)[:20],
        "message": f"Found {len(missing)} potentially missing symbols. Use the suggestions to fetch their definitions."
    }


@mcp.tool()
def ospack_impact(
    root: str,
    file: str,
    function: str | None = None,
    max_depth: int = 3,
) -> dict:
    """Find all files that would be affected by changes to a file/function.

    This is REVERSE dependency analysis - finds who USES this code,
    not what this code uses. Essential before refactoring to avoid breaking
    consumers of an API.

    HOW IT WORKS:
    1. Builds reverse dependency graph by scanning all imports in the repo
    2. Finds files that directly import the target file
    3. Follows the chain transitively up to max_depth levels
    4. Optionally uses fuzzy matching to catch DI framework references

    WHEN TO USE:
    - Before changing a function signature ("who calls login()?")
    - Before renaming or moving a file
    - To understand the blast radius of a refactor
    - Before deprecating an API

    WHEN NOT TO USE:
    - To understand what a file depends ON (use ospack_pack instead)
    - For simple one-file changes with no public API

    Args:
        root: Repository root directory
        file: Path to the file being changed (relative to root)
        function: Optional specific function being changed (for context)
        max_depth: How many levels of transitive dependents to include (default: 3)

    Returns:
        Dict with:
        - target: The file being analyzed
        - function: The function being changed (if specified)
        - directly_affected: Files that directly import/use the target
        - transitively_affected: Files affected through dependency chain
        - total_affected: Total count of affected files
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return {"error": f"Root directory does not exist: {root}"}

    workflows = get_workflows(str(root_path))

    try:
        result = workflows.analyze_impact(
            file=file,
            function=function,
            max_depth=max_depth,
            fuzzy_matching=True,
        )
    except FileNotFoundError as e:
        return {"error": str(e)}

    # Convert paths to relative strings for cleaner output
    def rel_path(p: Path) -> str:
        try:
            return str(p.relative_to(root_path))
        except ValueError:
            return str(p)

    return {
        "target": rel_path(result.target),
        "function": result.function,
        "directly_affected": [rel_path(p) for p in result.directly_affected],
        "transitively_affected": [rel_path(p) for p in result.transitively_affected],
        "total_affected": result.total_affected,
    }


@mcp.tool()
def ospack_audit(
    root: str,
    focus: str | None = None,
    query: str | None = None,
    max_files: int = 10,
    import_depth: int = 2,
    skeleton: bool = False,
) -> dict:
    """Dry-run pack to check token costs BEFORE loading content.

    Returns a detailed breakdown of what would be packed and how many tokens
    it would consume, WITHOUT returning the actual code. Use this to make
    informed decisions about context budget.

    WHEN TO USE:
    - Before packing a large directory or query
    - When low on context window budget
    - To decide between full content vs skeleton mode
    - To identify which files are consuming the most tokens

    WORKFLOW EXAMPLE:
    1. ospack_audit(focus="src/core") -> "12,500 tokens, 15 files"
    2. If too large: ospack_audit(focus="src/core", skeleton=True) -> "4,200 tokens"
    3. If acceptable: ospack_pack(focus="src/core", skeleton=True)

    Args:
        root: Repository root directory
        focus: Entry point file for import resolution (relative to root)
        query: Search query
        max_files: Maximum files to include (default: 10)
        import_depth: Levels of imports to follow (default: 2)
        skeleton: Simulate skeleton mode (signatures only) for token estimate

    Returns:
        Dict with:
        - total_tokens: Estimated total token count
        - total_files: Number of files that would be included
        - files: List of files with individual token costs (sorted by size)
        - recommendation: Suggestion based on token count
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return {"error": f"Root directory does not exist: {root}"}

    if not focus and not query:
        return {"error": "Must specify focus and/or query"}

    packer = Packer(str(root_path))

    # Run pack to get file list
    result = packer.pack(
        focus=focus,
        query=query,
        max_files=max_files,
        max_chunks=20,
        depth=import_depth,
        rerank=True,
        hybrid=True,
        chunk_mode=False,  # Get full files for accurate count
        skeletonize=skeleton,
    )

    # Build file breakdown
    file_breakdown = []
    for f in result.files:
        tokens = _estimate_tokens(f.content)
        try:
            rel_path = str(f.path.relative_to(root_path))
        except ValueError:
            rel_path = str(f.path)
        file_breakdown.append({
            "file": rel_path,
            "tokens": tokens,
            "lines": f.content.count("\n") + 1,
            "reason": f.reason,
        })

    # Sort by token count (heaviest first)
    file_breakdown.sort(key=lambda x: x["tokens"], reverse=True)

    # Generate recommendation
    total_tokens = result.total_tokens
    if total_tokens > 15000:
        if skeleton:
            recommendation = f"Very large ({total_tokens} tokens). Consider reducing max_files or focusing on specific files."
        else:
            recommendation = f"Large result ({total_tokens} tokens). Try skeleton=True to reduce to ~{total_tokens // 3} tokens."
    elif total_tokens > 8000:
        if skeleton:
            recommendation = f"Moderate size ({total_tokens} tokens). Should fit in most contexts."
        else:
            recommendation = f"Moderate size ({total_tokens} tokens). Consider skeleton=True if context is limited."
    else:
        recommendation = f"Compact result ({total_tokens} tokens). Safe to pack."

    return {
        "total_tokens": total_tokens,
        "total_files": len(result.files),
        "total_lines": result.total_lines,
        "files": file_breakdown,
        "skeleton_mode": skeleton,
        "recommendation": recommendation,
        "truncated": result.truncation.truncated if result.truncation else False,
    }


def main():
    """Run the MCP server with stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
