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
from .trigram import get_trigram_index
from .workflows import get_workflows

mcp = FastMCP(
    "ospack",
    instructions=(
        "Semantic code context packer for AI assistants. "
        "Use ospack_map for a birds-eye view of repo structure, ospack_pack to gather relevant code context, "
        "ospack_search to find code by concept, ospack_grep for exact/regex pattern search (preserves punctuation), "
        "ospack_index to build the search index, ospack_probe to find missing symbols in packed context, "
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

    This is the main tool for gathering code context. It combines import resolution (following
    imports from a file) with semantic search (finding code by concept). The result is formatted
    code ready to use as context for understanding, debugging, or modifying code.

    STRATEGIES:
    1. IMPORT RESOLUTION (focus): Starting from a file, follow import statements to find
       dependencies. Use when you have a specific file and need to understand what it uses.
       Example: focus="src/auth.py" finds auth.py plus all files it imports.

    2. SEMANTIC SEARCH (query): Find code matching a natural language description.
       Use when searching for functionality by concept rather than file location.
       Example: query="JWT token validation" finds relevant code across the codebase.

    You can use both together: focus finds dependencies, query adds semantically related code.

    WHEN TO USE:
    - Need context about specific functionality
    - Investigating how features work
    - Understanding code dependencies before making changes

    WHEN NOT TO USE:
    - Just reading a single known file (use file read instead - faster)
    - Making simple edits (use direct file edit)
    - Quick exploration (use ospack_search first to find relevant files)

    OUTPUT FORMAT:
    - "compact" (default): Markdown-formatted code blocks with file headers
    - "xml": Structured XML with <file> tags, optimized for Claude's XML parsing

    TOKEN SAVINGS:
    - skeleton=True (default): Collapses non-focus file function bodies to just signatures.
      Typically reduces tokens by 60-70% while preserving API structure.
    - focus_only=True: Skips semantic search entirely. Much faster, no index needed.

    Args:
        root: Repository root directory. Must be absolute path.
        focus: Entry point file for import resolution, relative to root. Example: "src/auth.py"
        query: Natural language search query. Example: "database connection pooling"
        max_files: Maximum files to include (default: 10). Use 3-5 for focused context, 15-20 for broad.
        import_depth: Levels of imports to follow from focus file (default: 2). 1=direct only, 3+=deep tree.
        format: Output format - "compact" (markdown) or "xml" (structured). Default: "compact"
        focus_only: Skip semantic search, only use import resolution. Much faster for large repos.
        skeleton: Collapse imported file bodies to signatures only (default: True). Saves 60-70% tokens.

    Returns:
        Packed code context as formatted string. Returns error message if root doesn't exist
        or neither focus nor query is specified.
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

    Creates a compressed tree-view showing directory structure with class names, function
    signatures, and docstrings - no implementation details. Methods are indented under their
    parent classes for visual hierarchy. This is the fastest way to understand a codebase structure.

    Use this FIRST when exploring an unfamiliar codebase to understand where files live and
    what they contain, before diving into specific files with ospack_pack.

    WHEN TO USE:
    - First thing when starting work on an unfamiliar repo
    - To understand overall project structure ("what modules exist?")
    - To find where specific functionality might live
    - Before using ospack_pack to know what files to focus on

    WHEN NOT TO USE:
    - Need actual code implementation (use ospack_pack)
    - Searching for specific functionality (use ospack_search)

    OUTPUT FORMAT:
    Returns an indented tree structure like:
        src/
          auth/
            login.py
              class LoginHandler
                def authenticate(username, password) -> bool
                def logout(session_id)
              def hash_password(pwd) -> str

    Args:
        root: Repository root directory. Must be absolute path.
        include_signatures: Include function/class signatures (default: True). Set False for
            just file names when you only need directory structure.
        max_sigs: Maximum signatures per file (default: None = unlimited). Use 10-30 for large
            repos to prevent context overflow. Files with more signatures show "[N more...]".

    Returns:
        Tree-formatted string showing directory structure with optional signatures.
        Returns error message if root directory doesn't exist.
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
    min_score: float | None = None,
) -> list[dict]:
    """Search codebase semantically using natural language.

    Finds code chunks that match the conceptual meaning of your query, even if they
    don't contain the exact keywords. Uses BM25+ keyword search with optional reranking
    for best relevance. This is lighter weight than ospack_pack - use it for quick
    exploration before committing to full context packing.

    WHEN TO USE:
    - Exploring unfamiliar codebase ("where is authentication handled?")
    - Finding where functionality is implemented
    - Discovering related code before using ospack_pack

    WHEN NOT TO USE:
    - Know exact file/function name (use file read instead)
    - Need full context with imports (use ospack_pack)
    - Need exact pattern match with punctuation (use ospack_grep)

    OUTPUT FORMAT:
    Returns a list of dicts, each containing: file_path (relative), name (function/class
    name if available), start_line, end_line, score (0-25 typical range, higher is better),
    and content (truncated to 500 chars for overview).

    Args:
        root: Repository root directory. Must be absolute path.
        query: Natural language description of what you're looking for. Be specific -
            "OAuth token refresh logic" works better than just "auth".
        limit: Maximum results to return (default: 10, max recommended: 20).
        min_score: Filter results below this BM25+ score. Typical good matches score 8-25.
            Use 5-8 for broad results, 10+ for high-confidence matches only. Default: None
            (no filtering). Set this when you're getting too many low-relevance results.

    Returns:
        List of matches. Each match has: file_path, name, start_line, end_line, score, content.
        Returns [{"error": "..."}] if root directory doesn't exist.
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
        # Get the best available score
        score = r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0)

        # Filter by min_score if specified
        if min_score is not None and score < min_score:
            continue

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
            "score": round(score, 3),
            "content": r.get("content", "")[:500],  # Truncate for overview
        })

    return clean_results


@mcp.tool()
def ospack_grep(
    root: str,
    pattern: str,
    regex: bool = False,
    limit: int = 20,
) -> list[dict]:
    """Fast exact/regex code search using trigram index.

    Unlike ospack_search (semantic/BM25), this finds EXACT patterns including punctuation
    and special characters. Uses trigram-based pre-filtering for speed - first narrows down
    candidate files using a trigram index, then verifies matches. Best for finding specific
    code patterns where you know the exact syntax.

    WHEN TO USE:
    - API calls with punctuation: ".map(", "useState(", "->>"
    - Operators and symbols: "=>", "??", "..."
    - Regex patterns: regex=True with "async\\s+function\\s+\\w+"
    - Exact strings that BM25 tokenization would break up

    WHEN NOT TO USE:
    - Fuzzy/conceptual search (use ospack_search - it understands meaning)
    - Finding code by meaning, not exact text
    - Very short patterns (<3 chars) - trigram index requires at least 3 characters

    REGEX MODE:
    When regex=True, the pattern is interpreted as a Python regular expression.
    The tool extracts literal substrings from the regex for trigram pre-filtering,
    then runs full regex matching on candidate files. Example patterns:
    - "class\\s+\\w+Service" - finds class definitions ending in Service
    - "def\\s+(get|set)_\\w+" - finds getter/setter methods
    - "TODO.*\\d{4}" - finds TODOs with year numbers

    OUTPUT FORMAT:
    Returns list of dicts with: file (relative path), line (1-indexed line number),
    match (the matched text), context (2 lines before/after for context).

    Args:
        root: Repository root directory. Must be absolute path.
        pattern: Literal string or regex pattern to search. Must be at least 3 characters
            for effective trigram filtering (shorter patterns may be slow).
        regex: If True, treat pattern as Python regex (default: False). Without this flag,
            special characters like .* are matched literally.
        limit: Maximum results to return (default: 20). Results are returned in file order.

    Returns:
        List of matches. Each match has: file, line, match, context.
        Returns [{"error": "..."}] if root directory doesn't exist.
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        return [{"error": f"Root directory does not exist: {root}"}]

    trigram_index = get_trigram_index(str(root_path))

    # Check if index is populated
    stats = trigram_index.get_stats()
    if stats["files"] == 0:
        # Build index if empty
        from .indexer import get_indexer
        indexer = get_indexer(str(root_path))
        indexer.build_index()

    results = trigram_index.search(pattern, regex=regex, limit=limit)

    # Make paths relative
    clean_results = []
    for r in results:
        file_path = Path(r["file"])
        try:
            rel_path = str(file_path.relative_to(root_path))
        except ValueError:
            rel_path = str(file_path)

        clean_results.append({
            "file": rel_path,
            "line": r["line"],
            "match": r["match"],
            "context": r["context"],
        })

    return clean_results


@mcp.tool()
def ospack_index(
    root: str,
    force: bool = False,
) -> dict:
    """Build or update the semantic search index for a repository.

    Creates a BM25+ keyword index of all code chunks to enable semantic search. Chunks are
    extracted using tree-sitter parsing (functions, classes, methods). The index runs
    incrementally by default - only processing files that changed since last indexing.

    Index storage: ~/.ospack/index/{repo-hash}/ - persists between sessions and is keyed
    by repository path hash. Each repo gets its own isolated index.

    WHEN TO USE:
    - First time using ospack on a new repository
    - After major refactoring or branch switches with many file changes
    - If search results seem stale or missing recent code
    - Use force=True if index seems corrupted

    WHEN NOT TO USE:
    - Before every search/pack (index auto-updates on search if needed)
    - After small file changes (handled incrementally and automatically)
    - Just want to search (ospack_search builds index automatically if missing)

    PERFORMANCE:
    - Initial indexing: ~1-3 seconds for small repos (<200 files), ~5-10s for medium repos
    - Incremental updates: Usually <1 second (only processes changed files)
    - force=True rebuilds everything from scratch

    OUTPUT FORMAT:
    Returns dict with: chunks_indexed (number of code chunks), time_taken (seconds),
    index_path (where index is stored), status ("rebuilt", "updated", or "up_to_date").

    Args:
        root: Repository root directory. Must be absolute path.
        force: Rebuild index from scratch, ignoring cached state (default: False).
            Use when index seems corrupted or after major branch switches.

    Returns:
        Stats dict with chunks_indexed, time_taken, index_path, status.
        Returns {"error": "..."} if root directory doesn't exist.
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

    This tool enables "Chain-of-Thought Retrieval" - instead of hoping one-shot retrieval
    gets everything right, you can iteratively discover and fetch missing dependencies.
    Pass the output from ospack_pack and get suggestions for what else to fetch.

    HOW IT WORKS:
    1. Parses the provided code to extract all symbol references (function calls,
       class instantiations, type annotations, imports)
    2. Identifies which symbols are used but not defined in the content
    3. Searches the codebase to find where these missing symbols are defined
    4. Returns actionable suggestions for follow-up ospack_pack or ospack_search calls

    WHEN TO USE:
    - After calling ospack_pack, to find symbols that were referenced but not included
    - When you notice undefined references in the context you received
    - To iteratively build complete context for complex features
    - Before making changes that depend on understanding related code

    WHEN NOT TO USE:
    - Content is already complete (no missing symbols)
    - You know exactly what files you need (use ospack_pack directly)

    WORKFLOW EXAMPLE:
    1. ospack_pack(focus="auth.py") -> get auth module context
    2. ospack_probe(content=<packed_output>) -> finds missing "User", "Token" classes
    3. ospack_pack(query="User class definition") -> fetch missing pieces
    4. Repeat until ospack_probe returns no missing symbols

    OUTPUT FORMAT:
    Returns dict with:
    - missing_symbols: List of symbol names that are referenced but not defined
    - suggestions: List of dicts with {symbol, file (if found), suggestion (ospack command)}
    - defined_symbols: List of symbols that ARE defined in the content (for reference)
    - message: Human-readable summary

    Args:
        root: Repository root directory. Must be absolute path.
        content: The packed code content to analyze. Pass the full output from ospack_pack.
        limit: Maximum number of missing symbol suggestions to return (default: 10).
            Higher values find more symbols but increase processing time.

    Returns:
        Dict with missing_symbols, suggestions, defined_symbols, and message.
        Returns {"error": "..."} if root directory doesn't exist.
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

    This is REVERSE dependency analysis - finds who USES this code, not what this code uses.
    Essential before refactoring to understand the blast radius and avoid breaking consumers.
    Unlike ospack_pack (which finds what a file depends ON), this finds what depends ON a file.

    HOW IT WORKS:
    1. Scans all imports in the repository to build a reverse dependency graph
    2. Finds files that directly import the target file
    3. Follows the import chain transitively up to max_depth levels
    4. Uses fuzzy matching to catch dependency injection and dynamic imports

    WHEN TO USE:
    - Before changing a function signature ("who calls login()?")
    - Before renaming or moving a file
    - To understand blast radius before a refactor
    - Before deprecating an API
    - To find all tests that cover a module

    WHEN NOT TO USE:
    - To understand what a file depends ON (use ospack_pack with focus instead)
    - For simple one-file changes with no public API

    TERMINOLOGY:
    - directly_affected: Files that have an explicit import of the target file
    - transitively_affected: Files that import files that import the target (indirect deps)

    OUTPUT FORMAT:
    Returns dict with:
    - target: The file being analyzed (relative path)
    - function: The specific function being changed (if specified)
    - directly_affected: List of file paths that directly import the target
    - transitively_affected: List of file paths affected through dependency chain
    - total_affected: Total count of all affected files

    Args:
        root: Repository root directory. Must be absolute path.
        file: Path to the file being changed, relative to root. Example: "src/auth/login.py"
        function: Optional specific function being changed. Included in output for context
            but doesn't change the analysis (impact is file-level).
        max_depth: How many levels of transitive dependents to include (default: 3).
            1=direct importers only, 2=importers of importers, etc.

    Returns:
        Dict with target, function, directly_affected, transitively_affected, total_affected.
        Returns {"error": "..."} if root or file doesn't exist.
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
    """Dry-run pack to check token costs BEFORE loading full content.

    Returns a detailed breakdown of what would be packed and how many tokens it would consume,
    WITHOUT returning the actual code. Use this to make informed decisions about context
    budget before committing to a potentially large ospack_pack call.

    Token estimation uses ~4 characters per token (GPT-style tokenization). Results are sorted
    by token count so you can see which files are consuming the most budget.

    WHEN TO USE:
    - Before packing a large directory or broad query
    - When running low on context window budget
    - To decide between full content vs skeleton mode
    - To identify which files are consuming the most tokens
    - To compare different packing strategies

    WHEN NOT TO USE:
    - You already know the files are small
    - You're doing quick exploration (just use ospack_pack)

    WORKFLOW EXAMPLE:
    1. ospack_audit(focus="src/core") -> "12,500 tokens, 15 files"
    2. Too large? ospack_audit(focus="src/core", skeleton=True) -> "4,200 tokens"
    3. Acceptable? ospack_pack(focus="src/core", skeleton=True)

    RECOMMENDATION THRESHOLDS:
    - <8,000 tokens: "Compact result - safe to pack"
    - 8,000-15,000 tokens: "Moderate size - consider skeleton mode"
    - >15,000 tokens: "Large result - reduce max_files or use skeleton mode"

    OUTPUT FORMAT:
    Returns dict with:
    - total_tokens: Estimated total token count (~4 chars/token)
    - total_files: Number of files that would be included
    - total_lines: Total lines of code
    - files: List of {file, tokens, lines, reason} sorted by tokens descending
    - skeleton_mode: Whether skeleton mode was simulated
    - recommendation: Human-readable suggestion based on token count
    - truncated: Whether results were truncated due to limits

    Args:
        root: Repository root directory. Must be absolute path.
        focus: Entry point file for import resolution, relative to root.
        query: Natural language search query.
        max_files: Maximum files to include (default: 10).
        import_depth: Levels of imports to follow from focus file (default: 2).
        skeleton: Simulate skeleton mode (signatures only) for token estimate (default: False).
            Use True to see how much skeleton mode would save.

    Returns:
        Dict with token breakdown, file list, and recommendation.
        Returns {"error": "..."} if root doesn't exist or neither focus nor query specified.
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
