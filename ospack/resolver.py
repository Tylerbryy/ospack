"""Auto-discovering import resolver with PageRank-based repo mapping."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import defaultdict
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from .log import get_logger


def _get_repo_hash(path: str) -> str:
    """Generate a hash for the repository path."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]

if TYPE_CHECKING:
    from tree_sitter import Node

logger = get_logger(__name__)

# Import patterns by language
# Updated regex with ^\s* to allow indented imports (inside functions/impl blocks)
IMPORT_PATTERNS = {
    "python": [
        re.compile(r"^\s*from\s+(\S+)\s+import", re.MULTILINE),
        re.compile(r"^\s*import\s+(\S+)", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r'from\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', re.MULTILINE),
        # Dynamic imports
        re.compile(r'import\s*\(\s*["\']([^"\']+)["\']\s*\)', re.MULTILINE),
    ],
    "javascript": [
        re.compile(r'from\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', re.MULTILINE),
    ],
    "rust": [
        re.compile(r"^\s*use\s+(\S+)", re.MULTILINE),
        re.compile(r"^\s*mod\s+(\S+)", re.MULTILINE),
    ],
    "go": [
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        # Capture the whole block inside () - we process it in Python code
        re.compile(r"import\s+\(([^)]+)\)", re.MULTILINE | re.DOTALL),
    ],
}

# Extension to language mapping
EXT_TO_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}

# Scoped extensions for candidate generation (language-specific)
LANG_EXTENSIONS = {
    "python": [".py"],
    "typescript": [".ts", ".tsx", ".js", ".jsx", ".d.ts"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "rust": [".rs"],
    "go": [".go"],
    "c": [".h", ".c"],
    "cpp": [".hpp", ".cpp", ".h"],
}


def get_language(file_path: Path) -> str | None:
    """Get the language for a file based on extension."""
    return EXT_TO_LANG.get(file_path.suffix.lower())


class ImportResolver:
    """Resolve imports with auto-discovery of project config."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()

        # Auto-discovered config
        self._ts_paths: dict[str, list[Path]] = {}
        self._python_roots: list[Path] = [self.root_dir]

        # AUTO-DISCOVERY: Run immediately on init
        self._scan_project_config()

    def _scan_project_config(self):
        """Auto-detect TSConfig, JSConfig, and Python layouts."""
        # 1. TypeScript / JavaScript Paths (tsconfig.json, jsconfig.json)
        for config_name in ["tsconfig.json", "jsconfig.json"]:
            config_path = self.root_dir / config_name
            if config_path.exists():
                try:
                    # Parse JSON with comments stripped
                    content = config_path.read_text(encoding="utf-8", errors="ignore")
                    # Strip single-line comments
                    content = re.sub(r"//.*", "", content, flags=re.MULTILINE)
                    # Strip multi-line comments
                    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

                    data = json.loads(content)
                    compiler_opts = data.get("compilerOptions", {})
                    base_url = compiler_opts.get("baseUrl", ".")
                    paths = compiler_opts.get("paths", {})

                    # Normalize paths relative to root
                    base_dir = (self.root_dir / base_url).resolve()

                    for alias, targets in paths.items():
                        # Remove wildcard * from alias (e.g., "@/*" -> "@/")
                        clean_alias = alias.replace("*", "")
                        clean_targets = []
                        for t in targets:
                            # Remove wildcard from target (e.g., "src/*" -> "src/")
                            t_clean = t.replace("*", "")
                            target_path = (base_dir / t_clean).resolve()
                            clean_targets.append(target_path)
                        self._ts_paths[clean_alias] = clean_targets

                    if self._ts_paths:
                        logger.debug("Loaded TS paths from %s: %s", config_name, list(self._ts_paths.keys()))
                except Exception as e:
                    logger.debug("Could not parse %s: %s", config_name, e)

        # 2. Python Src Layout Detection
        # If there is a 'src' folder containing packages, add it to roots
        src_dir = self.root_dir / "src"
        if src_dir.exists() and src_dir.is_dir():
            self._python_roots.append(src_dir)
            logger.debug("Added Python src layout: %s", src_dir)

    def _resolve_alias(self, import_path: str) -> list[Path]:
        """Try to resolve using auto-detected aliases."""
        candidates = []
        for alias, targets in self._ts_paths.items():
            if import_path.startswith(alias):
                suffix = import_path[len(alias):]
                for target_base in targets:
                    candidates.append(target_base / suffix.lstrip("/"))
        return candidates

    def _extract_imports(self, file_path: Path, content: str) -> list[str]:
        """Extract import paths from file content."""
        language = get_language(file_path)
        if not language or language not in IMPORT_PATTERNS:
            return []

        imports = []
        for pattern in IMPORT_PATTERNS[language]:
            matches = pattern.findall(content)

            # Special handling for Go multi-line blocks
            if language == "go" and "(" in pattern.pattern:
                for block in matches:
                    # Extract quoted strings inside the block
                    block_imports = re.findall(r'["\']([^"\']+)["\']', block)
                    imports.extend(block_imports)
            else:
                imports.extend(matches)

        return imports

    def _get_candidates(self, base_path: Path, language: str | None) -> Iterator[Path]:
        """Generate candidate file paths using language-specific rules."""
        # 1. Direct match (if it has extension)
        yield base_path

        # 2. Try language-specific extensions only
        valid_exts = LANG_EXTENSIONS.get(language, [])
        for ext in valid_exts:
            yield base_path.with_suffix(ext)

        # 3. Directory index files (language-specific)
        if language == "python":
            yield base_path / "__init__.py"
        elif language in ("typescript", "javascript"):
            for index in ["index.ts", "index.tsx", "index.js", "index.jsx"]:
                yield base_path / index
        elif language == "rust":
            yield base_path / "mod.rs"
        elif language == "go":
            yield base_path / "main.go"

    def _resolve_import(self, import_path: str, source_file: Path) -> Path | None:
        """Try to resolve an import path to a file."""
        language = get_language(source_file)

        # 1. Filter obvious external packages
        if import_path.startswith(("node_modules", "http")):
            return None
        if language == "python" and import_path in ("sys", "os", "typing", "__future__", "datetime", "json", "re", "pathlib"):
            return None

        candidates: list[Path] = []

        # 2. Handle TS/JS Aliases (Auto-detected from tsconfig.json)
        if language in ("typescript", "javascript"):
            alias_matches = self._resolve_alias(import_path)
            for m in alias_matches:
                candidates.extend(self._get_candidates(m, language))

        # 3. Handle Relative Imports
        if import_path.startswith("."):
            base_dir = source_file.parent

            if language == "python":
                # Python . / .. resolution
                dots = 0
                for char in import_path:
                    if char == ".":
                        dots += 1
                    else:
                        break

                # Navigate up
                for _ in range(dots - 1):
                    base_dir = base_dir.parent

                module_name = import_path[dots:]
                target = base_dir / module_name.replace(".", "/")
                candidates.extend(self._get_candidates(target, language))
            else:
                # JS/TS/Rust relative resolution
                target = base_dir / import_path
                candidates.extend(self._get_candidates(target, language))

        # 4. Handle Absolute / Package Imports
        elif not candidates:  # Only if alias didn't produce candidates
            if language == "python":
                # Try all Python roots (project root, src/, etc.)
                clean_path = import_path.replace(".", "/")
                for root in self._python_roots:
                    target = root / clean_path
                    candidates.extend(self._get_candidates(target, language))

            elif language in ("typescript", "javascript"):
                # Try simple root resolution
                candidates.extend(self._get_candidates(self.root_dir / import_path, language))

                # If path starts with @, try common alias patterns
                if import_path.startswith("@"):
                    alias_path = import_path[1:].lstrip("/")
                    # Try @/foo -> src/foo
                    candidates.extend(self._get_candidates(
                        self.root_dir / "src" / alias_path, language
                    ))
                    # Try @/foo -> /foo
                    candidates.extend(self._get_candidates(
                        self.root_dir / alias_path, language
                    ))

            else:
                # Rust/Go
                target = self.root_dir / import_path
                candidates.extend(self._get_candidates(target, language))

        # 5. Check existence
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        return None

    @lru_cache(maxsize=1000)
    def resolve_imports(self, file_path: str | Path) -> tuple[Path, ...]:
        """Resolve all imports in a file to actual file paths.

        Note: Returns tuple for lru_cache hashability.
        """
        file_path = Path(file_path).resolve()

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ()

        raw_imports = self._extract_imports(file_path, content)
        resolved = []
        seen = set()

        for imp in raw_imports:
            # Clean up imports (remove trailing semicolons, quotes, etc)
            imp = imp.strip().strip("';\"")

            res = self._resolve_import(imp, file_path)
            if res and res not in seen and res != file_path:
                resolved.append(res)
                seen.add(res)

        return tuple(resolved)

    def get_dependency_graph(
        self, entry_file: str | Path, max_depth: int = 3
    ) -> dict[Path, list[Path]]:
        """Build a dependency graph starting from an entry file."""
        entry_path = Path(entry_file).resolve()
        graph: dict[Path, list[Path]] = {}
        visited: set[Path] = set()

        def traverse(current_path: Path, depth: int):
            if depth > max_depth or current_path in visited:
                return
            if not current_path.exists():
                return

            visited.add(current_path)

            # Use cached resolver
            imports = list(self.resolve_imports(current_path))
            graph[current_path] = imports

            for imp in imports:
                traverse(imp, depth + 1)

        traverse(entry_path, 0)
        return graph


def get_resolver(root_dir: str) -> ImportResolver:
    """Get an import resolver for the given root directory."""
    return ImportResolver(root_dir)


# ============================================================================
# PageRank-based Repo Map
# ============================================================================

# Symbol reference patterns per language (function calls, variable references)
CALL_PATTERNS: dict[str, set[str]] = {
    "python": {"call", "attribute"},
    "javascript": {"call_expression", "member_expression"},
    "typescript": {"call_expression", "member_expression"},
    "tsx": {"call_expression", "member_expression"},
    "go": {"call_expression", "selector_expression"},
    "rust": {"call_expression", "field_expression", "macro_invocation"},
    "java": {"method_invocation", "field_access"},
    "c": {"call_expression", "field_expression"},
    "cpp": {"call_expression", "field_expression"},
}

# Definition patterns for extracting symbol names
DEFINITION_PATTERNS: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "variable_declarator"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "variable_declarator", "interface_declaration", "type_alias_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "variable_declarator", "interface_declaration", "type_alias_declaration"},
    "go": {"function_declaration", "method_declaration", "type_spec"},
    "rust": {"function_item", "struct_item", "impl_item", "trait_item", "enum_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
}

# Map extensions to tree-sitter language names
EXT_TO_TS_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}


class RepoMap:
    """Build a call graph of the repository and rank symbols using PageRank.

    This provides smarter dependency resolution by understanding which symbols
    are "hubs" (referenced by many files) vs leaf nodes.

    The graph is persisted to disk for fast reloading across sessions.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self._parsers: dict[str, any] = {}

        # Persistence setup
        self.storage_dir = Path.home() / ".ospack" / "repomap" / _get_repo_hash(str(self.root_dir))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self.storage_dir / "graph.pkl"
        self._mtime_path = self.storage_dir / "mtimes.pkl"

        # Graph structure: symbol -> set of symbols it references
        # Symbol format: "file_path:symbol_name"
        self._call_graph: dict[str, set[str]] = defaultdict(set)

        # Reverse graph: symbol -> set of symbols that reference it
        self._reverse_graph: dict[str, set[str]] = defaultdict(set)

        # Symbol -> file mapping
        self._symbol_to_file: dict[str, Path] = {}

        # File -> symbols defined in it
        self._file_symbols: dict[Path, set[str]] = defaultdict(set)

        # PageRank scores
        self._pagerank: dict[str, float] = {}

        # Track if graph has been built
        self._built = False

        # File modification times (for incremental updates)
        self._file_mtimes: dict[str, float] = {}

    def _get_parser(self, lang: str):
        """Get or create a parser for the given language."""
        if lang in self._parsers:
            return self._parsers[lang]

        try:
            import importlib
            from tree_sitter import Language, Parser

            # Language module mappings (same as chunker.py)
            module_map = {
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

            if lang not in module_map:
                return None

            module_name, func_name = module_map[lang]
            lang_module = importlib.import_module(module_name)
            lang_func = getattr(lang_module, func_name)
            language = Language(lang_func())
            parser = Parser(language)
            self._parsers[lang] = parser
            return parser
        except Exception as e:
            logger.debug("Failed to get parser for %s: %s", lang, e)
            return None

    def _extract_node_name(self, node: Node, lang: str) -> str | None:
        """Extract the name from a definition or call node."""
        # Try common field names
        for field in ("name", "function", "method", "field"):
            name_node = node.child_by_field_name(field)
            if name_node:
                if name_node.type == "identifier" or name_node.type == "type_identifier":
                    return name_node.text.decode("utf-8")
                # For call expressions, the function might be a member expression
                if name_node.type in ("member_expression", "attribute", "selector_expression"):
                    # Get the property/attribute name
                    prop = name_node.child_by_field_name("property") or name_node.child_by_field_name("attribute")
                    if prop:
                        return prop.text.decode("utf-8")

        # Fallback: look for identifier children
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")

        # For decorated definitions, recurse into the inner definition
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    return self._extract_node_name(child, lang)

        return None

    def _collect_definitions(self, node: Node, lang: str, file_path: Path, definitions: dict[str, Path]):
        """Recursively collect all definitions from an AST."""
        definition_types = DEFINITION_PATTERNS.get(lang, set())

        if node.type in definition_types:
            name = self._extract_node_name(node, lang)
            if name:
                symbol_key = f"{file_path}:{name}"
                definitions[symbol_key] = file_path
                self._symbol_to_file[symbol_key] = file_path
                self._file_symbols[file_path].add(symbol_key)

        for child in node.children:
            self._collect_definitions(child, lang, file_path, definitions)

    def _collect_references(self, node: Node, lang: str, file_path: Path,
                           current_scope: str | None, references: list[tuple[str, str]]):
        """Recursively collect all symbol references (calls, accesses)."""
        call_types = CALL_PATTERNS.get(lang, set())
        definition_types = DEFINITION_PATTERNS.get(lang, set())

        # Update scope when entering a definition
        new_scope = current_scope
        if node.type in definition_types:
            name = self._extract_node_name(node, lang)
            if name:
                new_scope = f"{file_path}:{name}"

        # Collect references
        if node.type in call_types:
            name = self._extract_node_name(node, lang)
            if name and current_scope:
                references.append((current_scope, name))

        # Also track simple identifier references in certain contexts
        if node.type == "identifier" and node.parent:
            parent_type = node.parent.type
            # Skip if it's a definition or import
            if parent_type not in definition_types and "import" not in parent_type.lower():
                name = node.text.decode("utf-8")
                if current_scope and len(name) > 2:  # Skip very short names
                    references.append((current_scope, name))

        for child in node.children:
            self._collect_references(child, lang, file_path, new_scope, references)

    def _resolve_reference(self, from_symbol: str, ref_name: str) -> str | None:
        """Try to resolve a reference name to a full symbol key."""
        from_file = self._symbol_to_file.get(from_symbol)
        if not from_file:
            return None

        # 1. Check same file first
        candidate = f"{from_file}:{ref_name}"
        if candidate in self._symbol_to_file:
            return candidate

        # 2. Check imported files (would need import tracking)
        # For now, do a global search for the symbol name
        for symbol_key in self._symbol_to_file:
            if symbol_key.endswith(f":{ref_name}"):
                return symbol_key

        return None

    def _save_graph(self) -> None:
        """Persist the call graph and PageRank scores to disk."""
        try:
            # Convert Path keys to strings for pickling
            data = {
                "call_graph": {str(k): list(v) for k, v in self._call_graph.items()},
                "reverse_graph": {str(k): list(v) for k, v in self._reverse_graph.items()},
                "symbol_to_file": {k: str(v) for k, v in self._symbol_to_file.items()},
                "file_symbols": {str(k): list(v) for k, v in self._file_symbols.items()},
                "pagerank": self._pagerank,
            }
            with open(self._graph_path, "wb") as f:
                pickle.dump(data, f)

            # Save file modification times
            with open(self._mtime_path, "wb") as f:
                pickle.dump(self._file_mtimes, f)

            logger.debug("Saved repo map to %s", self._graph_path)
        except Exception as e:
            logger.warning("Failed to save repo map: %s", e)

    def _load_graph(self) -> bool:
        """Load the call graph from disk if available.

        Returns:
            True if graph was loaded successfully, False otherwise.
        """
        if not self._graph_path.exists():
            return False

        try:
            with open(self._graph_path, "rb") as f:
                data = pickle.load(f)

            # Convert string keys back to proper types
            self._call_graph = defaultdict(set, {
                k: set(v) for k, v in data["call_graph"].items()
            })
            self._reverse_graph = defaultdict(set, {
                k: set(v) for k, v in data["reverse_graph"].items()
            })
            # Convert string values back to Path objects
            self._symbol_to_file = {k: Path(v) for k, v in data["symbol_to_file"].items()}
            self._file_symbols = defaultdict(set, {
                Path(k): set(v) for k, v in data["file_symbols"].items()
            })
            self._pagerank = data["pagerank"]

            # Load file modification times
            if self._mtime_path.exists():
                with open(self._mtime_path, "rb") as f:
                    self._file_mtimes = pickle.load(f)

            self._built = True
            logger.debug("Loaded repo map from disk (%d symbols)", len(self._symbol_to_file))
            return True
        except Exception as e:
            logger.warning("Failed to load repo map: %s", e)
            return False

    def _get_stale_files(self, files: list[Path]) -> tuple[list[Path], list[Path]]:
        """Check which files have changed since last build.

        Returns:
            Tuple of (files_to_rebuild, files_to_remove)
        """
        current_mtimes: dict[str, float] = {}
        for f in files:
            try:
                current_mtimes[str(f)] = f.stat().st_mtime
            except OSError:
                pass

        to_rebuild: list[Path] = []
        to_remove: list[Path] = []

        # Check for new or modified files
        for path_str, mtime in current_mtimes.items():
            if path_str not in self._file_mtimes or mtime > self._file_mtimes[path_str]:
                to_rebuild.append(Path(path_str))

        # Check for deleted files
        for path_str in self._file_mtimes:
            if path_str not in current_mtimes:
                to_remove.append(Path(path_str))

        # Update stored mtimes
        self._file_mtimes = current_mtimes

        return to_rebuild, to_remove

    def build(self, files: list[Path] | None = None, force: bool = False) -> None:
        """Build the call graph for the repository.

        Args:
            files: Optional list of files to process. If None, scans repo.
            force: If True, rebuild from scratch even if cache exists.
        """
        if files is None:
            # Scan for source files
            files = []
            for ext in EXT_TO_TS_LANG:
                files.extend(self.root_dir.rglob(f"*{ext}"))

        # Try to load from cache first (unless force rebuild)
        if not force and self._load_graph():
            stale, removed = self._get_stale_files(files)
            if not stale and not removed:
                logger.debug("Repo map is up to date (cached)")
                return
            logger.info("Repo map has %d stale files, rebuilding...", len(stale) + len(removed))
            # For now, do full rebuild on any changes
            # Future: incremental update
            self._call_graph = defaultdict(set)
            self._reverse_graph = defaultdict(set)
            self._symbol_to_file = {}
            self._file_symbols = defaultdict(set)
            self._pagerank = {}
            self._built = False

        logger.debug("Building repo map for %d files...", len(files))

        # Collect file modification times for cache invalidation
        for f in files:
            try:
                self._file_mtimes[str(f)] = f.stat().st_mtime
            except OSError:
                pass

        # Phase 1: Collect all definitions
        definitions: dict[str, Path] = {}
        for file_path in files:
            lang = EXT_TO_TS_LANG.get(file_path.suffix.lower())
            if not lang:
                continue

            parser = self._get_parser(lang)
            if not parser:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                tree = parser.parse(bytes(content, "utf-8"))
                self._collect_definitions(tree.root_node, lang, file_path, definitions)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", file_path, e)

        logger.debug("Found %d symbol definitions", len(definitions))

        # Phase 2: Collect references and build graph
        for file_path in files:
            lang = EXT_TO_TS_LANG.get(file_path.suffix.lower())
            if not lang:
                continue

            parser = self._get_parser(lang)
            if not parser:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                tree = parser.parse(bytes(content, "utf-8"))

                # Use file-level scope as default
                file_scope = f"{file_path}:__module__"
                self._symbol_to_file[file_scope] = file_path
                self._file_symbols[file_path].add(file_scope)

                references: list[tuple[str, str]] = []
                self._collect_references(tree.root_node, lang, file_path, file_scope, references)

                # Resolve references and build edges
                for from_symbol, ref_name in references:
                    to_symbol = self._resolve_reference(from_symbol, ref_name)
                    if to_symbol and to_symbol != from_symbol:
                        self._call_graph[from_symbol].add(to_symbol)
                        self._reverse_graph[to_symbol].add(from_symbol)
            except Exception as e:
                logger.debug("Failed to collect references from %s: %s", file_path, e)

        logger.debug("Built call graph with %d edges",
                    sum(len(v) for v in self._call_graph.values()))

        self._built = True

        # Save to disk for next session
        self._save_graph()

    def compute_pagerank(self, damping: float = 0.85, iterations: int = 30) -> dict[str, float]:
        """Compute PageRank scores for all symbols.

        Args:
            damping: Damping factor (probability of following a link)
            iterations: Number of iterations for convergence

        Returns:
            Dict mapping symbol keys to PageRank scores
        """
        if not self._built:
            self.build()

        all_symbols = set(self._symbol_to_file.keys())
        n = len(all_symbols)

        if n == 0:
            return {}

        # Initialize uniform scores
        scores = {s: 1.0 / n for s in all_symbols}

        for _ in range(iterations):
            new_scores = {}

            for symbol in all_symbols:
                # Base score from random jumps
                rank = (1 - damping) / n

                # Add contributions from incoming links
                for in_symbol in self._reverse_graph.get(symbol, set()):
                    out_degree = len(self._call_graph.get(in_symbol, set()))
                    if out_degree > 0:
                        rank += damping * scores[in_symbol] / out_degree

                new_scores[symbol] = rank

            scores = new_scores

        self._pagerank = scores

        # Save updated PageRank scores
        self._save_graph()

        return scores

    def get_ranked_dependencies(
        self,
        focus_file: Path,
        max_files: int = 10,
        boost_focus: float = 10.0,
    ) -> list[tuple[Path, float]]:
        """Get files ranked by importance relative to a focus file.

        Uses personalized PageRank - starts with high weight on the focus file
        and propagates importance through the call graph.

        Args:
            focus_file: The file to focus on
            max_files: Maximum number of files to return
            boost_focus: How much to boost the focus file's initial weight

        Returns:
            List of (file_path, score) tuples, sorted by relevance
        """
        if not self._built:
            self.build()

        if not self._pagerank:
            self.compute_pagerank()

        focus_file = focus_file.resolve()

        # Get symbols in the focus file
        focus_symbols = self._file_symbols.get(focus_file, set())

        if not focus_symbols:
            # Fall back to returning files by global PageRank
            file_scores: dict[Path, float] = defaultdict(float)
            for symbol, score in self._pagerank.items():
                file_path = self._symbol_to_file.get(symbol)
                if file_path:
                    file_scores[file_path] += score

            sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
            return [(f, s) for f, s in sorted_files if f != focus_file][:max_files]

        # Personalized PageRank: boost focus file symbols
        personalized_scores = dict(self._pagerank)
        for symbol in focus_symbols:
            personalized_scores[symbol] = personalized_scores.get(symbol, 0) * boost_focus

        # Propagate scores through call graph (one hop)
        propagated_scores: dict[str, float] = defaultdict(float)

        for symbol in focus_symbols:
            # Add score from symbols this file calls
            for called in self._call_graph.get(symbol, set()):
                propagated_scores[called] += personalized_scores.get(symbol, 0) * 0.5

            # Add score from symbols that call this file
            for caller in self._reverse_graph.get(symbol, set()):
                propagated_scores[caller] += personalized_scores.get(symbol, 0) * 0.3

        # Combine with base PageRank
        final_scores: dict[str, float] = {}
        for symbol in self._symbol_to_file:
            base = self._pagerank.get(symbol, 0)
            propagated = propagated_scores.get(symbol, 0)
            final_scores[symbol] = base + propagated

        # Aggregate by file
        file_scores: dict[Path, float] = defaultdict(float)
        for symbol, score in final_scores.items():
            file_path = self._symbol_to_file.get(symbol)
            if file_path and file_path != focus_file:
                file_scores[file_path] += score

        # Sort and return
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:max_files]

    def get_file_symbols(self, file_path: Path) -> list[str]:
        """Get all symbols defined in a file."""
        file_path = file_path.resolve()
        symbols = self._file_symbols.get(file_path, set())
        # Return just the symbol names, not full keys
        return [s.split(":")[-1] for s in symbols if not s.endswith(":__module__")]


# Global repo map cache
_repo_maps: dict[str, RepoMap] = {}


def get_repo_map(root_dir: str) -> RepoMap:
    """Get or create a RepoMap for the given directory."""
    root_dir = str(Path(root_dir).resolve())
    if root_dir not in _repo_maps:
        _repo_maps[root_dir] = RepoMap(root_dir)
    return _repo_maps[root_dir]
