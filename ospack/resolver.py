"""Auto-discovering import resolver for code dependencies.

Features:
- Tree-sitter AST-based import extraction (more accurate than regex)
- Disk-based dependency graph caching for speed
- Auto-detection of tsconfig.json paths and Python src layouts
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from .indexer import get_repo_hash
from .log import get_logger

logger = get_logger(__name__)

# Import patterns by language (fallback for when tree-sitter fails)
IMPORT_PATTERNS = {
    "python": [
        re.compile(r"^\s*from\s+(\S+)\s+import", re.MULTILINE),
        re.compile(r"^\s*import\s+(\S+)", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r'from\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', re.MULTILINE),
        re.compile(r'import\s*\(\s*["\']([^"\']+)["\']\s*\)', re.MULTILINE),
        # Re-exports
        re.compile(r'export\s+\*\s+from\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'export\s+\{[^}]*\}\s+from\s+["\']([^"\']+)["\']', re.MULTILINE),
    ],
    "javascript": [
        re.compile(r'from\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', re.MULTILINE),
        re.compile(r'export\s+\*\s+from\s+["\']([^"\']+)["\']', re.MULTILINE),
    ],
    "rust": [
        re.compile(r"^\s*use\s+(\S+)", re.MULTILINE),
        re.compile(r"^\s*mod\s+(\S+)", re.MULTILINE),
    ],
    "go": [
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
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

# Tree-sitter language name mapping
EXT_TO_TS_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
}

# Tree-sitter node types for imports/exports
TS_IMPORT_NODES = {
    "import_statement",      # Python, JS/TS
    "import_from_statement", # Python
    "import_declaration",    # Go
    "export_statement",      # JS/TS
    "use_declaration",       # Rust
    "mod_item",              # Rust
}

# Scoped extensions for candidate generation
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
    """Resolve imports with auto-discovery and caching."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()

        # Auto-discovered config
        self._ts_paths: dict[str, list[Path]] = {}
        self._python_roots: list[Path] = [self.root_dir]

        # Disk-based cache for dependency graph
        self._cache_dir = Path.home() / ".ospack" / "cache" / get_repo_hash(str(self.root_dir))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._deps_cache_path = self._cache_dir / "deps.json"
        self._deps_cache: dict[str, dict] = {}
        self._load_cache()

        # Tree-sitter parser (lazy loaded)
        self._chunker = None

        # AUTO-DISCOVERY
        self._scan_project_config()

    def _load_cache(self):
        """Load cached dependency graph from disk."""
        if self._deps_cache_path.exists():
            try:
                with open(self._deps_cache_path, "r") as f:
                    data = json.load(f)
                # Validate mtimes - only keep fresh entries
                for file_str, entry in data.items():
                    path = Path(file_str)
                    try:
                        if path.exists() and abs(path.stat().st_mtime - entry["mtime"]) < 0.01:
                            self._deps_cache[file_str] = entry
                    except OSError:
                        pass
                logger.debug("Loaded %d cached dependency entries", len(self._deps_cache))
            except Exception as e:
                logger.debug("Could not load deps cache: %s", e)
                self._deps_cache = {}

    def _save_cache(self):
        """Persist dependency cache to disk."""
        try:
            with open(self._deps_cache_path, "w") as f:
                json.dump(self._deps_cache, f)
        except Exception as e:
            logger.debug("Could not save deps cache: %s", e)

    def _get_chunker(self):
        """Lazy-load the chunker for tree-sitter parsing."""
        if self._chunker is None:
            from .chunker import get_chunker
            self._chunker = get_chunker()
        return self._chunker

    def _scan_project_config(self):
        """Auto-detect TSConfig, JSConfig, and Python layouts."""
        # TypeScript / JavaScript Paths
        for config_name in ["tsconfig.json", "jsconfig.json"]:
            config_path = self.root_dir / config_name
            if config_path.exists():
                try:
                    content = config_path.read_text(encoding="utf-8", errors="ignore")
                    content = re.sub(r"//.*", "", content, flags=re.MULTILINE)
                    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

                    data = json.loads(content)
                    compiler_opts = data.get("compilerOptions", {})
                    base_url = compiler_opts.get("baseUrl", ".")
                    paths = compiler_opts.get("paths", {})

                    base_dir = (self.root_dir / base_url).resolve()

                    for alias, targets in paths.items():
                        clean_alias = alias.replace("*", "")
                        clean_targets = []
                        for t in targets:
                            t_clean = t.replace("*", "")
                            target_path = (base_dir / t_clean).resolve()
                            clean_targets.append(target_path)
                        self._ts_paths[clean_alias] = clean_targets

                    if self._ts_paths:
                        logger.debug("Loaded TS paths from %s: %s", config_name, list(self._ts_paths.keys()))
                except Exception as e:
                    logger.debug("Could not parse %s: %s", config_name, e)

        # Python Src Layout
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

    def _extract_imports_ast(self, file_path: Path, content: str) -> list[str]:
        """Extract imports using tree-sitter AST (more accurate than regex)."""
        ts_lang = EXT_TO_TS_LANG.get(file_path.suffix.lower())
        if not ts_lang:
            return []

        try:
            chunker = self._get_chunker()
            parser = chunker._parsers.get(ts_lang)
            if not parser:
                return []

            tree = parser.parse(bytes(content, "utf-8"))
            imports = []

            def extract_string_literals(node) -> list[str]:
                """Extract string literal values from a node."""
                strings = []
                if node.type == "string":
                    text = node.text.decode("utf-8").strip("'\"")
                    if text:
                        strings.append(text)
                for child in node.children:
                    strings.extend(extract_string_literals(child))
                return strings

            def visit(node):
                if node.type in TS_IMPORT_NODES:
                    # Extract string literals (the import paths)
                    imports.extend(extract_string_literals(node))

                    # Python: also extract dotted names from import statements
                    if node.type in ("import_statement", "import_from_statement"):
                        for child in node.children:
                            if child.type in ("dotted_name", "relative_import"):
                                name = child.text.decode("utf-8")
                                if name and not name.startswith("."):
                                    imports.append(name)
                            elif child.type == "aliased_import":
                                for gc in child.children:
                                    if gc.type == "dotted_name":
                                        imports.append(gc.text.decode("utf-8"))

                for child in node.children:
                    visit(child)

            visit(tree.root_node)
            return imports

        except Exception as e:
            logger.debug("AST parse failed for %s: %s", file_path, e)
            return []

    def _extract_imports_regex(self, file_path: Path, content: str) -> list[str]:
        """Extract imports using regex (fallback)."""
        language = get_language(file_path)
        if not language or language not in IMPORT_PATTERNS:
            return []

        imports = []
        for pattern in IMPORT_PATTERNS[language]:
            matches = pattern.findall(content)

            if language == "go" and "(" in pattern.pattern:
                for block in matches:
                    block_imports = re.findall(r'["\']([^"\']+)["\']', block)
                    imports.extend(block_imports)
            else:
                imports.extend(matches)

        return imports

    def _extract_imports(self, file_path: Path, content: str) -> list[str]:
        """Extract import paths from file content (AST-first, regex fallback)."""
        # Try AST-based extraction first
        imports = self._extract_imports_ast(file_path, content)

        # Fallback to regex if AST produced nothing
        if not imports:
            imports = self._extract_imports_regex(file_path, content)

        return imports

    def _get_candidates(self, base_path: Path, language: str | None) -> Iterator[Path]:
        """Generate candidate file paths using language-specific rules."""
        yield base_path

        valid_exts = LANG_EXTENSIONS.get(language, [])
        for ext in valid_exts:
            yield base_path.with_suffix(ext)

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

        # Filter obvious external packages
        if import_path.startswith(("node_modules", "http")):
            return None
        if language == "python" and import_path in (
            "sys", "os", "typing", "__future__", "datetime", "json", "re",
            "pathlib", "collections", "functools", "itertools", "dataclasses",
            "abc", "enum", "logging", "io", "copy", "math", "random", "time",
            "threading", "multiprocessing", "subprocess", "shutil", "tempfile",
            "hashlib", "base64", "uuid", "contextlib", "warnings", "traceback",
        ):
            return None

        candidates: list[Path] = []

        # Handle TS/JS Aliases
        if language in ("typescript", "javascript"):
            alias_matches = self._resolve_alias(import_path)
            for m in alias_matches:
                candidates.extend(self._get_candidates(m, language))

        # Handle Relative Imports
        if import_path.startswith("."):
            base_dir = source_file.parent

            if language == "python":
                dots = 0
                for char in import_path:
                    if char == ".":
                        dots += 1
                    else:
                        break

                for _ in range(dots - 1):
                    base_dir = base_dir.parent

                module_name = import_path[dots:]
                target = base_dir / module_name.replace(".", "/")
                candidates.extend(self._get_candidates(target, language))
            else:
                target = base_dir / import_path
                candidates.extend(self._get_candidates(target, language))

        # Handle Absolute / Package Imports
        elif not candidates:
            if language == "python":
                clean_path = import_path.replace(".", "/")
                for root in self._python_roots:
                    target = root / clean_path
                    candidates.extend(self._get_candidates(target, language))

            elif language in ("typescript", "javascript"):
                candidates.extend(self._get_candidates(self.root_dir / import_path, language))

                if import_path.startswith("@"):
                    alias_path = import_path[1:].lstrip("/")
                    candidates.extend(self._get_candidates(
                        self.root_dir / "src" / alias_path, language
                    ))
                    candidates.extend(self._get_candidates(
                        self.root_dir / alias_path, language
                    ))
            else:
                target = self.root_dir / import_path
                candidates.extend(self._get_candidates(target, language))

        # Check existence
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
            except OSError:
                pass

        return None

    def resolve_imports(self, file_path: str | Path) -> tuple[Path, ...]:
        """Resolve all imports in a file to actual file paths.

        Uses disk cache for speed on subsequent calls.
        """
        file_path = Path(file_path).resolve()
        file_str = str(file_path)

        # Check cache
        if file_str in self._deps_cache:
            cached = self._deps_cache[file_str]
            try:
                if file_path.exists() and abs(file_path.stat().st_mtime - cached["mtime"]) < 0.01:
                    return tuple(Path(p) for p in cached["deps"])
            except OSError:
                pass

        # Resolve fresh
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ()

        raw_imports = self._extract_imports(file_path, content)
        resolved = []
        seen = set()

        for imp in raw_imports:
            imp = imp.strip().strip("';\"")

            res = self._resolve_import(imp, file_path)
            if res and res not in seen and res != file_path:
                resolved.append(res)
                seen.add(res)

        result = tuple(resolved)

        # Update cache
        try:
            self._deps_cache[file_str] = {
                "mtime": file_path.stat().st_mtime,
                "deps": [str(p) for p in result],
            }
            self._save_cache()
        except OSError:
            pass

        return result

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

            imports = list(self.resolve_imports(current_path))
            graph[current_path] = imports

            for imp in imports:
                traverse(imp, depth + 1)

        traverse(entry_path, 0)
        return graph

    def clear_cache(self):
        """Clear the dependency cache."""
        self._deps_cache = {}
        if self._deps_cache_path.exists():
            self._deps_cache_path.unlink()


# Singleton cache
_resolvers: dict[str, ImportResolver] = {}


def get_resolver(root_dir: str) -> ImportResolver:
    """Get or create an import resolver for the given root directory."""
    root_dir = str(Path(root_dir).resolve())
    if root_dir not in _resolvers:
        _resolvers[root_dir] = ImportResolver(root_dir)
    return _resolvers[root_dir]
