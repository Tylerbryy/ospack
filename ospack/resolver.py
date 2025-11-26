"""Auto-discovering import resolver for code dependencies."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from .log import get_logger

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
