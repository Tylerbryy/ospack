"""Import resolution for finding related files."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

# Import patterns by language
IMPORT_PATTERNS = {
    "python": [
        re.compile(r"^from\s+(\S+)\s+import", re.MULTILINE),
        re.compile(r"^import\s+(\S+)", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r'from\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', re.MULTILINE),
    ],
    "javascript": [
        re.compile(r'from\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', re.MULTILINE),
    ],
    "rust": [
        re.compile(r"^use\s+(\S+)", re.MULTILINE),
        re.compile(r"^mod\s+(\S+)", re.MULTILINE),
    ],
    "go": [
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r"import\s+\(\s*([^)]+)\)", re.MULTILINE | re.DOTALL),
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
}


def get_language(file_path: Path) -> str | None:
    """Get the language for a file based on extension."""
    return EXT_TO_LANG.get(file_path.suffix.lower())


class ImportResolver:
    """Resolve imports to actual files."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()

    def _extract_imports(self, file_path: Path, content: str) -> list[str]:
        """Extract import paths from file content."""
        language = get_language(file_path)
        if not language or language not in IMPORT_PATTERNS:
            return []

        imports = []
        for pattern in IMPORT_PATTERNS[language]:
            matches = pattern.findall(content)
            imports.extend(matches)

        return imports

    def _resolve_import(self, import_path: str, source_file: Path) -> Path | None:
        """Try to resolve an import path to a file."""
        # Skip external packages and stdlib
        if import_path.startswith(("@", "node_modules", "http", "__future__")):
            return None

        # Handle relative imports (Python-style: .module, ..module, etc.)
        if import_path.startswith("."):
            base_dir = source_file.parent

            # Count leading dots and strip them
            dots = 0
            for char in import_path:
                if char == ".":
                    dots += 1
                else:
                    break

            # Go up directories for each dot beyond the first
            for _ in range(dots - 1):
                base_dir = base_dir.parent

            # Get the module name (after the dots)
            module_name = import_path[dots:]

            # Convert dotted path to directory path (e.g., foo.bar -> foo/bar)
            if module_name:
                module_path = module_name.replace(".", "/")
                candidates = self._get_candidates(base_dir / module_path)
            else:
                # Just dots - refers to the package itself
                candidates = self._get_candidates(base_dir)
        else:
            # Absolute import - try from root, convert dots to path
            module_path = import_path.replace(".", "/")
            candidates = self._get_candidates(self.root_dir / module_path)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    def _get_candidates(self, base_path: Path) -> Iterator[Path]:
        """Generate candidate file paths for an import."""
        # Direct file
        yield base_path

        # With common extensions
        for ext in [".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go"]:
            yield base_path.with_suffix(ext)

        # Index files
        if base_path.is_dir() or not base_path.suffix:
            dir_path = base_path if base_path.is_dir() else base_path
            for index in [
                "index.ts",
                "index.tsx",
                "index.js",
                "index.jsx",
                "__init__.py",
                "mod.rs",
            ]:
                yield dir_path / index

    def resolve_imports(self, file_path: str | Path, content: str | None = None) -> list[Path]:
        """Resolve all imports in a file to actual file paths."""
        file_path = Path(file_path).resolve()

        if content is None:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return []

        import_paths = self._extract_imports(file_path, content)
        resolved = []

        for import_path in import_paths:
            resolved_path = self._resolve_import(import_path, file_path)
            if resolved_path and resolved_path not in resolved:
                resolved.append(resolved_path)

        return resolved

    def get_dependency_graph(
        self, entry_file: str | Path, max_depth: int = 3
    ) -> dict[Path, list[Path]]:
        """Build a dependency graph starting from an entry file."""
        entry_path = Path(entry_file).resolve()
        graph: dict[Path, list[Path]] = {}
        visited: set[Path] = set()

        def traverse(file_path: Path, depth: int):
            if depth > max_depth or file_path in visited:
                return
            if not file_path.exists():
                return

            visited.add(file_path)
            imports = self.resolve_imports(file_path)
            graph[file_path] = imports

            for imp in imports:
                traverse(imp, depth + 1)

        traverse(entry_path, 0)
        return graph


def get_resolver(root_dir: str) -> ImportResolver:
    """Get an import resolver for the given root directory."""
    return ImportResolver(root_dir)
