"""Repository structure mapper - generates compressed structural overview."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .log import get_logger

if TYPE_CHECKING:
    from tree_sitter import Node

logger = get_logger(__name__)

# Directories to skip
SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", ".venv", "venv", "vendor", "__pycache__",
    "dist", "build", "target", ".next", ".turbo",
    ".idea", ".vscode", "coverage",
}

# Extensions to process for signatures
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".cs",
}

# Map extensions to tree-sitter language names
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}

LANG_TO_MODULE = {
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

# Definition types to extract
DEFINITION_TYPES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "lexical_declaration"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration", "lexical_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "trait_item", "enum_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "c": {"function_definition", "struct_specifier", "declaration"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
}


@dataclass
class Signature:
    """A single signature with its nesting depth."""
    text: str
    depth: int = 0


@dataclass
class FileSignatures:
    """Signatures extracted from a file."""
    path: Path
    signatures: list[Signature] = field(default_factory=list)
    error: str | None = None


class SignatureExtractor:
    """Extract signatures from source files using tree-sitter."""

    def __init__(self):
        self._parsers: dict[str, any] = {}

    def _get_parser(self, lang: str):
        """Get or create a parser for the given language."""
        if lang in self._parsers:
            return self._parsers[lang]

        try:
            import importlib
            from tree_sitter import Language, Parser

            if lang not in LANG_TO_MODULE:
                return None

            module_name, func_name = LANG_TO_MODULE[lang]
            lang_module = importlib.import_module(module_name)
            lang_func = getattr(lang_module, func_name)
            language = Language(lang_func())
            parser = Parser(language)
            self._parsers[lang] = parser
            return parser
        except Exception as e:
            logger.debug("Failed to get parser for %s: %s", lang, e)
            return None

    def _get_node_name(self, node: Node) -> str | None:
        """Extract the name from a definition node."""
        for field_name in ("name", "declarator"):
            name_node = node.child_by_field_name(field_name)
            if name_node:
                while name_node.type in ("function_declarator", "pointer_declarator"):
                    inner = name_node.child_by_field_name("declarator")
                    if inner:
                        name_node = inner
                    else:
                        break

                if name_node.type in ("identifier", "type_identifier"):
                    return name_node.text.decode("utf-8") if name_node.text else None
                elif name_node.text:
                    return name_node.text.decode("utf-8")

        # Handle decorated_definition
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    return self._get_node_name(child)

        # Fallback
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8") if child.text else None

        return None

    def _extract_signature(self, node: Node, content: str, lang: str) -> str | None:
        """Extract signature line from a definition node."""
        node_text = content[node.start_byte:node.end_byte]
        lines = node_text.split("\n")

        if not lines:
            return None

        # For decorated definitions, skip to the actual def/class line
        if node.type == "decorated_definition":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("def ", "class ", "async def ")):
                    sig = stripped
                    # Handle multi-line params in decorated functions
                    if "(" in sig and ")" not in sig:
                        idx = lines.index(line)
                        for cont_line in lines[idx + 1 : idx + 10]:
                            sig += " " + cont_line.strip()
                            if ")" in cont_line:
                                break
                    sig = re.sub(r"\s+", " ", sig)  # Collapse whitespace
                    if not sig.endswith(":"):
                        sig = sig + ":"
                    return sig
            return None

        # Get first line
        sig = lines[0].strip()

        # Handle multi-line signatures (unclosed parentheses)
        if "(" in sig and ")" not in sig:
            for line in lines[1:10]:
                sig += " " + line.strip()
                if ")" in line:
                    break

        # Collapse multiple whitespace into single space
        sig = re.sub(r"\s+", " ", sig)

        # Clean up
        if lang == "python":
            if not sig.endswith(":"):
                sig = sig + ":"
            return sig
        else:
            # C-like languages
            if "{" in sig:
                sig = sig.split("{")[0].rstrip()
            return sig

    def _is_class_like(self, node_type: str) -> bool:
        """Check if node type is a class/struct that should increase depth for children."""
        return node_type in {
            "class_definition",  # Python
            "class_declaration",  # JS/TS/Java
            "class_specifier",  # C++
            "struct_item",  # Rust
            "impl_item",  # Rust
            "trait_item",  # Rust
            "interface_declaration",  # TS/Java
        }

    def _collect_signatures(
        self,
        node: Node,
        content: str,
        lang: str,
        signatures: list[Signature],
        depth: int = 0,
        seen_sigs: set | None = None,
    ):
        """Recursively collect signatures from AST with depth tracking."""
        if seen_sigs is None:
            seen_sigs = set()

        definition_types = DEFINITION_TYPES.get(lang, set())

        if node.type in definition_types:
            sig_text = self._extract_signature(node, content, lang)
            if sig_text and sig_text not in seen_sigs:
                seen_sigs.add(sig_text)
                signatures.append(Signature(text=sig_text, depth=depth))

            # For decorated definitions, don't recurse into the inner def/class
            # as we've already extracted it
            if node.type == "decorated_definition":
                return

            # Increase depth for children if this is a class-like container
            next_depth = depth + 1 if self._is_class_like(node.type) else depth

            # Recurse for nested definitions (e.g., methods in classes)
            for child in node.children:
                self._collect_signatures(child, content, lang, signatures, next_depth, seen_sigs)
        else:
            for child in node.children:
                self._collect_signatures(child, content, lang, signatures, depth, seen_sigs)

    def extract(self, file_path: Path) -> FileSignatures:
        """Extract signatures from a file."""
        result = FileSignatures(path=file_path)

        lang = EXT_TO_LANG.get(file_path.suffix.lower())
        if not lang:
            return result

        parser = self._get_parser(lang)
        if not parser:
            return result

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = parser.parse(bytes(content, "utf-8"))
            self._collect_signatures(tree.root_node, content, lang, result.signatures)
        except Exception as e:
            result.error = str(e)
            logger.debug("Failed to extract signatures from %s: %s", file_path, e)

        return result


def generate_repo_map(
    root: Path,
    format: str = "tree",
    include_signatures: bool = True,
    max_sigs: int | None = None,
) -> str:
    """Generate a structural map of the repository.

    Args:
        root: Repository root directory
        format: Output format - "tree" or "flat"
        include_signatures: Whether to include function/class signatures
        max_sigs: Maximum signatures per file (None = no limit)

    Returns:
        Formatted string representation of the repo structure
    """
    root = root.resolve()
    extractor = SignatureExtractor() if include_signatures else None

    lines: list[str] = []

    def process_dir(dir_path: Path, prefix: str = "", is_last: bool = True):
        """Recursively process a directory."""
        # Get directory name
        rel_path = dir_path.relative_to(root) if dir_path != root else Path(".")
        dir_name = dir_path.name if dir_path != root else str(root.name)

        if format == "tree":
            connector = "└── " if is_last else "├── "
            if dir_path == root:
                lines.append(f"{dir_name}/")
            else:
                lines.append(f"{prefix}{connector}{dir_name}/")

        # Get children, filtering out hidden and excluded
        try:
            children = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        # Filter children
        filtered = []
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_dir() and child.name in SKIP_DIRS:
                continue
            filtered.append(child)

        # Process children
        for i, child in enumerate(filtered):
            is_last_child = i == len(filtered) - 1

            if format == "tree":
                if dir_path == root:
                    child_prefix = ""
                else:
                    child_prefix = prefix + ("    " if is_last else "│   ")
            else:
                child_prefix = ""

            if child.is_dir():
                process_dir(child, child_prefix, is_last_child)
            else:
                process_file(child, child_prefix, is_last_child)

    def process_file(file_path: Path, prefix: str, is_last: bool):
        """Process a single file."""
        rel_path = file_path.relative_to(root)

        if format == "tree":
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{file_path.name}")
        else:
            lines.append(f"{rel_path}")

        # Extract signatures if requested and file is a code file
        if include_signatures and file_path.suffix.lower() in CODE_EXTENSIONS:
            result = extractor.extract(file_path)
            if result.signatures:
                # Base prefix for signatures
                if format == "tree":
                    base_sig_prefix = prefix + ("    " if is_last else "│   ")
                else:
                    base_sig_prefix = ""

                # Apply max_sigs limit
                sigs_to_show = result.signatures
                truncated_count = 0
                if max_sigs is not None and len(result.signatures) > max_sigs:
                    sigs_to_show = result.signatures[:max_sigs]
                    truncated_count = len(result.signatures) - max_sigs

                for sig in sigs_to_show:
                    # Calculate indentation based on depth (cap at 2 levels)
                    indent_level = min(sig.depth, 2)
                    indent_str = "    " * indent_level

                    if format == "tree":
                        sig_prefix = base_sig_prefix + "  " + indent_str
                    else:
                        sig_prefix = "  " + indent_str

                    # Truncate long signatures
                    sig_text = sig.text
                    if len(sig_text) > 80:
                        sig_text = sig_text[:77] + "..."

                    lines.append(f"{sig_prefix}{sig_text}")

                # Show truncation notice
                if truncated_count > 0:
                    if format == "tree":
                        trunc_prefix = base_sig_prefix + "  "
                    else:
                        trunc_prefix = "  "
                    lines.append(f"{trunc_prefix}... ({truncated_count} more)")

    process_dir(root)

    return "\n".join(lines)
