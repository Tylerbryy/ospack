"""Tree-sitter based code chunking using py-tree-sitter.

Uses individual tree-sitter language packages for each language.
No fallbacks - tree-sitter is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tree_sitter import Language, Parser, Query

from .log import get_logger

logger = get_logger(__name__)

# Lazy-loaded language modules
_LANGUAGE_MODULES: dict[str, Any] = {}


def _get_language(lang: str) -> Language | None:
    """Get a tree-sitter Language object."""
    if lang in _LANGUAGE_MODULES:
        return _LANGUAGE_MODULES[lang]

    try:
        if lang == "python":
            import tree_sitter_python as ts_lang
        elif lang == "typescript":
            import tree_sitter_typescript as ts_lang
            # TypeScript module has .language_typescript()
            _LANGUAGE_MODULES[lang] = Language(ts_lang.language_typescript())
            return _LANGUAGE_MODULES[lang]
        elif lang == "tsx":
            import tree_sitter_typescript as ts_lang
            _LANGUAGE_MODULES[lang] = Language(ts_lang.language_tsx())
            return _LANGUAGE_MODULES[lang]
        elif lang == "javascript":
            import tree_sitter_javascript as ts_lang
        elif lang == "go":
            import tree_sitter_go as ts_lang
        elif lang == "rust":
            import tree_sitter_rust as ts_lang
        elif lang == "java":
            import tree_sitter_java as ts_lang
        elif lang == "c":
            import tree_sitter_c as ts_lang
        elif lang == "cpp":
            import tree_sitter_cpp as ts_lang
        elif lang == "c_sharp":
            import tree_sitter_c_sharp as ts_lang
        elif lang == "ruby":
            import tree_sitter_ruby as ts_lang
        elif lang == "json":
            import tree_sitter_json as ts_lang
        elif lang == "html":
            import tree_sitter_html as ts_lang
        elif lang == "css":
            import tree_sitter_css as ts_lang
        elif lang == "bash":
            import tree_sitter_bash as ts_lang
        elif lang == "yaml":
            import tree_sitter_yaml as ts_lang
        elif lang == "toml":
            import tree_sitter_toml as ts_lang
        elif lang == "markdown":
            import tree_sitter_markdown as ts_lang
        else:
            _LANGUAGE_MODULES[lang] = None
            return None

        _LANGUAGE_MODULES[lang] = Language(ts_lang.language())
        return _LANGUAGE_MODULES[lang]

    except ImportError:
        logger.debug("Language package for %s not installed", lang)
        _LANGUAGE_MODULES[lang] = None
        return None


# Map file extensions to language names
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".json": "json",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".markdown": "markdown",
}

# Node types that represent semantic code units
DEFINITION_TYPES = frozenset([
    # Functions
    "function_declaration",
    "function_definition",
    "method_definition",
    "method_declaration",
    "arrow_function",
    "func_literal",
    "function_item",
    "async_function_definition",
    # Classes
    "class_declaration",
    "class_definition",
    "impl_item",
    "struct_item",
    "trait_item",
    # Other
    "interface_declaration",
    "type_alias_declaration",
    "enum_item",
    "module",
    "mod_item",
    "decorated_definition",
])

# Top-level value definitions (const foo = () => ...)
VALUE_DEF_TYPES = frozenset([
    "lexical_declaration",
    "variable_declaration",
])


@dataclass
class Chunk:
    """A semantic chunk of code."""
    content: str
    start_line: int
    end_line: int
    type: Literal["function", "class", "block", "other"]
    context: list[str] = field(default_factory=list)
    chunk_index: int = -1
    is_anchor: bool = False

    @property
    def name(self) -> str | None:
        """Extract name from context if available."""
        for ctx in reversed(self.context):
            if ctx.startswith(("Function:", "Class:", "Method:", "Symbol:")):
                return ctx.split(":", 1)[1].strip()
        return None

    @property
    def node_type(self) -> str:
        """Return type as string for compatibility."""
        return self.type


class TreeSitterChunker:
    """Tree-sitter based chunker using py-tree-sitter."""

    # Tuned for speed: Fill context window (512 tokens) more aggressively
    MAX_CHUNK_LINES = 75
    MAX_CHUNK_CHARS = 2000
    OVERLAP_LINES = 10
    OVERLAP_CHARS = 200  # For low-newline regions (JSON, lockfiles)

    def __init__(self):
        self._parsers: dict[str, Parser] = {}

    def _get_parser(self, lang: str) -> Parser | None:
        """Get or create a parser for the given language."""
        if lang in self._parsers:
            return self._parsers[lang]

        language = _get_language(lang)
        if not language:
            return None

        parser = Parser(language)
        self._parsers[lang] = parser
        return parser

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        """Chunk a file into semantic units."""
        ext = Path(file_path).suffix.lower()
        lang = EXTENSION_TO_LANGUAGE.get(ext)

        if not lang:
            # Unknown extension - create single file chunk
            return [self._create_file_chunk(file_path, content)]

        parser = self._get_parser(lang)
        if not parser:
            # Language not installed - create single file chunk
            return [self._create_file_chunk(file_path, content)]

        try:
            tree = parser.parse(content.encode("utf-8"))
        except Exception as e:
            logger.warning("Parse failed for %s: %s", file_path, e)
            return [self._create_file_chunk(file_path, content)]

        raw_chunks = self._extract_chunks(file_path, content, tree)

        if not raw_chunks:
            return [self._create_file_chunk(file_path, content)]

        # Split any oversized chunks
        result: list[Chunk] = []
        for chunk in raw_chunks:
            result.extend(self._split_if_too_big(chunk))

        return result

    def _create_file_chunk(self, file_path: str, content: str) -> Chunk:
        """Create a chunk for the entire file."""
        lines = content.split("\n")
        return Chunk(
            content=content,
            start_line=0,
            end_line=len(lines),
            type="block",
            context=[f"File: {file_path}"],
        )

    def _extract_chunks(
        self, file_path: str, content: str, tree: Any
    ) -> list[Chunk]:
        """Extract semantic chunks from parsed tree."""
        root = tree.root_node
        file_context = f"File: {file_path}"
        chunks: list[Chunk] = []
        block_chunks: list[Chunk] = []
        cursor_index = 0
        cursor_row = 0
        saw_definition = False

        def classify(node: Any) -> Literal["function", "class", "other"]:
            t = node.type
            if "class" in t or t in ("impl_item", "struct_item", "trait_item"):
                return "class"
            if t in DEFINITION_TYPES or "function" in t or "method" in t:
                return "function"
            return "other"

        def unwrap_export(node: Any) -> Any:
            if node.type == "export_statement" and node.named_children:
                return node.named_children[0]
            return node

        def is_top_level_value_def(node: Any) -> bool:
            if node.type not in VALUE_DEF_TYPES:
                return False
            parent_type = node.parent.type if node.parent else ""
            allowed = {"program", "module", "source_file", "class_body"}
            if parent_type and parent_type not in allowed:
                return False
            text = node.text.decode("utf-8") if node.text else ""
            return "=>" in text or "function " in text or "class " in text

        def get_node_name(node: Any) -> str | None:
            # Try field-based access
            for field_name in ("name", "property", "identifier"):
                try:
                    child = node.child_by_field_name(field_name)
                    if child and child.text:
                        return child.text.decode("utf-8")
                except Exception:
                    pass

            # Try finding identifier child
            id_types = {"identifier", "property_identifier", "type_identifier", "field_identifier"}
            for child in node.named_children or []:
                if child.type in id_types and child.text:
                    return child.text.decode("utf-8")

            # Try variable_declarator pattern
            for child in node.named_children or []:
                if child.type == "variable_declarator":
                    for subchild in child.named_children or []:
                        if subchild.type in id_types and subchild.text:
                            return subchild.text.decode("utf-8")

            # Regex fallback
            text = node.text.decode("utf-8") if node.text else ""
            match = re.search(r"(?:class|function|def|async def)\s+([A-Za-z0-9_$]+)", text)
            if match:
                return match.group(1)
            match = re.search(r"(?:const|let|var)\s+([A-Za-z0-9_$]+)", text)
            if match:
                return match.group(1)
            return None

        def label_for_node(node: Any) -> str | None:
            name = get_node_name(node)
            t = node.type
            if "class" in t:
                return f"Class: {name or '<anonymous class>'}"
            if "method" in t:
                return f"Method: {name or '<anonymous method>'}"
            if "function" in t or t in DEFINITION_TYPES:
                return f"Function: {name or '<anonymous function>'}"
            if is_top_level_value_def(node):
                return f"Function: {name or '<anonymous function>'}"
            return f"Symbol: {name}" if name else None

        def add_chunk(node: Any, context: list[str]) -> None:
            text = node.text.decode("utf-8") if node.text else ""
            chunks.append(Chunk(
                content=text,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                type=classify(node),
                context=context.copy(),
            ))

        def visit(node: Any, stack: list[str]) -> None:
            nonlocal saw_definition
            effective = unwrap_export(node)
            is_definition = (
                effective.type in DEFINITION_TYPES or
                is_top_level_value_def(effective)
            )
            next_stack = stack

            if is_definition:
                saw_definition = True
                label = label_for_node(effective)
                context = [*stack, label] if label else stack.copy()
                add_chunk(effective, context)
                next_stack = context

            for child in effective.named_children or []:
                visit(child, next_stack)

        # Visit all top-level children
        for child in root.named_children or []:
            visit(child, [file_context])

            effective = unwrap_export(child)
            is_definition = (
                effective.type in DEFINITION_TYPES or
                is_top_level_value_def(effective)
            )
            if not is_definition:
                continue

            # Capture gaps between definitions
            if child.start_byte > cursor_index:
                gap_text = content[cursor_index:child.start_byte]
                if gap_text.strip():
                    block_chunks.append(Chunk(
                        content=gap_text,
                        start_line=cursor_row,
                        end_line=child.start_point[0],
                        type="block",
                        context=[file_context],
                    ))

            cursor_index = child.end_byte
            cursor_row = child.end_point[0]

        # Capture trailing content
        if cursor_index < len(content):
            tail_text = content[cursor_index:]
            if tail_text.strip():
                block_chunks.append(Chunk(
                    content=tail_text,
                    start_line=cursor_row,
                    end_line=root.end_point[0],
                    type="block",
                    context=[file_context],
                ))

        if not saw_definition:
            return []

        # Combine and sort by position
        combined = [*block_chunks, *chunks]
        combined.sort(key=lambda c: (c.start_line, c.end_line))
        return combined

    def _split_if_too_big(self, chunk: Chunk) -> list[Chunk]:
        """Split oversized chunks with overlap."""
        char_count = len(chunk.content)
        lines = chunk.content.split("\n")
        line_count = len(lines)

        if line_count <= self.MAX_CHUNK_LINES and char_count <= self.MAX_CHUNK_CHARS:
            return [chunk]

        # If huge but low-newline, split by chars
        if char_count > self.MAX_CHUNK_CHARS and line_count <= self.MAX_CHUNK_LINES:
            return self._split_by_chars(chunk)

        # Line-based sliding window split
        sub_chunks: list[Chunk] = []
        stride = max(1, self.MAX_CHUNK_LINES - self.OVERLAP_LINES)
        header = self._extract_header_line(chunk.content)

        for i in range(0, line_count, stride):
            end = min(i + self.MAX_CHUNK_LINES, line_count)
            sub_lines = lines[i:end]
            if len(sub_lines) < 3 and i > 0:
                continue

            sub_content = "\n".join(sub_lines)
            if header and i > 0 and chunk.type != "block":
                sub_content = f"{header}\n{sub_content}"

            sub_chunks.append(Chunk(
                content=sub_content,
                start_line=chunk.start_line + i,
                end_line=chunk.start_line + end,
                type=chunk.type,
                context=chunk.context.copy(),
            ))

        # Safety: char split any leftover giant subchunks
        result: list[Chunk] = []
        for sc in sub_chunks:
            if len(sc.content) > self.MAX_CHUNK_CHARS:
                result.extend(self._split_by_chars(sc))
            else:
                result.append(sc)
        return result

    def _split_by_chars(self, chunk: Chunk) -> list[Chunk]:
        """Split chunk by character count with overlap."""
        result: list[Chunk] = []
        stride = max(1, self.MAX_CHUNK_CHARS - self.OVERLAP_CHARS)

        for i in range(0, len(chunk.content), stride):
            end = min(i + self.MAX_CHUNK_CHARS, len(chunk.content))
            sub = chunk.content[i:end]
            if not sub.strip():
                continue

            prefix_lines = chunk.content[:i].count("\n")
            sub_line_count = sub.count("\n") + 1

            result.append(Chunk(
                content=sub,
                start_line=chunk.start_line + prefix_lines,
                end_line=chunk.start_line + prefix_lines + sub_line_count,
                type=chunk.type,
                context=chunk.context.copy(),
            ))

        return result

    def _extract_header_line(self, text: str) -> str | None:
        """Extract first non-blank line as header."""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped:
                return stripped
        return None


# --- Helper Functions ---

def extract_top_comments(lines: list[str]) -> list[str]:
    """Extract leading comments from file."""
    comments: list[str] = []
    in_block = False

    for line in lines:
        trimmed = line.strip()
        if in_block:
            comments.append(line)
            if "*/" in trimmed:
                in_block = False
            continue
        if trimmed == "":
            comments.append(line)
            continue
        if trimmed.startswith(("//", "#!", "# ", "#-", '"""', "'''")):
            comments.append(line)
            continue
        if trimmed.startswith("/*"):
            comments.append(line)
            if "*/" not in trimmed:
                in_block = True
            continue
        break

    # Remove trailing blank lines
    while comments and not comments[-1].strip():
        comments.pop()

    return comments


def extract_imports(lines: list[str], limit: int = 200) -> list[str]:
    """Extract import/require statements."""
    modules: list[str] = []

    for raw in lines[:limit]:
        trimmed = raw.strip()
        if not trimmed:
            continue

        # Python imports
        if trimmed.startswith("from "):
            match = re.match(r'from\s+([^\s]+)\s+import', trimmed)
            if match:
                modules.append(match.group(1))
            continue

        if trimmed.startswith("import "):
            # ES6 import
            from_match = re.search(r'from\s+["\']([^"\']+)["\']', trimmed)
            side_effect = re.match(r'^import\s+["\']([^"\']+)["\']', trimmed)
            named = re.match(r'import\s+(?:\* as\s+)?([A-Za-z0-9_$]+)', trimmed)

            if from_match:
                modules.append(from_match.group(1))
            elif side_effect:
                modules.append(side_effect.group(1))
            elif named:
                # Python style import
                match = re.match(r'import\s+([^\s,]+)', trimmed)
                if match:
                    modules.append(match.group(1).split(".")[0])
            continue

        # CommonJS require
        require_match = re.search(r'require\(\s*["\']([^"\']+)["\']\s*\)', trimmed)
        if require_match:
            modules.append(require_match.group(1))

    return list(dict.fromkeys(modules))  # Dedupe preserving order


def extract_exports(lines: list[str], limit: int = 200) -> list[str]:
    """Extract export statements."""
    exports: list[str] = []

    for raw in lines[:limit]:
        trimmed = raw.strip()
        if not trimmed.startswith("export") and "module.exports" not in trimmed:
            continue

        # Named export declaration
        decl = re.match(
            r'^export\s+(?:default\s+)?'
            r'(class|function|const|let|var|interface|type|enum)\s+'
            r'([A-Za-z0-9_$]+)',
            trimmed
        )
        if decl:
            exports.append(decl.group(2))
            continue

        # Brace export
        brace = re.match(r'^export\s+\{([^}]+)\}', trimmed)
        if brace:
            names = [n.strip() for n in brace.group(1).split(",") if n.strip()]
            exports.extend(names)
            continue

        if trimmed.startswith("export default"):
            exports.append("default")

        if "module.exports" in trimmed:
            exports.append("module.exports")

    return list(dict.fromkeys(exports))


def format_chunk_text(chunk: Chunk, file_path: str) -> str:
    """Format chunk with breadcrumb context."""
    breadcrumb = list(chunk.context)
    file_label = f"File: {file_path or 'unknown'}"

    has_file_label = any(
        isinstance(entry, str) and entry.startswith("File: ")
        for entry in breadcrumb
    )
    if not has_file_label:
        breadcrumb.insert(0, file_label)

    header = " > ".join(breadcrumb) if breadcrumb else file_label
    return f"{header}\n---\n{chunk.content}"


def build_anchor_chunk(file_path: str, content: str) -> Chunk:
    """Build an anchor chunk summarizing file structure."""
    lines = content.split("\n")
    top_comments = extract_top_comments(lines)
    imports = extract_imports(lines)
    exports = extract_exports(lines)

    # Extract preamble (first 30 non-blank lines or 1200 chars)
    preamble: list[str] = []
    non_blank = 0
    total_chars = 0
    for line in lines:
        if not line.strip():
            continue
        preamble.append(line)
        non_blank += 1
        total_chars += len(line)
        if non_blank >= 30 or total_chars >= 1200:
            break

    sections: list[str] = [f"File: {file_path}"]
    if imports:
        sections.append(f"Imports: {', '.join(imports)}")
    if exports:
        sections.append(f"Exports: {', '.join(exports)}")
    if top_comments:
        sections.append(f"Top comments:\n{chr(10).join(top_comments)}")
    if preamble:
        sections.append(f"Preamble:\n{chr(10).join(preamble)}")
    sections.append("---")
    sections.append("(anchor)")

    anchor_text = "\n\n".join(sections)
    approx_end_line = min(len(lines), max(1, non_blank or len(preamble) or 5))

    return Chunk(
        content=anchor_text,
        start_line=0,
        end_line=approx_end_line,
        type="block",
        context=[f"File: {file_path}", "Anchor"],
        chunk_index=-1,
        is_anchor=True,
    )


# --- Compatibility Layer ---

class Chunker(TreeSitterChunker):
    """Alias for backward compatibility."""
    pass


# Global singleton
_chunker: Chunker | None = None


def get_chunker() -> Chunker:
    """Get or create the global chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = Chunker()
    return _chunker
