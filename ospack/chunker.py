"""AST-based code chunking using tree-sitter.

Uses tree-sitter for semantic extraction of functions, classes, and methods
with accurate type classification and nesting context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .log import get_logger

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = get_logger(__name__)


# Map file extensions to tree-sitter language module names
# These map to tree-sitter-{name} packages
EXT_TO_TS_LANG: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "tsx",
    # Systems languages
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    # JVM languages
    ".java": "java",
    # Other languages
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    # Markup/Config
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    # Shell/Scripts
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
}

# Map language names to their module import names and function names
# Format: (module_name, language_function_name)
# Most modules use .language(), but some have special names
LANG_TO_MODULE: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "bash": ("tree_sitter_bash", "language"),
    "markdown": ("tree_sitter_markdown", "language"),
    "html": ("tree_sitter_html", "language"),
    "css": ("tree_sitter_css", "language"),
    "json": ("tree_sitter_json", "language"),
    "yaml": ("tree_sitter_yaml", "language"),
    "toml": ("tree_sitter_toml", "language"),
}


# Definition node types per language for semantic extraction
LANG_DEFINITION_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "function_expression",
        "generator_function_declaration",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "function_expression",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "function_expression",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "type_spec",  # Struct/interface definitions inside type_declaration
    },
    "rust": {
        "function_item",
        "struct_item",
        "impl_item",
        "trait_item",
        "enum_item",
        "type_item",
        "mod_item",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
        "enum_declaration",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "enum_specifier",
        "type_definition",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "template_declaration",
        "namespace_definition",
    },
    "c_sharp": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "constructor_declaration",
    },
    "ruby": {
        "method",
        "singleton_method",
        "class",
        "module",
    },
    "php": {
        "function_definition",
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
    },
    "kotlin": {
        "function_declaration",
        "class_declaration",
        "object_declaration",
        "interface_declaration",
    },
    "scala": {
        "function_definition",
        "class_definition",
        "object_definition",
        "trait_definition",
    },
    "swift": {
        "function_declaration",
        "class_declaration",
        "struct_declaration",
        "protocol_declaration",
        "enum_declaration",
    },
    "elixir": {
        "call",  # def, defp, defmodule are calls in Elixir
    },
    "haskell": {
        "function",
        "type_signature",
        "data_type",
        "newtype",
        "type_class_declaration",
    },
}

# Node types that indicate a class-like construct
CLASS_TYPES = {
    "class_definition",
    "class_declaration",
    "class_specifier",
    "class",
    "struct_specifier",
    "struct_declaration",
    "struct_item",
    "interface_declaration",
    "trait_declaration",
    "trait_item",
    "protocol_declaration",
    "module",
    "object_declaration",
    "object_definition",
    "impl_item",
    "enum_declaration",
    "enum_specifier",
    "enum_item",
    "type_declaration",
    "type_spec",
    "type_alias_declaration",
}

# Node types that indicate a function-like construct
FUNCTION_TYPES = {
    "function_definition",
    "function_declaration",
    "function_item",
    "method_definition",
    "method_declaration",
    "method",
    "singleton_method",
    "arrow_function",
    "function_expression",
    "generator_function_declaration",
    "constructor_declaration",
}

# Import/export node types per language for anchor chunks
IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement", "import_declaration"},
    "typescript": {"import_statement", "import_declaration"},
    "tsx": {"import_statement", "import_declaration"},
    "go": {"import_declaration", "import_spec"},
    "rust": {"use_declaration", "extern_crate_declaration"},
    "java": {"import_declaration"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include", "using_declaration"},
    "c_sharp": {"using_directive"},
    "ruby": {"require", "require_relative"},
    "php": {"use_declaration", "require_expression", "include_expression"},
}

EXPORT_TYPES: dict[str, set[str]] = {
    "javascript": {"export_statement", "export_declaration"},
    "typescript": {"export_statement", "export_declaration"},
    "tsx": {"export_statement", "export_declaration"},
    "rust": {"attribute_item"},  # #[no_mangle] pub, etc.
}

# Constants for chunk processing
OVERLAP_LINES = 5  # Lines of overlap between adjacent chunks
MAX_ANCHOR_SIZE = 2000  # Max chars for anchor chunk summary
MAX_CONTEXT_PREVIEW = 500  # Max chars for context_prev/next


@dataclass
class Chunk:
    """A semantic chunk of code."""

    content: str
    start_line: int
    end_line: int
    type: Literal["function", "class", "block", "other", "anchor"]
    context: list[str] = field(default_factory=list)
    chunk_index: int = -1
    is_anchor: bool = False
    # Context neighbors for search - populated after chunking
    context_prev: str | None = None  # Content of previous chunk (for search context)
    context_next: str | None = None  # Content of next chunk (for search context)

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
    """AST-based chunker using tree-sitter for semantic code extraction."""

    MAX_CHUNK_SIZE = 4000  # chars - split large functions
    MIN_CHUNK_SIZE = 50  # chars - skip trivial chunks

    def __init__(self):
        self._parsers: dict[str, Any] = {}

    def _get_parser(self, lang: str) -> Any | None:
        """Get or create a parser for the given language."""
        if lang in self._parsers:
            return self._parsers[lang]

        module_info = LANG_TO_MODULE.get(lang)
        if not module_info:
            logger.debug("No module mapping for language: %s", lang)
            return None

        module_name, func_name = module_info

        try:
            import importlib
            from tree_sitter import Language, Parser

            # Dynamically import the language module
            lang_module = importlib.import_module(module_name)
            lang_func = getattr(lang_module, func_name)
            language = Language(lang_func())
            parser = Parser(language)
            self._parsers[lang] = parser
            return parser
        except ImportError as e:
            logger.debug("Language module not installed for %s: %s", lang, e)
            return None
        except Exception as e:
            logger.debug("Failed to get parser for %s: %s", lang, e)
            return None

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        """Chunk a file into semantic units using tree-sitter."""
        if not content.strip():
            return []

        ext = Path(file_path).suffix.lower()
        lang = EXT_TO_TS_LANG.get(ext)

        if not lang:
            return self._fallback_chunk(file_path, content)

        parser = self._get_parser(lang)
        if not parser:
            return self._fallback_chunk(file_path, content)

        try:
            tree = parser.parse(bytes(content, "utf-8"))
            chunks = self._extract_semantic_chunks(tree, lang, content, file_path)

            if not chunks:
                return self._fallback_chunk(file_path, content)

            return chunks
        except Exception as e:
            logger.warning("Tree-sitter parsing failed for %s: %s", file_path, e)
            return self._fallback_chunk(file_path, content)

    def _extract_semantic_chunks(
        self, tree: Tree, lang: str, content: str, file_path: str
    ) -> list[Chunk]:
        """Extract semantic chunks from the syntax tree."""
        chunks: list[Chunk] = []
        definition_types = LANG_DEFINITION_TYPES.get(lang, set())

        if not definition_types:
            # No definition types configured, fall back
            return []

        # Build anchor chunk first (imports, exports, signatures summary)
        anchor = self._build_anchor_chunk(tree, lang, content, file_path)
        if anchor:
            chunks.append(anchor)

        # Collect all definition nodes
        definition_nodes: list[Node] = []
        self._collect_definitions(tree.root_node, definition_types, definition_nodes)

        # Convert nodes to chunks
        seen_ranges: set[tuple[int, int]] = set()
        for node in definition_nodes:
            range_key = (node.start_byte, node.end_byte)

            # Skip duplicates (e.g., decorated_definition contains function_definition)
            if range_key in seen_ranges:
                continue

            # Skip if this node is contained within another definition
            skip = False
            for other_start, other_end in seen_ranges:
                if other_start <= node.start_byte and node.end_byte <= other_end:
                    skip = True
                    break
            if skip:
                continue

            seen_ranges.add(range_key)
            chunk = self._node_to_chunk(node, content, file_path)
            if chunk and len(chunk.content) >= self.MIN_CHUNK_SIZE:
                chunks.append(chunk)

        # Fill gaps between definitions with "block" chunks
        chunks = self._fill_gaps(chunks, content, file_path)

        # Split oversized chunks
        chunks = self._split_large_chunks(chunks, content)

        # Sort by start line (anchor always first)
        chunks.sort(key=lambda c: (not c.is_anchor, c.start_line))

        # Add overlap between adjacent chunks
        chunks = self._add_overlap(chunks, content)

        # Populate context neighbors
        self._populate_context_neighbors(chunks)

        return chunks

    def _collect_definitions(
        self, node: Node, definition_types: set[str], result: list[Node]
    ) -> None:
        """Recursively collect definition nodes from the tree."""
        if node.type in definition_types:
            result.append(node)
            # Don't recurse into definitions to avoid nested duplicates
            # (e.g., inner functions) - they'll be captured as separate chunks
            # Actually, let's recurse to get nested definitions too
            for child in node.children:
                self._collect_definitions(child, definition_types, result)
        else:
            for child in node.children:
                self._collect_definitions(child, definition_types, result)

    def _node_to_chunk(self, node: Node, content: str, file_path: str) -> Chunk | None:
        """Convert a tree-sitter node to a Chunk."""
        try:
            chunk_content = content[node.start_byte : node.end_byte]
            chunk_type = self._classify_node(node)
            context = self._extract_context(node, file_path)

            # Get the name and add it to context
            name = self._get_node_name(node)
            if name:
                prefix = "Class" if chunk_type == "class" else "Function"
                if "method" in node.type.lower():
                    prefix = "Method"
                context.append(f"{prefix}: {name}")

            return Chunk(
                content=chunk_content,
                start_line=node.start_point[0] + 1,  # 0-indexed to 1-indexed
                end_line=node.end_point[0] + 1,
                type=chunk_type,
                context=context,
            )
        except Exception as e:
            logger.debug("Failed to convert node to chunk: %s", e)
            return None

    def _classify_node(self, node: Node) -> Literal["function", "class", "block", "other"]:
        """Classify a node as function, class, block, or other."""
        node_type = node.type

        if node_type in CLASS_TYPES:
            return "class"
        if node_type in FUNCTION_TYPES:
            return "function"

        # Check if it's a decorated definition
        if node_type == "decorated_definition":
            for child in node.children:
                if child.type in CLASS_TYPES:
                    return "class"
                if child.type in FUNCTION_TYPES:
                    return "function"

        return "other"

    def _get_node_name(self, node: Node) -> str | None:
        """Extract the name of a definition node."""
        # Try common field names
        for field_name in ("name", "declarator"):
            name_node = node.child_by_field_name(field_name)
            if name_node:
                # Handle nested declarators (C/C++)
                while name_node.type in ("function_declarator", "pointer_declarator"):
                    inner = name_node.child_by_field_name("declarator")
                    if inner:
                        name_node = inner
                    else:
                        break

                if name_node.type == "identifier" or name_node.type == "type_identifier":
                    return name_node.text.decode("utf-8") if name_node.text else None
                elif name_node.text:
                    return name_node.text.decode("utf-8")

        # Handle decorated_definition - look at the inner definition
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    return self._get_node_name(child)

        # Fallback: look for first identifier child
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8") if child.text else None

        return None

    def _extract_context(self, node: Node, file_path: str) -> list[str]:
        """Extract nesting context by walking up the tree."""
        context = [f"File: {file_path}"]
        parts: list[str] = []
        current = node.parent

        while current:
            if current.type in CLASS_TYPES:
                name = self._get_node_name(current)
                if name:
                    parts.append(f"Class: {name}")
            elif current.type in FUNCTION_TYPES:
                name = self._get_node_name(current)
                if name:
                    prefix = "Method" if "method" in current.type.lower() else "Function"
                    parts.append(f"{prefix}: {name}")
            elif current.type == "decorated_definition":
                # Check the inner definition
                for child in current.children:
                    if child.type in CLASS_TYPES:
                        name = self._get_node_name(child)
                        if name:
                            parts.append(f"Class: {name}")
                        break
                    elif child.type in FUNCTION_TYPES:
                        name = self._get_node_name(child)
                        if name:
                            parts.append(f"Function: {name}")
                        break

            current = current.parent

        # Reverse to get outermost first
        context.extend(reversed(parts))
        return context

    def _fill_gaps(
        self, chunks: list[Chunk], content: str, file_path: str
    ) -> list[Chunk]:
        """Fill gaps between definition chunks with block chunks."""
        if not chunks:
            # No definitions found, treat entire file as one block
            lines = content.split("\n")
            if content.strip():
                return [
                    Chunk(
                        content=content,
                        start_line=1,
                        end_line=len(lines),
                        type="block",
                        context=[f"File: {file_path}"],
                    )
                ]
            return []

        # Sort by start line
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        lines = content.split("\n")
        result: list[Chunk] = []
        last_end = 0

        for chunk in sorted_chunks:
            gap_start = last_end
            gap_end = chunk.start_line - 1

            # Add gap chunk if there's substantial content
            if gap_end > gap_start:
                gap_content = "\n".join(lines[gap_start:gap_end])
                if len(gap_content.strip()) >= self.MIN_CHUNK_SIZE:
                    result.append(
                        Chunk(
                            content=gap_content,
                            start_line=gap_start + 1,
                            end_line=gap_end,
                            type="block",
                            context=[f"File: {file_path}"],
                        )
                    )

            result.append(chunk)
            last_end = chunk.end_line

        # Add trailing content
        if last_end < len(lines):
            trailing_content = "\n".join(lines[last_end:])
            if len(trailing_content.strip()) >= self.MIN_CHUNK_SIZE:
                result.append(
                    Chunk(
                        content=trailing_content,
                        start_line=last_end + 1,
                        end_line=len(lines),
                        type="block",
                        context=[f"File: {file_path}"],
                    )
                )

        return result

    def _split_large_chunks(self, chunks: list[Chunk], content: str) -> list[Chunk]:
        """Split chunks that exceed MAX_CHUNK_SIZE."""
        result: list[Chunk] = []

        for chunk in chunks:
            if len(chunk.content) <= self.MAX_CHUNK_SIZE:
                result.append(chunk)
            else:
                result.extend(self._split_chunk(chunk))

        return result

    def _split_chunk(self, chunk: Chunk) -> list[Chunk]:
        """Split a large chunk at logical boundaries."""
        lines = chunk.content.split("\n")
        sub_chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_size = 0
        part_num = 1

        for i, line in enumerate(lines):
            current_lines.append(line)
            current_size += len(line) + 1

            # Check if we should split
            should_split = current_size >= self.MAX_CHUNK_SIZE * 0.8
            # Prefer splitting at blank lines or after certain patterns
            is_good_split_point = (
                line.strip() == ""
                or (i + 1 < len(lines) and not lines[i + 1].startswith((" ", "\t")))
            )

            if should_split and (is_good_split_point or current_size >= self.MAX_CHUNK_SIZE):
                sub_content = "\n".join(current_lines)
                lines_in_part = len(current_lines)
                sub_start = chunk.start_line + sum(
                    len(c.content.split("\n")) for c in sub_chunks
                )

                sub_chunks.append(
                    Chunk(
                        content=sub_content,
                        start_line=sub_start,
                        end_line=sub_start + lines_in_part - 1,
                        type=chunk.type,
                        context=chunk.context + [f"Part: {part_num}"],
                    )
                )
                current_lines = []
                current_size = 0
                part_num += 1

        # Add remaining lines
        if current_lines:
            sub_content = "\n".join(current_lines)
            sub_start = chunk.start_line + sum(
                len(c.content.split("\n")) for c in sub_chunks
            )
            context = chunk.context + ([f"Part: {part_num}"] if part_num > 1 else [])

            sub_chunks.append(
                Chunk(
                    content=sub_content,
                    start_line=sub_start,
                    end_line=chunk.end_line,
                    type=chunk.type,
                    context=context,
                )
            )

        return sub_chunks if sub_chunks else [chunk]

    def _build_anchor_chunk(
        self, tree: Tree, lang: str, content: str, file_path: str
    ) -> Chunk | None:
        """Build an anchor chunk summarizing file imports, exports, and signatures.

        The anchor chunk provides high-level context about what the file contains
        without the full implementation details.
        """
        import_types = IMPORT_TYPES.get(lang, set())
        export_types = EXPORT_TYPES.get(lang, set())
        definition_types = LANG_DEFINITION_TYPES.get(lang, set())

        imports: list[str] = []
        exports: list[str] = []
        signatures: list[str] = []

        # Walk the root level nodes only
        for node in tree.root_node.children:
            node_text = content[node.start_byte : node.end_byte]

            # Collect imports
            if node.type in import_types:
                imports.append(node_text.strip())
            # Collect exports
            elif node.type in export_types:
                # For exports, just grab the first line (signature)
                first_line = node_text.split("\n")[0].strip()
                exports.append(first_line)
            # Collect top-level definition signatures
            elif node.type in definition_types:
                sig = self._extract_signature(node, content)
                if sig:
                    signatures.append(sig)
            # Handle decorated definitions
            elif node.type == "decorated_definition":
                sig = self._extract_signature(node, content)
                if sig:
                    signatures.append(sig)

        # Build anchor content
        parts: list[str] = []

        if imports:
            parts.append("// Imports:\n" + "\n".join(imports[:20]))  # Limit to 20 imports

        if exports:
            parts.append("// Exports:\n" + "\n".join(exports[:10]))  # Limit to 10 exports

        if signatures:
            parts.append("// Definitions:\n" + "\n".join(signatures[:30]))  # Limit to 30

        if not parts:
            return None

        anchor_content = "\n\n".join(parts)

        # Truncate if too large
        if len(anchor_content) > MAX_ANCHOR_SIZE:
            anchor_content = anchor_content[:MAX_ANCHOR_SIZE] + "\n// ... (truncated)"

        return Chunk(
            content=anchor_content,
            start_line=1,
            end_line=1,  # Anchor is virtual, not tied to specific lines
            type="anchor",
            context=[f"File: {file_path}", "Anchor: Summary"],
            is_anchor=True,
        )

    def _extract_signature(self, node: Node, content: str) -> str | None:
        """Extract just the signature (first line) of a definition."""
        node_text = content[node.start_byte : node.end_byte]
        lines = node_text.split("\n")

        if not lines:
            return None

        # For most languages, the signature is the first line
        sig = lines[0].strip()

        # For Python decorated definitions, include decorators and def line
        if node.type == "decorated_definition":
            sig_lines = []
            for line in lines:
                stripped = line.strip()
                sig_lines.append(stripped)
                if stripped.startswith(("def ", "class ", "async def ")):
                    break
            sig = " ".join(sig_lines)

        # For multi-line signatures (e.g., long parameter lists), grab more
        if sig.endswith("(") or (sig.count("(") > sig.count(")")):
            # Incomplete signature, grab until we close the paren
            paren_depth = sig.count("(") - sig.count(")")
            for line in lines[1:5]:  # Max 5 more lines
                sig += " " + line.strip()
                paren_depth += line.count("(") - line.count(")")
                if paren_depth <= 0:
                    break

        # Truncate very long signatures
        if len(sig) > 200:
            sig = sig[:200] + "..."

        return sig

    def _add_overlap(self, chunks: list[Chunk], content: str) -> list[Chunk]:
        """Add overlap lines between adjacent chunks for context continuity.

        Each chunk gets OVERLAP_LINES from the previous and next chunk prepended/appended.
        """
        if len(chunks) <= 1 or OVERLAP_LINES <= 0:
            return chunks

        lines = content.split("\n")
        result: list[Chunk] = []

        for i, chunk in enumerate(chunks):
            # Skip anchor chunks - they don't need overlap
            if chunk.is_anchor:
                result.append(chunk)
                continue

            new_content_parts: list[str] = []

            # Add overlap from previous chunk (if not anchor)
            if i > 0 and not chunks[i - 1].is_anchor:
                prev_chunk = chunks[i - 1]
                # Get last OVERLAP_LINES from previous chunk
                prev_end_line = prev_chunk.end_line
                overlap_start = max(0, prev_end_line - OVERLAP_LINES)
                overlap_lines = lines[overlap_start:prev_end_line]
                if overlap_lines:
                    new_content_parts.append(
                        f"// ... (context from lines {overlap_start + 1}-{prev_end_line})\n"
                        + "\n".join(overlap_lines)
                    )

            # Add the chunk's own content
            new_content_parts.append(chunk.content)

            # Add overlap from next chunk (if exists and not anchor)
            if i + 1 < len(chunks) and not chunks[i + 1].is_anchor:
                next_chunk = chunks[i + 1]
                # Get first OVERLAP_LINES from next chunk
                next_start_line = next_chunk.start_line - 1  # 0-indexed
                overlap_end = min(len(lines), next_start_line + OVERLAP_LINES)
                overlap_lines = lines[next_start_line:overlap_end]
                if overlap_lines:
                    new_content_parts.append(
                        "\n".join(overlap_lines)
                        + f"\n// ... (continues at line {overlap_end + 1})"
                    )

            # Create new chunk with overlap
            result.append(
                Chunk(
                    content="\n\n".join(new_content_parts),
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    type=chunk.type,
                    context=chunk.context,
                    is_anchor=chunk.is_anchor,
                )
            )

        return result

    def _populate_context_neighbors(self, chunks: list[Chunk]) -> None:
        """Populate context_prev and context_next for each chunk.

        This allows search results to include surrounding context without
        fetching additional chunks.
        """
        for i, chunk in enumerate(chunks):
            # Previous context (skip anchors)
            if i > 0:
                prev_chunk = chunks[i - 1]
                if not prev_chunk.is_anchor:
                    preview = prev_chunk.content[-MAX_CONTEXT_PREVIEW:]
                    if len(prev_chunk.content) > MAX_CONTEXT_PREVIEW:
                        preview = "..." + preview
                    chunk.context_prev = preview

            # Next context (skip anchors)
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                if not next_chunk.is_anchor:
                    preview = next_chunk.content[:MAX_CONTEXT_PREVIEW]
                    if len(next_chunk.content) > MAX_CONTEXT_PREVIEW:
                        preview = preview + "..."
                    chunk.context_next = preview

    def _fallback_chunk(self, file_path: str, content: str) -> list[Chunk]:
        """Simple line-based fallback for unsupported languages."""
        if not content.strip():
            return []

        lines = content.split("\n")
        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_size = 0
        start_line = 0

        for i, line in enumerate(lines):
            current_lines.append(line)
            current_size += len(line) + 1

            if current_size >= self.MAX_CHUNK_SIZE:
                chunk_content = "\n".join(current_lines)
                if len(chunk_content.strip()) >= self.MIN_CHUNK_SIZE:
                    chunks.append(
                        Chunk(
                            content=chunk_content,
                            start_line=start_line + 1,
                            end_line=i + 1,
                            type="block",
                            context=[f"File: {file_path}"],
                        )
                    )
                current_lines = []
                current_size = 0
                start_line = i + 1

        # Add remaining lines
        if current_lines:
            chunk_content = "\n".join(current_lines)
            if len(chunk_content.strip()) >= self.MIN_CHUNK_SIZE:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        start_line=start_line + 1,
                        end_line=len(lines),
                        type="block",
                        context=[f"File: {file_path}"],
                    )
                )

        return chunks


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
