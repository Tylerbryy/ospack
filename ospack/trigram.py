"""Trigram-based code search index (Blackbird/Google Code Search inspired).

Uses trigrams (3-character sequences) to enable exact pattern matching
including punctuation, which BM25 tokenization loses.

Features:
- SQLite-backed for memory efficiency
- WAL mode for concurrency
- Content-hash deduplication
- Regex support via sre_parse AST literal extraction
"""

from __future__ import annotations

import bisect
import hashlib
import re
import sqlite3
import sre_constants
import sre_parse
from pathlib import Path
from typing import Any

from .log import get_logger

logger = get_logger(__name__)


def extract_trigrams(text: str) -> set[str]:
    """Extract unique 3-character sequences from text."""
    if len(text) < 3:
        return set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def extract_context(
    content: str, match_start: int, match_end: int, context_lines: int = 2
) -> str:
    """Extract lines surrounding the match for LLM context.

    Args:
        content: The full file content
        match_start: Start index of the match
        match_end: End index of the match
        context_lines: Number of lines before/after to include
    """
    # Find line boundaries
    line_starts = [0] + [i + 1 for i, char in enumerate(content) if char == "\n"]

    # Bisect to find which line the match is in
    match_line_idx = bisect.bisect_right(line_starts, match_start) - 1

    start_line_idx = max(0, match_line_idx - context_lines)
    end_line_idx = min(len(line_starts), match_line_idx + context_lines + 2)

    start_char = line_starts[start_line_idx]
    end_char = (
        line_starts[end_line_idx] if end_line_idx < len(line_starts) else len(content)
    )

    return content[start_char:end_char]


def extract_literals_from_regex(pattern: str, min_len: int = 3) -> list[str]:
    """Parse regex AST to find guaranteed literal substrings.

    Uses Python's internal sre_parse module to walk the regex AST
    and extract contiguous literal sequences.

    Examples:
        r"function\\s+login" -> ["function", "login"]
        r"useState\\(.+\\)" -> ["useState("]
        r"\\d+" -> []  (no literals, must scan all)
        r"class [A-Z]\\w+" -> ["class "]
    """
    try:
        parsed = sre_parse.parse(pattern)
    except sre_constants.error:
        return []

    required_literals: list[str] = []
    current_chunk: list[str] = []

    def flush_chunk():
        nonlocal current_chunk
        if len(current_chunk) >= min_len:
            required_literals.append("".join(current_chunk))
        current_chunk = []

    # Iterate AST: LITERAL ops accumulate, anything else breaks the chain
    for op, value in parsed:
        if op == sre_constants.LITERAL:
            current_chunk.append(chr(value))
        else:
            flush_chunk()

    flush_chunk()
    return required_literals


class TrigramIndex:
    """SQLite-backed trigram index for exact/regex code search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not self._conn:
            # check_same_thread=False allows background watcher threads to use this connection
            # WAL mode provides safe concurrent reads, and writes are serialized
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # Enable WAL mode for concurrency and speed
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        return self._conn

    def _init_db(self):
        """Initialize the SQLite schema."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE,
                    content_hash TEXT,
                    content TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash)"
            )

            # Composite primary key ensures automatic deduplication
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trigrams (
                    trigram TEXT,
                    file_id INTEGER,
                    PRIMARY KEY (trigram, file_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trigram_t ON trigrams(trigram)"
            )

    def add(self, path: str, content: str):
        """Add a file to the index.

        Uses content hashing to skip trigram indexing if content hasn't changed.
        """
        conn = self._get_conn()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 1. Check if content already exists (deduplication)
        cursor = conn.execute(
            "SELECT id FROM files WHERE content_hash = ?", (content_hash,)
        )
        existing = cursor.fetchone()

        if existing:
            # Content exists, just update path mapping
            cursor.execute(
                "INSERT OR REPLACE INTO files (path, content_hash, content) VALUES (?, ?, ?)",
                (path, content_hash, content),
            )
            return

        # 2. Insert new file
        cursor.execute(
            "INSERT OR REPLACE INTO files (path, content_hash, content) VALUES (?, ?, ?)",
            (path, content_hash, content),
        )
        file_id = cursor.lastrowid

        # 3. Bulk Insert Trigrams
        trigrams = extract_trigrams(content)
        if trigrams:
            data = [(t, file_id) for t in trigrams]
            cursor.executemany(
                "INSERT OR IGNORE INTO trigrams (trigram, file_id) VALUES (?, ?)", data
            )

    def remove(self, path: str):
        """Remove a file and its trigrams from the index."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT id FROM files WHERE path = ?", (path,))
        row = cursor.fetchone()
        if row:
            file_id = row[0]
            conn.execute("DELETE FROM trigrams WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def begin_bulk_operation(self):
        """Drop indexes to speed up massive inserts.

        When inserting 10M+ rows, the B-Tree rebalancing on every insert
        is the main bottleneck. Dropping the index first, then rebuilding
        after all inserts, is much faster (sort once vs rebalance 10M times).
        """
        conn = self._get_conn()
        conn.execute("DROP INDEX IF EXISTS idx_trigram_t")
        conn.commit()

    def end_bulk_operation(self):
        """Recreate indexes after massive inserts."""
        conn = self._get_conn()
        logger.info("Rebuilding trigram B-Tree index...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trigram_t ON trigrams(trigram)")
        conn.commit()

    def commit(self):
        """Commit changes to disk."""
        if self._conn:
            self._conn.commit()

    def search(
        self, query: str, regex: bool = False, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search for code patterns.

        Strategy:
        1. Extract reliable trigrams from the query.
        2. Use SQL INTERSECT to find files containing ALL those trigrams.
        3. Verify the actual matches in Python (regex or substring).
        """
        literals: list[str] = []

        # Step 1: Analyze Query
        if regex:
            literals = extract_literals_from_regex(query)
            if not literals and query != "":
                # Query too broad (e.g., ".*"), limit scan
                logger.warning("Regex has no literals, limiting scan")
        else:
            literals = [query]

        # Collect all trigrams from all literals
        search_trigrams: set[str] = set()
        for lit in literals:
            search_trigrams.update(extract_trigrams(lit))

        trigram_list = list(search_trigrams)

        # Step 2: SQL Filtering (The "Blackbird" approach)
        candidate_ids = self._get_candidate_ids(trigram_list)

        # Step 3: Verification & Context Extraction
        results: list[dict[str, Any]] = []
        conn = self._get_conn()

        # Compile regex if needed
        matcher = re.compile(query, re.MULTILINE) if regex else None

        for file_id in candidate_ids:
            if len(results) >= limit:
                break

            row = conn.execute(
                "SELECT path, content FROM files WHERE id = ?", (file_id,)
            ).fetchone()
            if not row:
                continue

            path, content = row
            matches: list[tuple[int, int]] = []

            if regex:
                if matcher:
                    for m in matcher.finditer(content):
                        matches.append((m.start(), m.end()))
            else:
                # Standard substring search
                start = 0
                while True:
                    idx = content.find(query, start)
                    if idx == -1:
                        break
                    matches.append((idx, idx + len(query)))
                    start = idx + 1

            # Format results
            for start_pos, end_pos in matches[:5]:  # Limit matches per file
                if len(results) >= limit:
                    break

                ctx = extract_context(content, start_pos, end_pos)
                results.append({
                    "file": path,
                    "line": content[:start_pos].count("\n") + 1,
                    "match": content[start_pos:end_pos],
                    "context": ctx,
                })

        return results

    def _get_candidate_ids(self, trigrams: list[str]) -> list[int]:
        """Generate SQL INTERSECT query to find files with ALL trigrams."""
        conn = self._get_conn()

        if not trigrams:
            # No trigrams (short query or regex with no literals)
            # Limit to 100 files to prevent timeout
            cursor = conn.execute("SELECT id FROM files LIMIT 100")
            return [row[0] for row in cursor.fetchall()]

        # Construct INTERSECT query
        query_parts = ["SELECT file_id FROM trigrams WHERE trigram = ?"] * len(trigrams)
        full_query = " INTERSECT ".join(query_parts)

        cursor = conn.execute(full_query, trigrams)
        return [row[0] for row in cursor.fetchall()]

    def get_stats(self) -> dict[str, int]:
        """Get index statistics."""
        conn = self._get_conn()
        files_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        trigrams_count = conn.execute(
            "SELECT COUNT(DISTINCT trigram) FROM trigrams"
        ).fetchone()[0]
        return {"files": files_count, "unique_trigrams": trigrams_count}

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Global singleton
_trigram_indexes: dict[str, TrigramIndex] = {}


def get_trigram_index(root_dir: str) -> TrigramIndex:
    """Get or create a trigram index for the given root directory."""
    from .indexer import get_repo_hash

    root_dir = str(Path(root_dir).resolve())
    if root_dir not in _trigram_indexes:
        storage_dir = Path.home() / ".ospack" / "index" / get_repo_hash(root_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = storage_dir / "trigram.db"
        _trigram_indexes[root_dir] = TrigramIndex(str(db_path))
    return _trigram_indexes[root_dir]
