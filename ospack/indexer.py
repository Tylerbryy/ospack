"""Hybrid search index with Incremental Updates and State Persistence."""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from rank_bm25 import BM25Okapi

from .chunker import Chunker
from .embedder import get_embedder, get_reranker
from .log import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Exclude patterns - everything else is indexed if it's a text file
EXCLUDE_PATTERNS = [
    # Version control
    ".git/**", ".svn/**", ".hg/**",
    # Dependencies
    "node_modules/**", ".venv/**", "venv/**", "vendor/**",
    # Build outputs
    "dist/**", "build/**", "target/**", "__pycache__/**",
    "*.pyc", "*.pyo",
    # IDE/Editor
    ".idea/**", ".vscode/**", "*.swp", "*.swo",
    # Framework caches
    ".next/**", ".turbo/**", "coverage/**",
    # Lock files (large, low semantic value)
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Pipfile.lock",
    # Minified files
    "*.min.js", "*.min.css",
]

# File size limit (1MB default) - skip very large files
MAX_FILE_SIZE = int(os.environ.get("OSPACK_MAX_FILE_SIZE", 1024 * 1024))


def is_text_file(path: Path, sample_size: int = 8192) -> bool:
    """Check if a file is text (not binary) using null-byte heuristic.

    Same approach as Git - fast and reliable for code files.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" not in chunk
    except Exception:
        return False

MAX_WORKERS = min(os.cpu_count() or 4, 8)


def get_repo_hash(path: str) -> str:
    """Generate a hash for the repository path."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]


# Schema version - increment when adding/removing fields to force reindex
SCHEMA_VERSION = 2  # v2: added is_anchor, context_prev, context_next, node_type


def _chunk_file_worker(args: tuple[str, float]) -> list[dict]:
    """Worker function for parallel chunking.

    Each worker creates its own Chunker (parsers not picklable).
    Returns dicts ready for embedding.
    """
    file_path, mtime = args
    try:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")
        chunker = Chunker()
        chunks = chunker.chunk(str(file_path), content)

        return [{
            "file_path": str(file_path),
            "content": c.content,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "node_type": c.node_type,
            "name": c.name or "",
            "is_anchor": c.is_anchor,
            "context_prev": c.context_prev or "",
            "context_next": c.context_next or "",
            "last_modified": mtime,
        } for c in chunks]
    except Exception:
        return []


def code_tokenize(text: str) -> list[str]:
    """Code-aware tokenizer for BM25.

    Handles camelCase, snake_case, dots, preserves original tokens.
    """
    text = text.lower()
    raw_tokens = re.findall(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", text)

    tokens = []
    seen = set()

    for t in raw_tokens:
        if t not in seen:
            tokens.append(t)
            seen.add(t)

        # Snake case split
        if "_" in t:
            for part in t.split("_"):
                if part and part not in seen:
                    tokens.append(part)
                    seen.add(part)

        # Dot split (file.path.ts)
        if "." in t:
            for part in t.split("."):
                if part and part not in seen:
                    tokens.append(part)
                    seen.add(part)

        # camelCase split
        camel_parts = re.findall(r"[a-z]+|[0-9]+", t)
        if len(camel_parts) > 1:
            for part in camel_parts:
                if part and part not in seen:
                    tokens.append(part)
                    seen.add(part)

    return tokens


def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Combine multiple ranked lists using RRF."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.get("id")
            if not doc_id:
                continue

            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [{**docs[did], "rrf_score": scores[did]} for did in sorted_ids]


class Indexer:
    """Hybrid search index with incremental updates."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.storage_dir = Path.home() / ".ospack" / "index" / get_repo_hash(str(self.root_dir))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Schema version file
        self.version_path = self.storage_dir / "schema_version"

        # LanceDB setup
        self.db = lancedb.connect(str(self.storage_dir / "lancedb"))
        self.embedder = get_embedder()
        self._table = None

        # BM25 State - persisted to disk
        self.bm25_path = self.storage_dir / "bm25.pkl"
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []  # Ordered list of IDs matching BM25 corpus
        self._bm25_map: dict[str, dict] = {}  # ID -> metadata (lightweight, no vectors)

    def _get_source_files(self) -> dict[str, float]:
        """Scan directory returning {filepath: mtime}.

        Indexes ALL text files, not just known extensions.
        Uses exclusion patterns and binary detection to filter.
        """
        exclude_spec = PathSpec.from_lines(GitWildMatchPattern, EXCLUDE_PATTERNS)

        results = {}
        for path in self.root_dir.rglob("*"):
            if not path.is_file():
                continue

            rel = str(path.relative_to(self.root_dir))

            # Check exclusion patterns first (fast)
            if exclude_spec.match_file(rel):
                continue

            # Check file size limit
            try:
                stat = path.stat()
                if stat.st_size > MAX_FILE_SIZE:
                    logger.debug("Skipping large file (%d bytes): %s", stat.st_size, rel)
                    continue
                if stat.st_size == 0:
                    continue  # Skip empty files
            except OSError:
                continue

            # Check if it's a text file (binary detection)
            if not is_text_file(path):
                continue

            results[str(path)] = stat.st_mtime
        return results

    def _load_resources(self):
        """Load DB table and BM25 index from disk."""
        # Load LanceDB Table
        if "chunks" in self.db.table_names():
            self._table = self.db.open_table("chunks")

        # Load BM25 from pickle (avoids RAM explosion from loading full DB)
        if self.bm25_path.exists() and self._bm25 is None:
            try:
                with open(self.bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self._bm25 = data["model"]
                    self._bm25_ids = data["ids"]
                    self._bm25_map = data["map"]
                logger.debug("Loaded BM25 index from disk (%d docs).", len(self._bm25_ids))
            except Exception:
                logger.warning("Corrupt BM25 index, will rebuild.")
                self._bm25 = None
                self._bm25_ids = []
                self._bm25_map = {}

    def _save_bm25(self):
        """Persist BM25 state to disk."""
        with open(self.bm25_path, "wb") as f:
            pickle.dump({
                "model": self._bm25,
                "ids": self._bm25_ids,
                "map": self._bm25_map,
            }, f)
        logger.debug("Saved BM25 index to disk.")

    def _check_schema_version(self) -> bool:
        """Check if schema version matches. Returns True if rebuild needed."""
        if not self.version_path.exists():
            return True  # No version file, need rebuild

        try:
            stored_version = int(self.version_path.read_text().strip())
            return stored_version != SCHEMA_VERSION
        except (ValueError, OSError):
            return True  # Corrupt or unreadable, need rebuild

    def _save_schema_version(self) -> None:
        """Save current schema version."""
        self.version_path.write_text(str(SCHEMA_VERSION))

    def build_index(self, force: bool = False) -> int:
        """Build or incrementally update the index.

        Returns number of chunks indexed.
        """
        self._load_resources()

        # Check schema version - force rebuild if outdated
        if self._check_schema_version():
            logger.info("Schema version changed, forcing full rebuild...")
            force = True

        if force:
            # Full rebuild requested
            if self._table:
                self.db.drop_table("chunks")
                self._table = None
            if self.bm25_path.exists():
                self.bm25_path.unlink()
            self._bm25 = None
            self._bm25_ids = []
            self._bm25_map = {}

        current_files = self._get_source_files()

        # Get existing state from DB
        existing_files: dict[str, float] = {}
        if self._table:
            try:
                # Only fetch path and mtime, not vectors (saves RAM)
                df = self._table.search().select(["file_path", "last_modified"]).limit(None).to_pandas()
                if not df.empty:
                    df = df.drop_duplicates(subset=["file_path"])
                    existing_files = dict(zip(df["file_path"], df["last_modified"]))
            except Exception:
                logger.warning("Could not read existing index state.")

        # Calculate Delta
        to_add: list[tuple[str, float]] = []
        to_remove: list[str] = []

        for path, mtime in current_files.items():
            if path not in existing_files:
                to_add.append((path, mtime))
            elif mtime > existing_files[path]:
                to_remove.append(path)  # Remove old version first
                to_add.append((path, mtime))

        for path in existing_files:
            if path not in current_files:
                to_remove.append(path)

        if not to_add and not to_remove:
            logger.debug("Index is up to date.")
            return 0

        logger.info("Syncing index: +%d files, -%d files", len(to_add), len(to_remove))

        # 1. Handle Removals
        if to_remove and self._table:
            # Delete from LanceDB
            placeholders = ", ".join([f"'{p}'" for p in to_remove])
            try:
                self._table.delete(f"file_path IN ({placeholders})")
                logger.debug("Removed %d files from LanceDB.", len(to_remove))
            except Exception as e:
                logger.warning("Failed to delete from LanceDB: %s", e)

        # 2. Handle Additions
        new_records: list[dict] = []
        if to_add:
            logger.info("Chunking %d files...", len(to_add))
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(_chunk_file_worker, arg) for arg in to_add]
                completed = 0
                for future in as_completed(futures):
                    new_records.extend(future.result())
                    completed += 1
                    if completed % 50 == 0:
                        logger.debug("Chunked %d/%d files...", completed, len(to_add))

            logger.info("Chunked %d files -> %d chunks", len(to_add), len(new_records))

            if new_records:
                # Embedding (sequential in main process)
                logger.info("Embedding %d chunks...", len(new_records))
                texts = [r["content"] for r in new_records]
                vectors = self.embedder.embed(texts)

                for r, v in zip(new_records, vectors):
                    r["vector"] = v
                    # Deterministic ID for stability
                    r["id"] = hashlib.md5(
                        f"{r['file_path']}:{r['start_line']}-{r['end_line']}".encode()
                    ).hexdigest()

                # Add to LanceDB
                if self._table:
                    self._table.add(new_records)
                else:
                    self._table = self.db.create_table("chunks", new_records)

                logger.info("Added %d chunks to index.", len(new_records))

        # 3. Rebuild BM25 (corpus stats changed)
        if to_add or to_remove:
            self._rebuild_bm25()

        # Save schema version after successful build
        self._save_schema_version()

        return len(new_records)

    def _rebuild_bm25(self):
        """Rebuild BM25 from current LanceDB state and persist."""
        if not self._table:
            return

        logger.info("Rebuilding BM25 index...")

        # Only fetch text fields, NOT vectors (huge RAM savings)
        df = self._table.search().select([
            "id", "file_path", "content", "start_line", "end_line", "name",
            "node_type", "is_anchor", "context_prev", "context_next"
        ]).limit(None).to_pandas()

        records = df.to_dict("records")

        tokenized_corpus = []
        self._bm25_ids = []
        self._bm25_map = {}

        for r in records:
            doc_id = r["id"]
            self._bm25_ids.append(doc_id)
            self._bm25_map[doc_id] = r

            # Tokenize for BM25
            text = f"{r['name']} {r['content']} {r['file_path']}"
            tokenized_corpus.append(code_tokenize(text))

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._save_bm25()

        logger.info("BM25 index built with %d documents.", len(records))

    def _dense_search(self, query: str, limit: int) -> list[dict]:
        """Vector similarity search via LanceDB."""
        if not self._table:
            return []

        try:
            query_vec = self.embedder.embed_single(query)
            results = self._table.search(query_vec).limit(limit).to_list()

            return [{
                "id": r.get("id", ""),
                "file_path": r["file_path"],
                "content": r["content"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "name": r["name"],
                "node_type": r.get("node_type", ""),
                "is_anchor": r.get("is_anchor", False),
                "context_prev": r.get("context_prev", ""),
                "context_next": r.get("context_next", ""),
                "score": 1 - r.get("_distance", 0),
            } for r in results]
        except Exception:
            logger.error("Dense search failed", exc_info=True)
            return []

    def _sparse_search(self, query: str, limit: int) -> list[dict]:
        """BM25 keyword search."""
        if not self._bm25 or not self._bm25_ids:
            return []

        try:
            tokens = code_tokenize(query)
            scores = self._bm25.get_scores(tokens)

            # Get top indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:limit]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc_id = self._bm25_ids[idx]
                    doc = self._bm25_map[doc_id]
                    results.append({
                        "id": doc_id,
                        "file_path": doc["file_path"],
                        "content": doc["content"],
                        "start_line": doc["start_line"],
                        "end_line": doc["end_line"],
                        "name": doc["name"],
                        "node_type": doc.get("node_type", ""),
                        "is_anchor": doc.get("is_anchor", False),
                        "context_prev": doc.get("context_prev", ""),
                        "context_next": doc.get("context_next", ""),
                        "score": float(scores[idx]),
                    })

            return results
        except Exception:
            logger.error("Sparse search failed", exc_info=True)
            return []

    def search(
        self,
        query: str,
        limit: int = 10,
        rerank: bool = True,
        hybrid: bool = True,
    ) -> list[dict]:
        """Hybrid search: dense + sparse with optional reranking."""
        self._load_resources()

        if not self._table:
            self.build_index()

        if not self._table:
            return []

        # Fetch more candidates for reranking
        fetch_limit = limit * 5 if rerank else limit * 2

        if hybrid and self._bm25:
            dense_results = self._dense_search(query, fetch_limit)
            sparse_results = self._sparse_search(query, fetch_limit)
            results = reciprocal_rank_fusion([dense_results, sparse_results])
        else:
            results = self._dense_search(query, fetch_limit)

        # Rerank with cross-encoder
        if rerank and results:
            reranker = get_reranker()
            results = reranker.rerank(query, results, top_k=limit)
        else:
            results = results[:limit]

        return results

    def close(self):
        """Close the database connection."""
        pass  # LanceDB handles cleanup automatically


# Global singleton
_indexer: dict[str, Indexer] = {}


def get_indexer(root_dir: str) -> Indexer:
    """Get or create an indexer for the given root directory."""
    root_dir = str(Path(root_dir).resolve())
    if root_dir not in _indexer:
        _indexer[root_dir] = Indexer(root_dir)
    return _indexer[root_dir]
