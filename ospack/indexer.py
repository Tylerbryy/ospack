"""Hybrid search index combining dense vectors (LanceDB) + sparse (BM25)."""

from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from rank_bm25 import BM25Okapi

from .chunker import Chunk, Chunker
from .embedder import get_embedder, get_reranker
from .log import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# File patterns to include
INCLUDE_PATTERNS = [
    "**/*.py",
    "**/*.ts",
    "**/*.tsx",
    "**/*.js",
    "**/*.jsx",
    "**/*.rs",
    "**/*.go",
    "**/*.java",
    "**/*.c",
    "**/*.h",
    "**/*.cpp",
    "**/*.hpp",
]

# Patterns to exclude
EXCLUDE_PATTERNS = [
    "node_modules/**",
    ".git/**",
    "dist/**",
    "build/**",
    ".next/**",
    ".turbo/**",
    "__pycache__/**",
    ".venv/**",
    "venv/**",
    "*.pyc",
    "coverage/**",
    ".pytest_cache/**",
]

# Number of workers for parallel chunking
MAX_WORKERS = min(os.cpu_count() or 4, 8)


def _chunk_file(file_path: str) -> list[tuple[str, Chunk]]:
    """Chunk a single file (for parallel processing).

    Returns list of (file_path, chunk) tuples.
    Each worker creates its own Chunker since parsers aren't picklable.
    """
    try:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")

        # Create chunker per-worker (parsers not picklable)
        chunker = Chunker()
        chunks = chunker.chunk(file_path, content)

        return [(file_path, chunk) for chunk in chunks]
    except Exception:
        return []


def get_repo_hash(root_dir: str) -> str:
    """Generate a hash for the repository path."""
    return hashlib.sha256(root_dir.encode()).hexdigest()[:16]


def code_tokenize(text: str) -> list[str]:
    """
    Code-aware tokenizer that handles:
    - camelCase: userController -> [user, Controller, userController]
    - snake_case: get_user_by_id -> [get, user, by, id, get_user_by_id]
    - dots: file.path.ts -> [file, path, ts]
    - Preserves original tokens for exact matching
    """
    tokens = []

    # Split on whitespace, punctuation (keeping dots for filenames)
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*", text.lower())

    for token in raw_tokens:
        # Add the original token
        tokens.append(token)

        # Split on dots (file.path.ts)
        if "." in token:
            parts = token.split(".")
            tokens.extend(parts)

        # Split on underscores (snake_case)
        if "_" in token:
            parts = token.split("_")
            tokens.extend(p for p in parts if p)

        # Split camelCase (getUserById -> get, User, By, Id)
        camel_parts = re.findall(r"[a-z]+|[A-Z][a-z]*|[0-9]+", token)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen and len(t) > 1:  # Skip single chars
            seen.add(t)
            unique.append(t)

    return unique


def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) across all lists.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.get("id") or f"{doc['file_path']}:{doc['start_line']}"

            # RRF formula
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

            # Keep the doc data (latest wins)
            if doc_id not in docs:
                docs[doc_id] = doc.copy()

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    result = []
    for doc_id in sorted_ids:
        doc = docs[doc_id]
        doc["rrf_score"] = scores[doc_id]
        result.append(doc)

    return result


class Indexer:
    """Hybrid search index: dense (LanceDB) + sparse (BM25)."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.db_path = Path.home() / ".ospack" / "lancedb" / get_repo_hash(str(self.root_dir))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(self.db_path))
        self.embedder = get_embedder()
        self._table = None

        # BM25 index (built alongside vector index)
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []

    def _get_source_files(self) -> list[Path]:
        """Get all source files matching patterns."""
        include_spec = PathSpec.from_lines(GitWildMatchPattern, INCLUDE_PATTERNS)
        exclude_spec = PathSpec.from_lines(GitWildMatchPattern, EXCLUDE_PATTERNS)

        files = []
        for path in self.root_dir.rglob("*"):
            if not path.is_file():
                continue
            rel_path = str(path.relative_to(self.root_dir))
            if include_spec.match_file(rel_path) and not exclude_spec.match_file(rel_path):
                files.append(path)

        return sorted(files)

    def _build_bm25_index(self, records: list[dict]):
        """Build BM25 index from records using code-aware tokenization."""
        self._bm25_docs = records

        # Tokenize all documents
        tokenized_corpus = []
        for record in records:
            # Combine content + name + file path for BM25
            text = f"{record['name']} {record['content']} {record['file_path']}"
            tokens = code_tokenize(text)
            tokenized_corpus.append(tokens)

        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.debug("BM25 index built with %d documents", len(records))

    def build_index(self, force: bool = False) -> int:
        """Build or rebuild the index.

        Uses parallel chunking for CPU-bound tree-sitter parsing,
        then sequential embedding (ML model not thread-safe).
        """
        table_names = self.db.table_names()

        if "chunks" in table_names and not force:
            self._table = self.db.open_table("chunks")
            # Check if index needs update
            if not self._needs_update():
                # Load existing data for BM25
                self._load_bm25_from_table()
                logger.debug("Index is up to date.")
                return 0

        logger.info("Building index...")
        source_files = self._get_source_files()
        logger.info("Found %d source files", len(source_files))

        # Phase 1: Parallel chunking (CPU-bound, tree-sitter parsing)
        all_chunks: list[tuple[str, Chunk]] = []
        file_paths = [str(f) for f in source_files]

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_chunk_file, fp): fp for fp in file_paths}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                all_chunks.extend(result)
                completed += 1
                if completed % 50 == 0:
                    logger.debug("Chunked %d/%d files...", completed, len(source_files))

        logger.info("Chunked %d files -> %d chunks", len(source_files), len(all_chunks))

        if not all_chunks:
            logger.warning("No chunks to index")
            return 0

        # Phase 2: Sequential embedding (ML model not thread-safe)
        logger.info("Generating embeddings...")
        texts = [chunk.content for _, chunk in all_chunks]
        embeddings = self.embedder.embed(texts)

        # Phase 3: Build records with embeddings
        all_records = []
        file_mtimes: dict[str, float] = {}  # Cache mtime lookups
        for (file_path, chunk), embedding in zip(all_chunks, embeddings, strict=False):
            if file_path not in file_mtimes:
                file_mtimes[file_path] = Path(file_path).stat().st_mtime
            record = {
                "id": hashlib.md5(
                    f"{file_path}:{chunk.start_line}-{chunk.end_line}".encode()
                ).hexdigest(),
                "file_path": file_path,
                "content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "node_type": chunk.node_type,
                "name": chunk.name or "",
                "last_modified": file_mtimes[file_path],
                "vector": embedding,
            }
            all_records.append(record)

        if not all_records:
            logger.warning("No chunks to index")
            return 0

        # Drop existing table and create new one
        if "chunks" in table_names:
            self.db.drop_table("chunks")

        self._table = self.db.create_table("chunks", all_records)

        # Build BM25 index
        self._build_bm25_index(all_records)

        logger.info("Index built with %d chunks", len(all_records))
        return len(all_records)

    def _load_bm25_from_table(self):
        """Load BM25 index from existing LanceDB table."""
        if self._table is None:
            return

        try:
            df = self._table.to_pandas()
            records = df.to_dict("records")
            self._build_bm25_index(records)
        except Exception:
            logger.warning("Could not load BM25 index", exc_info=True)

    def _needs_update(self) -> bool:
        """Check if any source files have changed."""
        if not self._table:
            return True

        try:
            # Get indexed file paths and their last modified times
            rows = self._table.to_pandas()[["file_path", "last_modified"]].drop_duplicates()
            indexed = {row["file_path"]: row["last_modified"] for _, row in rows.iterrows()}

            # Check current files
            source_files = self._get_source_files()
            for file_path in source_files:
                path_str = str(file_path)
                if path_str not in indexed:
                    return True
                if file_path.stat().st_mtime > indexed[path_str]:
                    return True

            return False
        except Exception:
            return True

    def _dense_search(self, query: str, limit: int) -> list[dict]:
        """Vector similarity search via LanceDB."""
        if self._table is None:
            return []

        try:
            query_embedding = self.embedder.embed_single(query)

            results = self._table.search(query_embedding).limit(limit).to_list()

            return [
                {
                    "id": r.get("id", ""),
                    "file_path": r["file_path"],
                    "content": r["content"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "name": r["name"],
                    "score": 1 - r.get("_distance", 0),
                }
                for r in results
            ]
        except Exception:
            logger.error("Dense search failed", exc_info=True)
            return []

    def _sparse_search(self, query: str, limit: int) -> list[dict]:
        """BM25 keyword search."""
        if self._bm25 is None or not self._bm25_docs:
            return []

        try:
            query_tokens = code_tokenize(query)
            scores = self._bm25.get_scores(query_tokens)

            # Get top results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include if there's any match
                    doc = self._bm25_docs[idx]
                    results.append(
                        {
                            "id": doc.get("id", ""),
                            "file_path": doc["file_path"],
                            "content": doc["content"],
                            "start_line": doc["start_line"],
                            "end_line": doc["end_line"],
                            "name": doc["name"],
                            "score": float(scores[idx]),
                        }
                    )

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
        """
        Hybrid search: dense + sparse with optional reranking.

        Pipeline:
        1. Dense search (semantic via LanceDB)
        2. Sparse search (BM25 keywords)
        3. RRF fusion
        4. Optional cross-encoder reranking
        """
        if self._table is None:
            self.build_index()

        if self._table is None:
            return []

        # Fetch more candidates for reranking
        fetch_limit = limit * 5 if rerank else limit * 2

        if hybrid and self._bm25 is not None:
            # Hybrid: combine dense + sparse
            dense_results = self._dense_search(query, fetch_limit)
            sparse_results = self._sparse_search(query, fetch_limit)

            # RRF fusion
            results = reciprocal_rank_fusion([dense_results, sparse_results])
        else:
            # Dense only
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
