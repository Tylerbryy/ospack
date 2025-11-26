"""BM25+ search index with incremental updates (using bm25s).

Features:
- BM25+ variant for better handling of varied document lengths
- Numba backend for ~2x faster retrieval
- Memory-mapped loading for reduced RAM usage
- PyStemmer for better recall (finds "running" when searching "run")
- Code-aware tokenization (camelCase, snake_case splitting)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import bm25s
import numpy as np
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

from .chunker import Chunker
from .log import get_logger

logger = get_logger(__name__)

# Optional: PyStemmer for better recall
try:
    import Stemmer
    _stemmer = Stemmer.Stemmer("english")
    logger.debug("PyStemmer loaded - stemming enabled")
except ImportError:
    _stemmer = None
    logger.debug("PyStemmer not installed - stemming disabled")

# Worker process state
_worker_chunker: Chunker | None = None


def _init_worker() -> None:
    """Initialize worker process with a reusable Chunker."""
    global _worker_chunker
    _worker_chunker = Chunker()


# Exclude patterns
EXCLUDE_PATTERNS = [
    ".git/**", ".svn/**", ".hg/**",
    "node_modules/**", ".venv/**", "venv/**", "vendor/**",
    "dist/**", "build/**", "target/**", "__pycache__/**",
    "*.pyc", "*.pyo",
    ".idea/**", ".vscode/**", "*.swp", "*.swo",
    ".next/**", ".turbo/**", "coverage/**",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Pipfile.lock",
    "*.min.js", "*.min.css",
]

MAX_FILE_SIZE = int(os.environ.get("OSPACK_MAX_FILE_SIZE", 1024 * 1024))
MAX_WORKERS = min(os.cpu_count() or 4, 8)
SCHEMA_VERSION = 6  # v6: BM25+ with stemming, numba backend, mmap loading


def get_repo_hash(path: str) -> str:
    """Generate a hash for the repository path."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def is_text_file(path: Path, sample_size: int = 8192) -> bool:
    """Check if a file is text (not binary)."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" not in chunk
    except Exception:
        return False


def _chunk_file_worker(args: tuple[str, float]) -> list[dict]:
    """Worker function for parallel chunking."""
    global _worker_chunker
    file_path, mtime = args
    try:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")

        if _worker_chunker is None:
            _worker_chunker = Chunker()
        chunks = _worker_chunker.chunk(str(file_path), content)

        return [{
            "id": hashlib.md5(f"{file_path}:{c.start_line}-{c.end_line}".encode()).hexdigest(),
            "file_path": str(file_path),
            "content": c.content,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "name": c.name or "",
            "node_type": c.node_type,
            "last_modified": mtime,
        } for c in chunks]
    except Exception:
        return []


def code_tokenize(text: str, stem: bool = True) -> list[str]:
    """Code-aware tokenizer for BM25 with optional stemming.

    Args:
        text: The text to tokenize
        stem: Whether to apply stemming (requires PyStemmer)

    Features:
        - Splits camelCase: getUserName -> [get, user, name]
        - Splits snake_case: get_user_name -> [get, user, name]
        - Splits dot notation: os.path.join -> [os, path, join]
        - Stems words: running -> run, authentication -> authent
    """
    text = text.lower()
    raw_tokens = re.findall(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", text)

    tokens = []
    seen = set()

    for t in raw_tokens:
        if t not in seen:
            tokens.append(t)
            seen.add(t)

        if "_" in t:
            for part in t.split("_"):
                if part and part not in seen:
                    tokens.append(part)
                    seen.add(part)

        if "." in t:
            for part in t.split("."):
                if part and part not in seen:
                    tokens.append(part)
                    seen.add(part)

        camel_parts = re.findall(r"[a-z]+|[0-9]+", t)
        if len(camel_parts) > 1:
            for part in camel_parts:
                if part and part not in seen:
                    tokens.append(part)
                    seen.add(part)

    # Apply stemming if available and requested
    if stem and _stemmer is not None:
        stemmed = _stemmer.stemWords(tokens)
        # Dedupe after stemming (different words may stem to same root)
        seen_stemmed = set()
        unique_stemmed = []
        for s in stemmed:
            if s not in seen_stemmed:
                unique_stemmed.append(s)
                seen_stemmed.add(s)
        return unique_stemmed

    return tokens


class Indexer:
    """BM25+ search index with incremental updates (using bm25s)."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.storage_dir = Path.home() / ".ospack" / "index" / get_repo_hash(str(self.root_dir))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.version_path = self.storage_dir / "schema_version"
        self.bm25_dir = self.storage_dir / "bm25s"  # bm25s uses a directory
        self.meta_path = self.storage_dir / "meta.json"

        self._bm25: bm25s.BM25 | None = None
        self._doc_ids: list[str] = []
        self._doc_map: dict[str, dict] = {}
        self._file_mtimes: dict[str, float] = {}
        self._loaded = False

    def _get_source_files(self) -> dict[str, float]:
        """Scan directory returning {filepath: mtime}."""
        exclude_spec = PathSpec.from_lines(GitWildMatchPattern, EXCLUDE_PATTERNS)

        results = {}
        for path in self.root_dir.rglob("*"):
            if not path.is_file():
                continue

            rel = str(path.relative_to(self.root_dir))
            if exclude_spec.match_file(rel):
                continue

            try:
                stat = path.stat()
                if stat.st_size > MAX_FILE_SIZE or stat.st_size == 0:
                    continue
            except OSError:
                continue

            if not is_text_file(path):
                continue

            results[str(path)] = stat.st_mtime
        return results

    def _load(self):
        """Load BM25+ index from disk with memory-mapping for efficiency."""
        if self._loaded:
            return

        if self.bm25_dir.exists() and self.meta_path.exists():
            try:
                # Load bm25s model with memory-mapping (reduces RAM usage significantly)
                self._bm25 = bm25s.BM25.load(str(self.bm25_dir), load_corpus=False, mmap=True)

                # Load metadata
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                    self._doc_ids = meta["ids"]
                    self._doc_map = meta["map"]
                    self._file_mtimes = meta.get("mtimes", {})
                logger.debug("Loaded BM25+ index (%d docs) with mmap.", len(self._doc_ids))
            except Exception as e:
                logger.warning("Corrupt BM25+ index, will rebuild: %s", e)

        self._loaded = True

    def _save(self):
        """Persist BM25+ state to disk."""
        if self._bm25 is not None:
            self._bm25.save(str(self.bm25_dir))

        with open(self.meta_path, "w") as f:
            json.dump({
                "ids": self._doc_ids,
                "map": self._doc_map,
                "mtimes": self._file_mtimes,
            }, f)

    def _check_schema_version(self) -> bool:
        """Check if schema version matches. Returns True if rebuild needed."""
        if not self.version_path.exists():
            return True
        try:
            stored_version = int(self.version_path.read_text().strip())
            return stored_version != SCHEMA_VERSION
        except (ValueError, OSError):
            return True

    def _save_schema_version(self) -> None:
        self.version_path.write_text(str(SCHEMA_VERSION))

    def build_index(self, force: bool = False) -> int:
        """Build or incrementally update the index."""
        self._load()

        if self._check_schema_version():
            logger.info("Schema version changed, forcing rebuild...")
            force = True

        if force:
            import shutil
            if self.bm25_dir.exists():
                shutil.rmtree(self.bm25_dir)
            if self.meta_path.exists():
                self.meta_path.unlink()
            self._bm25 = None
            self._doc_ids = []
            self._doc_map = {}
            self._file_mtimes = {}

        current_files = self._get_source_files()

        # Calculate delta
        to_add: list[tuple[str, float]] = []
        to_remove: set[str] = set()

        for path, mtime in current_files.items():
            if path not in self._file_mtimes:
                to_add.append((path, mtime))
            elif mtime > self._file_mtimes[path]:
                to_remove.add(path)
                to_add.append((path, mtime))

        for path in self._file_mtimes:
            if path not in current_files:
                to_remove.add(path)

        if not to_add and not to_remove:
            logger.debug("Index is up to date.")
            return 0

        logger.info("Indexing: +%d files, -%d files", len(to_add), len(to_remove))

        # Remove old chunks
        if to_remove:
            self._doc_ids = [i for i in self._doc_ids if self._doc_map.get(i, {}).get("file_path") not in to_remove]
            self._doc_map = {k: v for k, v in self._doc_map.items() if v.get("file_path") not in to_remove}
            for path in to_remove:
                self._file_mtimes.pop(path, None)

        # Chunk new files
        new_chunks: list[dict] = []
        if to_add:
            logger.info("Chunking %d files...", len(to_add))
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker) as executor:
                futures = [executor.submit(_chunk_file_worker, arg) for arg in to_add]
                for future in as_completed(futures):
                    new_chunks.extend(future.result())

            logger.info("Chunked -> %d chunks", len(new_chunks))

            # Add to maps
            for chunk in new_chunks:
                self._doc_ids.append(chunk["id"])
                self._doc_map[chunk["id"]] = chunk

            # Update mtimes
            for path, mtime in to_add:
                self._file_mtimes[path] = mtime

        # Rebuild BM25+ model
        if self._doc_ids:
            logger.info("Building BM25+ index with numba backend...")
            corpus_tokens = []
            for doc_id in self._doc_ids:
                doc = self._doc_map[doc_id]
                text = f"{doc['name']} {doc['content']} {doc['file_path']}"
                corpus_tokens.append(code_tokenize(text))

            # Use BM25+ for better handling of varied document lengths
            # Use numba backend for ~2x faster retrieval on large datasets
            self._bm25 = bm25s.BM25(method="bm25+", backend="numba")
            self._bm25.index(corpus_tokens)

        self._save()
        self._save_schema_version()

        return len(new_chunks)

    def search(
        self,
        query: str,
        limit: int = 10,
        rerank: bool = False,  # Ignored (no reranker)
        hybrid: bool = False,  # Ignored (BM25+ only)
    ) -> list[dict]:
        """BM25+ keyword search."""
        self._load()

        if not self._bm25 or not self._doc_ids:
            self.build_index()

        if not self._bm25:
            return []

        query_tokens = [code_tokenize(query)]  # bm25s expects list of token lists
        indices, scores = self._bm25.retrieve(query_tokens, k=limit)

        # indices and scores are 2D arrays (queries x results)
        indices = indices[0]  # First (only) query
        scores = scores[0]

        results = []
        for idx, score in zip(indices, scores):
            if score > 0 and idx < len(self._doc_ids):
                doc_id = self._doc_ids[idx]
                doc = self._doc_map[doc_id]
                results.append({
                    "id": doc_id,
                    "file_path": doc["file_path"],
                    "content": doc["content"],
                    "start_line": doc["start_line"],
                    "end_line": doc["end_line"],
                    "name": doc["name"],
                    "node_type": doc.get("node_type", ""),
                    "score": float(score),
                })

        return results


# Global singleton
_indexers: dict[str, Indexer] = {}


def get_indexer(root_dir: str) -> Indexer:
    """Get or create an indexer for the given root directory."""
    root_dir = str(Path(root_dir).resolve())
    if root_dir not in _indexers:
        _indexers[root_dir] = Indexer(root_dir)
    return _indexers[root_dir]
