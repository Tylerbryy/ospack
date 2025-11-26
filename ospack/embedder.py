"""GPU-accelerated embeddings using sentence-transformers (lazy loaded)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .log import get_logger

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = get_logger(__name__)

# Flag to prevent accidental model loading in worker processes
_in_worker_process = False


def mark_as_worker() -> None:
    """Mark current process as a worker (called by ProcessPoolExecutor initializer).

    This prevents accidental loading of heavy ML models in worker processes,
    which would cause each worker to load its own 2GB+ model copy.
    """
    global _in_worker_process
    _in_worker_process = True

# MiniLM - fast and lightweight (22M params, 256 token context)
# Good balance of speed vs quality for code search
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-encoder for reranking
# NOTE: This model only sees the first 512 tokens - ensure relevant info is at start
# For longer contexts, consider BAAI/bge-reranker-v2-m3
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_device() -> str:
    """Auto-detect best available device for inference."""
    if env_device := os.environ.get("OSPACK_DEVICE"):
        return env_device.lower()

    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


class Embedder:
    """Embedding model with batching and normalization."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.device = get_device()
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info("Loading embedding model %s on %s...", self.model_name, self.device)
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                # MiniLM has 256 token context - chunks should be small
                self._model.max_seq_length = 256
                logger.info("Embedding model loaded.")
            except Exception as e:
                logger.error("Failed to load embedding model: %s", e)
                raise
        return self._model

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings with batching to prevent OOM.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts to process at once on GPU.
            normalize: Whether to normalize embeddings (required for Cosine Similarity).
        """
        if not texts:
            return []

        # Clean empty strings - can cause issues with some models
        cleaned_texts = [t if t.strip() else " " for t in texts]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,  # CRITICAL for vector DBs using cosine
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class Reranker:
    """Cross-encoder for reranking search results."""

    def __init__(self, model_name: str = RERANK_MODEL):
        self.device = get_device()
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder on first use."""
        if self._model is None:
            logger.info("Loading reranker model %s...", self.model_name)
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info("Reranker loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 10,
        batch_size: int = 32,
    ) -> list[dict]:
        """Rerank results using cross-encoder scores."""
        if not results:
            return []

        # Create query-document pairs
        pairs = [(query, r["content"]) for r in results]

        # Get cross-encoder scores with batching
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        # Add rerank scores and sort
        for r, score in zip(results, scores, strict=False):
            r["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# Global singletons - loaded in main process only, not in worker processes
_embedder: Embedder | None = None
_reranker: Reranker | None = None


def get_embedder() -> Embedder:
    """Get or create the global embedder instance.

    Raises:
        RuntimeError: If called from a worker process (prevents OOM from
            each worker loading its own 2GB+ model copy).
    """
    global _embedder
    if _in_worker_process:
        raise RuntimeError(
            "get_embedder() called from worker process. "
            "Embedding must run in main process only. "
            "Chunk in parallel (CPU), embed sequentially (GPU)."
        )
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_reranker() -> Reranker:
    """Get or create the global reranker instance.

    Raises:
        RuntimeError: If called from a worker process.
    """
    global _reranker
    if _in_worker_process:
        raise RuntimeError(
            "get_reranker() called from worker process. "
            "Reranking must run in main process only."
        )
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
