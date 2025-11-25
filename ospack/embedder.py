"""GPU-accelerated embeddings using sentence-transformers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from .log import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Jina V2 handles 8192 tokens - critical for code chunks that exceed 512 tokens
# NOTE: Requires trust_remote_code=True (uses custom ALiBi architecture)
DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-code"

# Cross-encoder for reranking
# NOTE: This model only sees the first 512 tokens - ensure relevant info is at start
# For longer contexts, consider BAAI/bge-reranker-v2-m3
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_device() -> str:
    """Auto-detect best available device for inference."""
    if env_device := os.environ.get("OSPACK_DEVICE"):
        return env_device.lower()

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
                    trust_remote_code=True,  # REQUIRED for Jina models (custom ALiBi)
                )
                # Explicitly set max seq length for Jina's 8192 context
                self._model.max_seq_length = 8192
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

    WARNING: Do not call this in ProcessPoolExecutor workers.
    Each worker would load its own 2GB model copy, causing OOM.
    Chunk in parallel (CPU), embed sequentially in main process (GPU).
    """
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_reranker() -> Reranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
