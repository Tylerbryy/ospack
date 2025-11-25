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

# Code-specific model with 8192 token context (vs 512 for MiniLM)
# This is critical - most code chunks exceed 512 tokens
DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-code"

# Cross-encoder for reranking (loaded lazily)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_device() -> str:
    """Auto-detect best available device for inference."""
    # Allow override via environment variable
    if env_device := os.environ.get("OSPACK_DEVICE"):
        return env_device.lower()

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


class Embedder:
    """Embedding model with GPU auto-detection."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.device = get_device()
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info("Loading embedding model on %s...", self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded.")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


# Global singleton for efficiency
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


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
            logger.info("Loading reranker model...")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker loaded.")
        return self._model

    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """Rerank results using cross-encoder scores."""
        if not results:
            return []

        # Create query-document pairs
        pairs = [(query, r["content"]) for r in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Add rerank scores and sort
        for r, score in zip(results, scores, strict=False):
            r["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# Global singleton for reranker
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
