"""Sentence transformer adapter for semantic embeddings.

This module provides an adapter for sentence-transformers models,
which are optimized for generating sentence embeddings for semantic
similarity tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from bead.items.adapters.base import ModelAdapter
from bead.items.cache import ModelOutputCache

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class HuggingFaceSentenceTransformer(ModelAdapter):
    """Adapter for sentence-transformers models.

    Supports sentence-transformers models like "all-MiniLM-L6-v2",
    "all-mpnet-base-v2", etc. These models are optimized for generating
    sentence embeddings for semantic similarity tasks.

    Parameters
    ----------
    model_name : str
        Sentence transformer model identifier.
    cache : ModelOutputCache
        Cache instance for storing model outputs.
    device : str | None
        Device to run model on. If None, uses sentence-transformers default.
    model_version : str
        Version string for cache tracking.
    normalize_embeddings : bool
        Whether to normalize embeddings to unit length.

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.items.cache import ModelOutputCache
    >>> cache = ModelOutputCache(cache_dir=Path(".cache"))
    >>> model = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", cache)
    >>> embedding = model.get_embedding("The cat sat on the mat.")
    >>> similarity = model.compute_similarity("The cat sat.", "The dog stood.")
    """

    def __init__(
        self,
        model_name: str,
        cache: ModelOutputCache,
        device: str | None = None,
        model_version: str = "unknown",
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__(model_name, cache, model_version)
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> None:
        """Load model lazily on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info(f"Loading sentence transformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def model(self) -> SentenceTransformer:
        """Get the model, loading if necessary."""
        self._load_model()
        assert self._model is not None
        return self._model

    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text.

        Not supported for sentence transformer models.

        Raises
        ------
        NotImplementedError
            Always raised, as sentence transformers don't provide log probabilities.
        """
        raise NotImplementedError(
            f"Log probability is not supported for sentence transformer "
            f"{self.model_name}. Use HuggingFaceLanguageModel or "
            "HuggingFaceMaskedLanguageModel instead."
        )

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text.

        Not supported for sentence transformer models.

        Raises
        ------
        NotImplementedError
            Always raised, as sentence transformers don't provide perplexity.
        """
        raise NotImplementedError(
            f"Perplexity is not supported for sentence transformer {self.model_name}. "
            "Use HuggingFaceLanguageModel or HuggingFaceMaskedLanguageModel instead."
        )

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Uses sentence-transformers encode() method to generate
        optimized sentence embeddings.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector for the text.
        """
        # Check cache
        cached = self.cache.get(self.model_name, "embedding", text=text)
        if cached is not None:
            return cached

        # Encode text
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Ensure it's a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Cache result
        self.cache.set(
            self.model_name,
            "embedding",
            embedding,
            model_version=self.model_version,
            text=text,
        )

        return embedding

    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores.

        Not supported for sentence transformer models.

        Raises
        ------
        NotImplementedError
            Always raised, as sentence transformers don't support NLI directly.
        """
        raise NotImplementedError(
            f"NLI is not supported for sentence transformer {self.model_name}. "
            "Use HuggingFaceNLI adapter with an NLI-trained model instead."
        )

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.

        Uses cosine similarity of embeddings. For sentence transformers,
        this is optimized as embeddings are already normalized (if
        normalize_embeddings=True).

        Parameters
        ----------
        text1 : str
            First text.
        text2 : str
            Second text.

        Returns
        -------
        float
            Similarity score in [-1, 1] (cosine similarity).
        """
        # Check cache
        cached = self.cache.get(self.model_name, "similarity", text1=text1, text2=text2)
        if cached is not None:
            return cached

        # Get embeddings
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Compute cosine similarity
        if self.normalize_embeddings:
            # Embeddings are already normalized, just dot product
            similarity = float(np.dot(emb1, emb2))
        else:
            # Need to normalize
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = float(dot_product / (norm1 * norm2))

        # Cache result
        self.cache.set(
            self.model_name,
            "similarity",
            similarity,
            model_version=self.model_version,
            text1=text1,
            text2=text2,
        )

        return similarity
