"""Tests for ModelAdapter base class."""

from __future__ import annotations

import numpy as np
import pytest

from bead.items.adapters.base import ModelAdapter
from bead.items.cache import ModelOutputCache


class ConcreteAdapter(ModelAdapter):
    """Concrete implementation for testing abstract base class."""

    def compute_log_probability(self, text: str) -> float:
        """Return dummy log probability."""
        return -5.0

    def compute_perplexity(self, text: str) -> float:
        """Return dummy perplexity."""
        return 100.0

    def get_embedding(self, text: str) -> np.ndarray:
        """Return dummy embedding."""
        return np.array([0.1, 0.2, 0.3])

    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return dummy NLI scores."""
        return {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05}


def test_model_adapter_initialization(in_memory_cache: ModelOutputCache) -> None:
    """Test ModelAdapter initialization."""
    adapter = ConcreteAdapter("test-model", in_memory_cache, model_version="1.0")

    assert adapter.model_name == "test-model"
    assert adapter.model_version == "1.0"
    assert adapter.cache is in_memory_cache


def test_model_adapter_abstract_enforcement(
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that ModelAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ModelAdapter("test", in_memory_cache)  # type: ignore[abstract]


def test_compute_similarity_default(in_memory_cache: ModelOutputCache) -> None:
    """Test default compute_similarity implementation using cosine similarity."""
    adapter = ConcreteAdapter("test-model", in_memory_cache)

    # Embeddings are [0.1, 0.2, 0.3] for both texts (same)
    similarity = adapter.compute_similarity("text1", "text2")

    # Cosine similarity of identical vectors is 1.0
    assert isinstance(similarity, float)
    assert similarity == pytest.approx(1.0, abs=0.01)


def test_compute_similarity_different_texts(
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test compute_similarity with varying embeddings."""

    class VaryingAdapter(ModelAdapter):
        """Adapter that returns different embeddings per text."""

        def __init__(
            self, model_name: str, cache: ModelOutputCache, model_version: str = "1.0"
        ) -> None:
            super().__init__(model_name, cache, model_version)
            self.embedding_map = {
                "cat": np.array([1.0, 0.0, 0.0]),
                "dog": np.array([0.8, 0.6, 0.0]),  # Similar direction
                "sky": np.array([0.0, 0.0, 1.0]),  # Orthogonal
            }

        def compute_log_probability(self, text: str) -> float:
            return 0.0

        def compute_perplexity(self, text: str) -> float:
            return 1.0

        def get_embedding(self, text: str) -> np.ndarray:
            return self.embedding_map.get(text, np.array([0.0, 0.0, 0.0]))

        def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
            return {"entailment": 0.33, "neutral": 0.33, "contradiction": 0.34}

    adapter = VaryingAdapter("varying-model", in_memory_cache)

    # Similar vectors (cat and dog)
    sim_cat_dog = adapter.compute_similarity("cat", "dog")
    assert 0.7 < sim_cat_dog < 0.9

    # Orthogonal vectors (cat and sky)
    sim_cat_sky = adapter.compute_similarity("cat", "sky")
    assert sim_cat_sky == pytest.approx(0.0, abs=0.01)


def test_compute_similarity_zero_norm(in_memory_cache: ModelOutputCache) -> None:
    """Test compute_similarity handles zero-norm vectors."""

    class ZeroNormAdapter(ModelAdapter):
        """Adapter that can return zero vectors."""

        def compute_log_probability(self, text: str) -> float:
            return 0.0

        def compute_perplexity(self, text: str) -> float:
            return 1.0

        def get_embedding(self, text: str) -> np.ndarray:
            if text == "zero":
                return np.array([0.0, 0.0, 0.0])
            return np.array([1.0, 0.0, 0.0])

        def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
            return {"entailment": 0.33, "neutral": 0.33, "contradiction": 0.34}

    adapter = ZeroNormAdapter("zero-norm-model", in_memory_cache)

    # Zero norm vector should return 0.0 similarity
    similarity = adapter.compute_similarity("zero", "zero")
    assert similarity == 0.0

    similarity = adapter.compute_similarity("zero", "normal")
    assert similarity == 0.0


def test_get_nli_label_default(in_memory_cache: ModelOutputCache) -> None:
    """Test default get_nli_label implementation returns max score label."""
    adapter = ConcreteAdapter("test-model", in_memory_cache)

    # Scores: entailment=0.8, neutral=0.15, contradiction=0.05
    label = adapter.get_nli_label("premise", "hypothesis")
    assert label == "entailment"


def test_get_nli_label_different_max(in_memory_cache: ModelOutputCache) -> None:
    """Test get_nli_label with different max scores."""

    class VaryingNLIAdapter(ConcreteAdapter):
        """Adapter with varying NLI scores."""

        def __init__(
            self, model_name: str, cache: ModelOutputCache, model_version: str = "1.0"
        ) -> None:
            super().__init__(model_name, cache, model_version)
            self.current_scores = {
                "entailment": 0.8,
                "neutral": 0.15,
                "contradiction": 0.05,
            }

        def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
            return self.current_scores

    adapter = VaryingNLIAdapter("varying-nli", in_memory_cache)

    # Test entailment max
    label = adapter.get_nli_label("p", "h")
    assert label == "entailment"

    # Test neutral max
    adapter.current_scores = {"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1}
    label = adapter.get_nli_label("p", "h")
    assert label == "neutral"

    # Test contradiction max
    adapter.current_scores = {"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7}
    label = adapter.get_nli_label("p", "h")
    assert label == "contradiction"


def test_abstract_methods_must_be_implemented(
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that all abstract methods must be implemented."""

    class IncompleteAdapter(ModelAdapter):
        """Adapter missing compute_nli implementation."""

        def compute_log_probability(self, text: str) -> float:
            return 0.0

        def compute_perplexity(self, text: str) -> float:
            return 1.0

        def get_embedding(self, text: str) -> np.ndarray:
            return np.array([0.0])

        # Missing: compute_nli

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteAdapter("incomplete", in_memory_cache)  # type: ignore[abstract]
