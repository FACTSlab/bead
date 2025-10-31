"""Tests for sentence transformer adapter."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_mock import MockerFixture

from bead.items.adapters.sentence_transformers import HuggingFaceSentenceTransformer
from bead.items.cache import ModelOutputCache


def test_sentence_transformer_initialization(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test sentence transformer initialization."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer(
        "all-MiniLM-L6-v2", in_memory_cache, device="cpu"
    )

    assert adapter.model_name == "all-MiniLM-L6-v2"
    assert adapter.device == "cpu"
    assert adapter.normalize_embeddings is True


def test_sentence_transformer_get_embedding(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test embedding generation."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    embedding = adapter.get_embedding(sample_texts[0])

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)  # all-MiniLM-L6-v2 dimension
    # Should be normalized (unit length)
    norm = np.linalg.norm(embedding)
    assert norm == pytest.approx(1.0, abs=0.01)


def test_sentence_transformer_compute_similarity(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test similarity computation between texts."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    # Compute similarity between two texts
    similarity = adapter.compute_similarity(sample_texts[0], sample_texts[1])

    assert isinstance(similarity, float)
    # Cosine similarity should be in [-1, 1]
    assert -1.0 <= similarity <= 1.0


def test_sentence_transformer_compute_similarity_identical(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test similarity of identical texts is close to 1.0."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    # Same text should have similarity ~1.0
    similarity = adapter.compute_similarity(sample_texts[0], sample_texts[0])

    assert similarity == pytest.approx(1.0, abs=0.01)


def test_sentence_transformer_log_prob_not_supported(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that log probability is not supported."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    with pytest.raises(NotImplementedError, match="Log probability is not supported"):
        adapter.compute_log_probability("text")


def test_sentence_transformer_perplexity_not_supported(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that perplexity is not supported."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    with pytest.raises(NotImplementedError, match="Perplexity is not supported"):
        adapter.compute_perplexity("text")


def test_sentence_transformer_nli_not_supported(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that NLI is not supported."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    with pytest.raises(NotImplementedError, match="NLI is not supported"):
        adapter.compute_nli("premise", "hypothesis")


def test_sentence_transformer_caching_embedding(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that embeddings are cached properly."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    text = "The cat sat on the mat."

    # First call - should compute
    emb1 = adapter.get_embedding(text)

    # Second call - should hit cache
    emb2 = adapter.get_embedding(text)

    # Should be identical (from cache)
    np.testing.assert_array_equal(emb1, emb2)


def test_sentence_transformer_caching_similarity(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that similarity scores are cached properly."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    text1 = "The cat sat on the mat."
    text2 = "The dog stood on the rug."

    # First call - should compute
    sim1 = adapter.compute_similarity(text1, text2)

    # Second call - should hit cache
    sim2 = adapter.compute_similarity(text1, text2)

    assert sim1 == sim2


def test_sentence_transformer_normalize_disabled(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test with normalization disabled."""
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer(
        "all-MiniLM-L6-v2", in_memory_cache, normalize_embeddings=False
    )

    embedding = adapter.get_embedding(sample_texts[0])

    assert isinstance(embedding, np.ndarray)
    # Without normalization, norm may not be 1.0
    # (but mock returns normalized anyway in our case)


def test_sentence_transformer_lazy_loading(
    mocker: MockerFixture,
    mock_sentence_transformer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that model is loaded lazily on first use."""
    mock_st_class = mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    adapter = HuggingFaceSentenceTransformer("all-MiniLM-L6-v2", in_memory_cache)

    # Model should not be loaded yet
    assert adapter._model is None

    # First use should trigger loading
    adapter.get_embedding("test")

    # Model should now be loaded
    assert adapter._model is not None
    mock_st_class.assert_called_once_with("all-MiniLM-L6-v2", device=None)
