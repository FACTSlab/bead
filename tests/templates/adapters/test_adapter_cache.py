"""Test model output cache."""

from __future__ import annotations

import json
from pathlib import Path

from bead.templates.adapters.cache import ModelOutputCache


def test_cache_init(tmp_cache_dir: Path) -> None:
    """Test cache initialization."""
    cache = ModelOutputCache(tmp_cache_dir, enabled=True)

    assert cache.cache_dir == tmp_cache_dir
    assert cache.enabled
    assert tmp_cache_dir.exists()


def test_cache_disabled() -> None:
    """Test cache with disabled mode."""
    cache = ModelOutputCache(Path("/nonexistent"), enabled=False)

    assert not cache.enabled
    assert cache.get("model", "text", 0, 10) is None

    # Set should be no-op
    cache.set("model", "text", 0, 10, [("token", -1.0)])


def test_cache_set_and_get(tmp_cache_dir: Path) -> None:
    """Test cache set and get."""
    cache = ModelOutputCache(tmp_cache_dir)

    predictions = [("the", -0.5), ("a", -1.0), ("an", -1.5)]

    # Cache miss
    result = cache.get("bert-base-uncased", "The cat [MASK]", 0, 3)
    assert result is None

    # Store
    cache.set("bert-base-uncased", "The cat [MASK]", 0, 3, predictions)

    # Cache hit
    result = cache.get("bert-base-uncased", "The cat [MASK]", 0, 3)
    assert result == predictions


def test_cache_key_determinism(tmp_cache_dir: Path) -> None:
    """Test cache key is deterministic."""
    cache = ModelOutputCache(tmp_cache_dir)

    predictions = [("sat", -0.3)]

    # Store with same inputs
    cache.set("model1", "text1", 0, 5, predictions)

    # Retrieve should work
    result = cache.get("model1", "text1", 0, 5)
    assert result == predictions

    # Different model name = different key
    result = cache.get("model2", "text1", 0, 5)
    assert result is None

    # Different text = different key
    result = cache.get("model1", "text2", 0, 5)
    assert result is None

    # Different mask position = different key
    result = cache.get("model1", "text1", 1, 5)
    assert result is None

    # Different top_k = different key
    result = cache.get("model1", "text1", 0, 10)
    assert result is None


def test_cache_clear(tmp_cache_dir: Path) -> None:
    """Test cache clear."""
    cache = ModelOutputCache(tmp_cache_dir)

    # Store multiple items
    cache.set("model1", "text1", 0, 5, [("a", -0.5)])
    cache.set("model1", "text2", 0, 5, [("b", -0.5)])

    # Verify they exist
    assert cache.get("model1", "text1", 0, 5) is not None
    assert cache.get("model1", "text2", 0, 5) is not None

    # Clear
    cache.clear()

    # Verify they're gone
    assert cache.get("model1", "text1", 0, 5) is None
    assert cache.get("model1", "text2", 0, 5) is None


def test_cache_corrupted_file(tmp_cache_dir: Path) -> None:
    """Test handling of corrupted cache files."""
    cache = ModelOutputCache(tmp_cache_dir)

    # Create corrupted cache file
    cache_key = cache._compute_key("model", "text", 0, 5)
    cache_file = tmp_cache_dir / f"{cache_key}.json"
    cache_file.write_text("invalid json {{{")

    # Should return None on corruption
    result = cache.get("model", "text", 0, 5)
    assert result is None


def test_cache_file_format(tmp_cache_dir: Path) -> None:
    """Test cache file format is correct."""
    cache = ModelOutputCache(tmp_cache_dir)

    predictions = [("token1", -0.5), ("token2", -1.0)]
    cache.set("model", "text", 0, 2, predictions)

    # Read file directly
    cache_key = cache._compute_key("model", "text", 0, 2)
    cache_file = tmp_cache_dir / f"{cache_key}.json"

    with open(cache_file) as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0] == {"token": "token1", "log_prob": -0.5}
    assert data[1] == {"token": "token2", "log_prob": -1.0}


def test_cache_with_different_top_k(tmp_cache_dir: Path) -> None:
    """Test caching with different top_k values."""
    cache = ModelOutputCache(tmp_cache_dir)

    predictions_5 = [("a", -0.1), ("b", -0.2), ("c", -0.3), ("d", -0.4), ("e", -0.5)]
    predictions_10 = predictions_5 + [("f", -0.6)]

    # Cache for top_k=5
    cache.set("model", "text", 0, 5, predictions_5)

    # Cache for top_k=10
    cache.set("model", "text", 0, 10, predictions_10)

    # Each should be stored separately
    result_5 = cache.get("model", "text", 0, 5)
    result_10 = cache.get("model", "text", 0, 10)

    assert result_5 == predictions_5
    assert result_10 == predictions_10


def test_cache_empty_predictions(tmp_cache_dir: Path) -> None:
    """Test caching empty predictions."""
    cache = ModelOutputCache(tmp_cache_dir)

    empty_preds: list[tuple[str, float]] = []
    cache.set("model", "text", 0, 5, empty_preds)

    result = cache.get("model", "text", 0, 5)
    assert result == []


def test_cache_disabled_clear() -> None:
    """Test clear on disabled cache."""
    cache = ModelOutputCache(Path("/nonexistent"), enabled=False)

    # Should not raise
    cache.clear()
