"""Tests for simulation standard library functions."""

from __future__ import annotations

import math

import pytest

from bead.dsl import SIMULATION_FUNCTIONS, stdlib


def test_sigmoid_zero() -> None:
    """Test sigmoid at x=0 returns 0.5."""
    assert stdlib.sigmoid(0.0) == 0.5


def test_sigmoid_positive() -> None:
    """Test sigmoid with positive value."""
    result = stdlib.sigmoid(5.0)
    assert 0.99 < result < 1.0


def test_sigmoid_negative() -> None:
    """Test sigmoid with negative value."""
    result = stdlib.sigmoid(-5.0)
    assert 0.0 < result < 0.01


def test_sigmoid_bounds() -> None:
    """Test sigmoid always returns value between 0 and 1."""
    for x in [-100, -10, -1, 0, 1, 10, 100]:
        result = stdlib.sigmoid(float(x))
        assert 0.0 <= result <= 1.0


def test_softmax_basic() -> None:
    """Test softmax basic functionality."""
    probs = stdlib.softmax([1.0, 2.0, 3.0])
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6
    assert probs[0] < probs[1] < probs[2]


def test_softmax_empty() -> None:
    """Test softmax with empty list."""
    probs = stdlib.softmax([])
    assert probs == []


def test_softmax_single() -> None:
    """Test softmax with single value."""
    probs = stdlib.softmax([5.0])
    assert len(probs) == 1
    assert abs(probs[0] - 1.0) < 1e-6


def test_softmax_uniform() -> None:
    """Test softmax with uniform values."""
    probs = stdlib.softmax([1.0, 1.0, 1.0])
    for p in probs:
        assert abs(p - 1.0 / 3.0) < 1e-6


def test_sample_categorical_deterministic() -> None:
    """Test categorical sampling with seed for determinism."""
    result1 = stdlib.sample_categorical([0.2, 0.5, 0.3], seed=42)
    result2 = stdlib.sample_categorical([0.2, 0.5, 0.3], seed=42)
    assert result1 == result2


def test_sample_categorical_range() -> None:
    """Test categorical sampling returns valid index."""
    for _ in range(10):
        result = stdlib.sample_categorical([0.2, 0.5, 0.3])
        assert result in [0, 1, 2]


def test_sample_categorical_certain() -> None:
    """Test categorical sampling with probability 1."""
    result = stdlib.sample_categorical([0.0, 1.0, 0.0])
    assert result == 1


def test_add_noise_gaussian() -> None:
    """Test adding gaussian noise."""
    result = stdlib.add_noise(5.0, "gaussian", 0.1, seed=42)
    assert isinstance(result, float)
    assert 4.5 < result < 5.5  # Should be close to original


def test_add_noise_uniform() -> None:
    """Test adding uniform noise."""
    result = stdlib.add_noise(5.0, "uniform", 0.1, seed=42)
    assert isinstance(result, float)
    assert 4.9 < result < 5.1  # Within range


def test_add_noise_unknown_type() -> None:
    """Test adding noise with unknown type returns value unchanged."""
    result = stdlib.add_noise(5.0, "unknown", 0.1)
    assert result == 5.0


def test_add_noise_deterministic() -> None:
    """Test noise is deterministic with seed."""
    result1 = stdlib.add_noise(5.0, "gaussian", 0.1, seed=123)
    result2 = stdlib.add_noise(5.0, "gaussian", 0.1, seed=123)
    assert result1 == result2


def test_distance_cosine() -> None:
    """Test cosine distance."""
    dist = stdlib.distance([1.0, 0.0], [0.0, 1.0], "cosine")
    assert abs(dist - 1.0) < 1e-6


def test_distance_euclidean() -> None:
    """Test euclidean distance."""
    dist = stdlib.distance([1.0, 0.0], [0.0, 1.0], "euclidean")
    assert abs(dist - math.sqrt(2.0)) < 1e-6


def test_distance_manhattan() -> None:
    """Test manhattan distance."""
    dist = stdlib.distance([1.0, 0.0], [0.0, 1.0], "manhattan")
    assert abs(dist - 2.0) < 1e-6


def test_distance_identical() -> None:
    """Test distance between identical vectors."""
    dist = stdlib.distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "cosine")
    assert abs(dist) < 1e-6


def test_distance_zero_vectors() -> None:
    """Test distance with zero vectors."""
    dist = stdlib.distance([0.0, 0.0], [0.0, 0.0], "cosine")
    assert dist == 1.0  # Defined as 1.0 for zero vectors


def test_distance_unknown_metric() -> None:
    """Test distance with unknown metric raises error."""
    with pytest.raises(ValueError, match="Unknown metric"):
        stdlib.distance([1.0, 0.0], [0.0, 1.0], "unknown")


def test_preference_prob_equal() -> None:
    """Test preference probability with equal scores."""
    prob = stdlib.preference_prob(5.0, 5.0)
    assert abs(prob - 0.5) < 1e-6


def test_preference_prob_higher_score() -> None:
    """Test preference probability favors higher score."""
    prob = stdlib.preference_prob(10.0, 5.0)
    assert prob > 0.9


def test_preference_prob_lower_score() -> None:
    """Test preference probability penalizes lower score."""
    prob = stdlib.preference_prob(5.0, 10.0)
    assert prob < 0.1


def test_preference_prob_temperature() -> None:
    """Test temperature scaling in preference probability."""
    prob_low_temp = stdlib.preference_prob(10.0, 5.0, temperature=0.5)
    prob_high_temp = stdlib.preference_prob(10.0, 5.0, temperature=5.0)
    assert prob_low_temp > prob_high_temp  # Lower temp = more confident


def test_simulation_functions_registry() -> None:
    """Test SIMULATION_FUNCTIONS contains all expected functions."""
    expected = [
        "sigmoid",
        "softmax",
        "sample_categorical",
        "add_noise",
        "model_output",
        "distance",
        "preference_prob",
    ]
    for func_name in expected:
        assert func_name in SIMULATION_FUNCTIONS


def test_model_output_no_attr() -> None:
    """Test model_output with object without model_outputs."""

    class DummyItem:
        pass

    item = DummyItem()
    result = stdlib.model_output(item, "lm_score", default=0.0)
    assert result == 0.0
