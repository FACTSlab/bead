"""Tests for temperature noise model."""

from __future__ import annotations

import numpy as np
import pytest

from bead.simulation.noise_models.temperature import TemperatureNoiseModel


def test_temperature_model_instantiation() -> None:
    """Test that temperature model can be instantiated."""
    model = TemperatureNoiseModel(temperature=1.5)
    assert model.temperature == 1.5


def test_temperature_model_default() -> None:
    """Test default temperature is 1.0."""
    model = TemperatureNoiseModel()
    assert model.temperature == 1.0


def test_temperature_model_negative_raises() -> None:
    """Test that negative temperature raises ValueError."""
    with pytest.raises(ValueError, match="Temperature must be positive"):
        TemperatureNoiseModel(temperature=-1.0)


def test_temperature_model_zero_raises() -> None:
    """Test that zero temperature raises ValueError."""
    with pytest.raises(ValueError, match="Temperature must be positive"):
        TemperatureNoiseModel(temperature=0.0)


def test_apply_forced_choice_no_modification() -> None:
    """Test that forced choice values pass through unchanged.

    Temperature is applied earlier in the strategy, not in noise model.
    """
    model = TemperatureNoiseModel(temperature=2.0)
    rng = np.random.RandomState(42)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "forced_choice"

    context = {"strategy": MockStrategy()}
    result = model.apply("option_a", context, rng)
    assert result == "option_a"


def test_apply_ordinal_scale_adds_noise() -> None:
    """Test that ordinal scale values get gaussian noise."""
    model = TemperatureNoiseModel(temperature=2.0)
    rng = np.random.RandomState(42)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "ordinal_scale"

    context = {"strategy": MockStrategy()}

    # Apply noise to integer value
    result = model.apply(5, context, rng)
    assert isinstance(result, (int, float))
    # Should be different from original due to noise
    # With temp=2.0, stddev=1.0, so fairly wide distribution
    assert result != 5


def test_apply_ordinal_scale_noise_distribution() -> None:
    """Test that ordinal scale noise follows expected distribution."""
    model = TemperatureNoiseModel(temperature=1.0)
    rng = np.random.RandomState(42)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "ordinal_scale"

    context = {"strategy": MockStrategy()}

    # Generate many samples
    values = []
    for _ in range(1000):
        result = model.apply(5.0, context, rng)
        values.append(result)

    values_array = np.array(values)

    # Check mean is close to original (5.0)
    assert np.abs(np.mean(values_array) - 5.0) < 0.1

    # Check stddev is close to temperature * 0.5 = 0.5
    assert np.abs(np.std(values_array) - 0.5) < 0.1


def test_apply_ordinal_scale_temperature_scaling() -> None:
    """Test that higher temperature produces more noise."""
    rng_low = np.random.RandomState(42)
    rng_high = np.random.RandomState(42)

    model_low = TemperatureNoiseModel(temperature=0.5)
    model_high = TemperatureNoiseModel(temperature=2.0)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "ordinal_scale"

    context = {"strategy": MockStrategy()}

    # Generate samples
    values_low = []
    values_high = []
    for _ in range(1000):
        values_low.append(model_low.apply(5.0, context, rng_low))
        values_high.append(model_high.apply(5.0, context, rng_high))

    # Higher temperature should have higher variance
    assert np.std(values_high) > np.std(values_low)


def test_apply_no_strategy_no_modification() -> None:
    """Test that values pass through when no strategy in context."""
    model = TemperatureNoiseModel(temperature=2.0)
    rng = np.random.RandomState(42)

    context = {}  # type: dict[str, object]
    result = model.apply("some_value", context, rng)
    assert result == "some_value"


def test_apply_unknown_task_type_no_modification() -> None:
    """Test that unknown task types pass through unchanged."""
    model = TemperatureNoiseModel(temperature=2.0)
    rng = np.random.RandomState(42)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "unknown_task"

    context = {"strategy": MockStrategy()}
    result = model.apply("some_value", context, rng)
    assert result == "some_value"


def test_apply_ordinal_scale_non_numeric_no_modification() -> None:
    """Test that non-numeric values pass through for ordinal scale."""
    model = TemperatureNoiseModel(temperature=2.0)
    rng = np.random.RandomState(42)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "ordinal_scale"

    context = {"strategy": MockStrategy()}
    result = model.apply("not_a_number", context, rng)
    assert result == "not_a_number"


def test_apply_preserves_value_type() -> None:
    """Test that apply preserves the type of value when not modifying."""
    model = TemperatureNoiseModel(temperature=1.0)
    rng = np.random.RandomState(42)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "forced_choice"

    context = {"strategy": MockStrategy()}

    # String
    assert isinstance(model.apply("test", context, rng), str)

    # Int (for forced choice, no modification)
    assert isinstance(model.apply(5, context, rng), int)

    # List
    test_list = ["a", "b"]
    result = model.apply(test_list, context, rng)
    assert result == test_list


def test_apply_with_float_value_ordinal() -> None:
    """Test that float values work with ordinal scale."""
    model = TemperatureNoiseModel(temperature=1.0)
    rng = np.random.RandomState(42)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "ordinal_scale"

    context = {"strategy": MockStrategy()}
    result = model.apply(3.5, context, rng)
    assert isinstance(result, float)
    assert result != 3.5  # Should have noise added


def test_apply_different_rng_states() -> None:
    """Test that different RNG states produce different results."""
    model = TemperatureNoiseModel(temperature=1.0)
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(43)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "ordinal_scale"

    context = {"strategy": MockStrategy()}

    result1 = model.apply(5.0, context, rng1)
    result2 = model.apply(5.0, context, rng2)

    # Should be different due to different random states
    assert result1 != result2


def test_apply_same_rng_state_reproducible() -> None:
    """Test that same RNG state produces reproducible results."""
    model = TemperatureNoiseModel(temperature=1.0)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "ordinal_scale"

    context = {"strategy": MockStrategy()}

    # Same seed
    rng1 = np.random.RandomState(42)
    result1 = model.apply(5.0, context, rng1)

    rng2 = np.random.RandomState(42)
    result2 = model.apply(5.0, context, rng2)

    assert result1 == result2


def test_apply_extreme_temperatures() -> None:
    """Test behavior with extreme temperature values."""
    # Very low temperature (almost deterministic)
    model_low = TemperatureNoiseModel(temperature=0.01)
    rng = np.random.RandomState(42)

    class MockStrategy:
        @property
        def supported_task_type(self) -> str:
            return "ordinal_scale"

    context = {"strategy": MockStrategy()}

    values = []
    for _ in range(100):
        values.append(model_low.apply(5.0, context, rng))

    # Should have very low variance
    assert np.std(values) < 0.1

    # Very high temperature
    model_high = TemperatureNoiseModel(temperature=10.0)
    values_high = []
    for _ in range(100):
        values_high.append(model_high.apply(5.0, context, rng))

    # Should have high variance
    assert np.std(values_high) > 1.0
