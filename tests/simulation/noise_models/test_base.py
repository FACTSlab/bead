"""Tests for base noise model."""

from __future__ import annotations

import numpy as np
import pytest

from bead.simulation.noise_models.base import NoiseModel


class ConcreteNoiseModel(NoiseModel):
    """Concrete implementation for testing."""

    def apply(
        self,
        value: str | int | float | list[str],
        context: dict[str, object],
        rng: np.random.RandomState,
    ) -> str | int | float | list[str]:
        """Apply test noise."""
        if isinstance(value, (int, float)):
            return value + 1.0
        return value


def test_noise_model_is_abstract() -> None:
    """Test that NoiseModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        NoiseModel()  # type: ignore[abstract]


def test_concrete_noise_model_instantiation() -> None:
    """Test that concrete noise model can be instantiated."""
    noise_model = ConcreteNoiseModel()
    assert noise_model is not None


def test_concrete_noise_model_apply() -> None:
    """Test that concrete noise model can apply noise."""
    noise_model = ConcreteNoiseModel()
    rng = np.random.RandomState(42)
    result = noise_model.apply(5.0, {}, rng)
    assert result == 6.0


def test_concrete_noise_model_apply_string() -> None:
    """Test that concrete noise model handles strings."""
    noise_model = ConcreteNoiseModel()
    rng = np.random.RandomState(42)
    result = noise_model.apply("test", {}, rng)
    assert result == "test"
