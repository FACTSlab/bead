"""Tests for base simulation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.models import Item, ItemTemplate


class ConcreteStrategy(SimulationStrategy):
    """Concrete implementation for testing."""

    @property
    def supported_task_type(self) -> str:
        """Return test task type."""
        return "test_task"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item."""
        if not hasattr(item_template, "task_type"):
            msg = "Missing task_type"
            raise ValueError(msg)

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> str:
        """Generate test response."""
        return "test_response"


def test_strategy_is_abstract() -> None:
    """Test that SimulationStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SimulationStrategy()  # type: ignore[abstract]


def test_concrete_strategy_instantiation() -> None:
    """Test that concrete strategy can be instantiated."""
    strategy = ConcreteStrategy()
    assert strategy.supported_task_type == "test_task"


def test_extract_model_outputs_no_outputs() -> None:
    """Test extract_model_outputs with item without model_outputs."""

    class DummyItem:
        pass

    strategy = ConcreteStrategy()
    item = DummyItem()  # type: ignore[var-annotated]
    result = strategy.extract_model_outputs(item, "lm_score")  # type: ignore[arg-type]
    assert result is None


def test_extract_model_outputs_with_metadata() -> None:
    """Test extract_model_outputs from item_metadata."""

    class DummyItem:
        item_metadata = {"lm_score1": -5.0, "lm_score2": -3.0}

    strategy = ConcreteStrategy()
    item = DummyItem()  # type: ignore[var-annotated]
    result = strategy.extract_model_outputs(item, "lm_score", required_count=2)  # type: ignore[arg-type]
    assert result == [-5.0, -3.0]


def test_extract_model_outputs_wrong_count() -> None:
    """Test extract_model_outputs returns None when count doesn't match."""

    class DummyItem:
        item_metadata = {"lm_score1": -5.0}

    strategy = ConcreteStrategy()
    item = DummyItem()  # type: ignore[var-annotated]
    result = strategy.extract_model_outputs(item, "lm_score", required_count=2)  # type: ignore[arg-type]
    assert result is None
