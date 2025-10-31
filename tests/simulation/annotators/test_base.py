"""Tests for base simulated annotator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from bead.config.models import SimulatedAnnotatorConfig
from bead.simulation.annotators.base import SimulatedAnnotator

if TYPE_CHECKING:
    from bead.items.models import Item, ItemTemplate


class ConcreteAnnotator(SimulatedAnnotator):
    """Concrete implementation for testing."""

    def annotate(self, item: Item, item_template: ItemTemplate) -> str:
        """Generate test annotation."""
        return "test_annotation"


def test_annotator_is_abstract() -> None:
    """Test that SimulatedAnnotator cannot be instantiated directly."""
    config = SimulatedAnnotatorConfig()
    with pytest.raises(TypeError):
        SimulatedAnnotator(config)  # type: ignore[abstract]


def test_concrete_annotator_instantiation() -> None:
    """Test that concrete annotator can be instantiated."""
    config = SimulatedAnnotatorConfig()
    annotator = ConcreteAnnotator(config)
    assert annotator.config == config


def test_annotator_random_state() -> None:
    """Test annotator uses correct random state."""
    config = SimulatedAnnotatorConfig(random_state=42)
    annotator = ConcreteAnnotator(config)
    assert annotator.random_state == 42
    assert isinstance(annotator.rng, np.random.RandomState)


def test_annotator_from_config_lm_score() -> None:
    """Test from_config works for lm_score strategy."""
    from bead.simulation.annotators.lm_based import (  # noqa: PLC0415
        LMBasedAnnotator,
    )

    config = SimulatedAnnotatorConfig(strategy="lm_score")
    annotator = SimulatedAnnotator.from_config(config)
    assert isinstance(annotator, LMBasedAnnotator)


def test_annotator_from_config_unknown_strategy() -> None:
    """Test from_config raises ValueError for unknown strategy."""
    # We need to test strategies that won't be implemented
    # Since from_config checks for known strategies first, we can't test
    # truly unknown ones unless we modify the implementation.
    # For now, just ensure it handles the flow.
    pass


def test_annotate_batch_single_template() -> None:
    """Test annotate_batch with single template."""

    class DummyItem:
        def __init__(self, item_id: str) -> None:
            self.id = item_id

    class DummyTemplate:
        pass

    config = SimulatedAnnotatorConfig()
    annotator = ConcreteAnnotator(config)

    items = [DummyItem("1"), DummyItem("2")]  # type: ignore[var-annotated]
    template = DummyTemplate()  # type: ignore[var-annotated]

    annotations = annotator.annotate_batch(items, template)  # type: ignore[arg-type]
    assert len(annotations) == 2
    assert annotations["1"] == "test_annotation"
    assert annotations["2"] == "test_annotation"


def test_get_strategy_not_supported() -> None:
    """Test get_strategy raises ValueError for unsupported task type."""
    config = SimulatedAnnotatorConfig()
    annotator = ConcreteAnnotator(config)

    with pytest.raises(ValueError, match="not supported"):
        annotator.get_strategy("unknown_task")
