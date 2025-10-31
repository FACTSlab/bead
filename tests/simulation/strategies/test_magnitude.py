"""Tests for magnitude estimation simulation strategy."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from bead.items.models import (
    Item,
    ItemTemplate,
    ModelOutput,
    PresentationSpec,
    TaskSpec,
)
from bead.simulation.strategies.magnitude import MagnitudeStrategy


def _create_model_output(score: float) -> ModelOutput:
    """Create ModelOutput with required fields."""
    return ModelOutput(
        model_name="test_model",
        model_version="1.0",
        operation="lm_score",
        inputs={},
        output=score,
        cache_key=f"key_{score}",
    )


def test_strategy_instantiation() -> None:
    """Test that strategy can be instantiated."""
    strategy = MagnitudeStrategy()
    assert strategy.supported_task_type == "magnitude"
    assert strategy.scale_factor == 10.0


def test_strategy_with_custom_scale_factor() -> None:
    """Test instantiation with custom scale factor."""
    strategy = MagnitudeStrategy(scale_factor=5.0)
    assert strategy.scale_factor == 5.0


def test_validate_item_correct_task_type() -> None:
    """Test validation passes with correct task type."""
    strategy = MagnitudeStrategy()

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"stimulus": "Some text"},
    )

    # Should not raise
    strategy.validate_item(item, template)


def test_validate_item_wrong_task_type() -> None:
    """Test validation fails with wrong task type."""
    strategy = MagnitudeStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this good?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="Expected task_type 'magnitude'"):
        strategy.validate_item(item, template)


def test_simulate_response_with_negative_score() -> None:
    """Test response with negative LM score (typical case)."""
    strategy = MagnitudeStrategy(scale_factor=10.0)

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Negative LM score (typical)
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-20.0)],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    # Should be positive
    assert response > 0
    # exp(-(-20)/10) = exp(2) ≈ 7.39
    assert 6 < response < 9


def test_simulate_response_with_positive_score() -> None:
    """Test response with positive score."""
    strategy = MagnitudeStrategy(scale_factor=10.0)

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Positive score
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(10.0)],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    # Should be positive
    assert response > 0
    # exp(10/10) = exp(1) ≈ 2.72
    assert 2 < response < 3.5


def test_simulate_response_different_scale_factors() -> None:
    """Test that scale factor affects output range."""
    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-30.0)],
    )

    rng = np.random.RandomState(42)

    # Smaller scale factor -> larger outputs
    strategy1 = MagnitudeStrategy(scale_factor=5.0)
    response1 = strategy1.simulate_response(item, template, "lm_score", rng)

    rng = np.random.RandomState(42)
    # Larger scale factor -> smaller outputs
    strategy2 = MagnitudeStrategy(scale_factor=15.0)
    response2 = strategy2.simulate_response(item, template, "lm_score", rng)

    # exp(-(-30)/5) = exp(6) vs exp(-(-30)/15) = exp(2)
    assert response1 > response2


def test_simulate_response_without_model_outputs() -> None:
    """Test fallback to random when model outputs missing."""
    strategy = MagnitudeStrategy()

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item without model outputs
    item = Item(item_template_id=uuid4())

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # All should be positive
    assert all(r > 0 for r in responses)
    # Should have some variation (log-normal)
    assert max(responses) > min(responses) * 2


def test_simulate_response_with_item_metadata() -> None:
    """Test extraction from item_metadata."""
    strategy = MagnitudeStrategy(scale_factor=10.0)

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Score in item_metadata
    item = Item(
        item_template_id=uuid4(),
        item_metadata={"lm_score1": -10.0},
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    # exp(-(-10)/10) = exp(1) ≈ 2.72
    assert 2 < response < 3.5


def test_simulate_response_deterministic_with_seed() -> None:
    """Test that responses are deterministic with fixed seed."""
    strategy = MagnitudeStrategy()

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-15.0)],
    )

    # Generate responses with same seed
    rng1 = np.random.RandomState(123)
    response1 = strategy.simulate_response(item, template, "lm_score", rng1)

    rng2 = np.random.RandomState(123)
    response2 = strategy.simulate_response(item, template, "lm_score", rng2)

    # Should be identical
    assert response1 == response2


def test_simulate_response_different_with_different_scores() -> None:
    """Test that different scores give different magnitudes."""
    strategy = MagnitudeStrategy(scale_factor=10.0)

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    rng = np.random.RandomState(42)

    # Better score (less negative)
    item1 = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-10.0)],
    )
    response1 = strategy.simulate_response(item1, template, "lm_score", rng)

    rng = np.random.RandomState(42)
    # Worse score (more negative)
    item2 = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-30.0)],
    )
    response2 = strategy.simulate_response(item2, template, "lm_score", rng)

    # Better score should give larger magnitude
    assert response1 < response2


def test_extract_model_outputs_single_score() -> None:
    """Test extracting single score from model outputs."""
    strategy = MagnitudeStrategy()

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-12.5)],
    )

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=1)

    assert scores is not None
    assert len(scores) == 1
    assert scores[0] == -12.5


def test_extract_model_outputs_missing() -> None:
    """Test that missing outputs return None."""
    strategy = MagnitudeStrategy()

    item = Item(item_template_id=uuid4())

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=1)

    assert scores is None


def test_magnitude_response_type() -> None:
    """Test that response is always a float."""
    strategy = MagnitudeStrategy()

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-20.0)],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert isinstance(response, float)


def test_magnitude_always_positive() -> None:
    """Test that magnitude is always positive."""
    strategy = MagnitudeStrategy()

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Rate the magnitude:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    rng = np.random.RandomState(42)

    # Test various scores
    for score in [-100, -50, -10, 0, 10, 50, 100]:
        item = Item(
            item_template_id=uuid4(),
            model_outputs=[_create_model_output(float(score))],
        )
        response = strategy.simulate_response(item, template, "lm_score", rng)
        assert response > 0, f"Magnitude should be positive for score {score}"
