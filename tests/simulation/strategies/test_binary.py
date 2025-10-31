"""Tests for binary simulation strategy."""

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
from bead.simulation.strategies.binary import BinaryStrategy


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
    strategy = BinaryStrategy()
    assert strategy.supported_task_type == "binary"


def test_validate_item_correct_task_type() -> None:
    """Test validation passes with correct task type."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"sentence": "The cat sat on the mat."},
    )

    # Should not raise
    strategy.validate_item(item, template)


def test_validate_item_wrong_task_type() -> None:
    """Test validation fails with wrong task type."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_forced_choice",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["a", "b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="Expected task_type 'binary'"):
        strategy.validate_item(item, template)


def test_simulate_response_with_positive_score() -> None:
    """Test response with positive score tends toward True."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Large positive score should give high probability of True
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(5.0)],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses to test distribution
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly True (>90%)
    true_count = sum(responses)
    assert true_count > 90


def test_simulate_response_with_negative_score() -> None:
    """Test response with negative score tends toward False."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Large negative score should give high probability of False
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-5.0)],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses to test distribution
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly False (<10%)
    true_count = sum(responses)
    assert true_count < 10


def test_simulate_response_with_zero_score() -> None:
    """Test response with zero score gives ~50% probability."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Zero score should give ~0.5 probability
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(0.0)],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses to test distribution
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(1000)
    ]

    # Should be approximately 50/50 (within 10%)
    true_count = sum(responses)
    assert 400 < true_count < 600


def test_simulate_response_without_model_outputs() -> None:
    """Test fallback to random when model outputs missing."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item without model outputs
    item = Item(item_template_id=uuid4())

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(1000)
    ]

    # Should be approximately 50/50 (within 10%)
    true_count = sum(responses)
    assert 400 < true_count < 600


def test_simulate_response_with_item_metadata() -> None:
    """Test extraction from item_metadata."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Score in item_metadata
    item = Item(
        item_template_id=uuid4(),
        item_metadata={"lm_score1": 3.0},
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly True with positive score
    true_count = sum(responses)
    assert true_count > 80


def test_simulate_response_deterministic_with_seed() -> None:
    """Test that responses are deterministic with fixed seed."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(0.5)],
    )

    # Generate responses with same seed
    rng1 = np.random.RandomState(123)
    response1 = strategy.simulate_response(item, template, "lm_score", rng1)

    rng2 = np.random.RandomState(123)
    response2 = strategy.simulate_response(item, template, "lm_score", rng2)

    # Should be identical
    assert response1 == response2


def test_simulate_response_different_with_different_seed() -> None:
    """Test that responses differ with different seeds."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(0.0)],
    )

    # Generate many responses with different seeds
    responses = []
    for seed in range(100):
        rng = np.random.RandomState(seed)
        response = strategy.simulate_response(item, template, "lm_score", rng)
        responses.append(response)

    # Should have both True and False
    assert True in responses
    assert False in responses


def test_extract_model_outputs_single_score() -> None:
    """Test extracting single score from model outputs."""
    strategy = BinaryStrategy()

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(2.5)],
    )

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=1)

    assert scores is not None
    assert len(scores) == 1
    assert scores[0] == 2.5


def test_extract_model_outputs_missing() -> None:
    """Test that missing outputs return None."""
    strategy = BinaryStrategy()

    item = Item(item_template_id=uuid4())

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=1)

    assert scores is None


def test_extract_model_outputs_wrong_count() -> None:
    """Test that wrong count returns None."""
    strategy = BinaryStrategy()

    # Item has 2 scores but we need 1
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(2.5),
            _create_model_output(3.5),
        ],
    )

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=1)

    # Should return None because count mismatch
    assert scores is None


def test_simulate_response_with_multiple_model_outputs() -> None:
    """Test that only first score is used when multiple provided."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item with multiple scores (should fallback to random)
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(5.0),
            _create_model_output(-5.0),
        ],
    )

    rng = np.random.RandomState(42)

    # Should fallback to random because count mismatch
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(1000)
    ]

    # Should be approximately 50/50
    true_count = sum(responses)
    assert 400 < true_count < 600


def test_binary_response_type() -> None:
    """Test that response is always a boolean."""
    strategy = BinaryStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(1.5)],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert isinstance(response, (bool, np.bool_))
