"""Tests for categorical simulation strategy."""

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
from bead.simulation.strategies.categorical import CategoricalStrategy


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
    strategy = CategoricalStrategy()
    assert strategy.supported_task_type == "categorical"


def test_validate_item_correct_task_type() -> None:
    """Test validation passes with correct task type."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_nli",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="What is the relationship?",
            options=["Entailment", "Neutral", "Contradiction"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"premise": "A", "hypothesis": "B"},
    )

    # Should not raise
    strategy.validate_item(item, template)


def test_validate_item_wrong_task_type() -> None:
    """Test validation fails with wrong task type."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="Expected task_type 'categorical'"):
        strategy.validate_item(item, template)


def test_validate_item_no_options() -> None:
    """Test validation fails when options not defined."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_no_options",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(prompt="Classify this"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="task_spec.options must be defined"):
        strategy.validate_item(item, template)


def test_validate_item_too_few_options() -> None:
    """Test validation fails with fewer than 2 options."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_one_option",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(prompt="Classify this", options=["Only one"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="categorical requires at least 2 options"):
        strategy.validate_item(item, template)


def test_simulate_response_with_clear_winner() -> None:
    """Test response with clear highest score."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_nli",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="What is the relationship?",
            options=["Entailment", "Neutral", "Contradiction"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Clear winner: Entailment has much higher score
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(10.0),  # Entailment
            _create_model_output(-5.0),  # Neutral
            _create_model_output(-5.0),  # Contradiction
        ],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly "Entailment"
    entailment_count = sum(1 for r in responses if r == "Entailment")
    assert entailment_count > 95


def test_simulate_response_with_uniform_scores() -> None:
    """Test response with equal scores gives uniform distribution."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_nli",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="What is the relationship?",
            options=["Entailment", "Neutral", "Contradiction"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Equal scores
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(1.0),
            _create_model_output(1.0),
            _create_model_output(1.0),
        ],
    )

    rng = np.random.RandomState(42)

    # Generate many responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(900)
    ]

    # Should be roughly uniform (each ~300 ± 50)
    counts = {
        "Entailment": sum(1 for r in responses if r == "Entailment"),
        "Neutral": sum(1 for r in responses if r == "Neutral"),
        "Contradiction": sum(1 for r in responses if r == "Contradiction"),
    }

    assert all(250 < count < 350 for count in counts.values())


def test_simulate_response_without_model_outputs() -> None:
    """Test fallback to random when model outputs missing."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_nli",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="What is the relationship?",
            options=["Entailment", "Neutral", "Contradiction"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item without model outputs
    item = Item(item_template_id=uuid4())

    rng = np.random.RandomState(42)

    # Generate many responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(900)
    ]

    # Should be roughly uniform
    counts = {
        "Entailment": sum(1 for r in responses if r == "Entailment"),
        "Neutral": sum(1 for r in responses if r == "Neutral"),
        "Contradiction": sum(1 for r in responses if r == "Contradiction"),
    }

    assert all(250 < count < 350 for count in counts.values())


def test_simulate_response_with_two_categories() -> None:
    """Test with binary categorical (2 options)."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_sentiment",
        judgment_type="preference",
        task_type="categorical",
        task_spec=TaskSpec(prompt="Sentiment?", options=["Positive", "Negative"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Positive is favored
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(3.0),  # Positive
            _create_model_output(-2.0),  # Negative
        ],
    )

    rng = np.random.RandomState(42)

    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly Positive
    positive_count = sum(1 for r in responses if r == "Positive")
    assert positive_count > 90


def test_simulate_response_with_many_categories() -> None:
    """Test with many categories (5 options)."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_multiclass",
        judgment_type="similarity",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="Classify",
            options=["Class A", "Class B", "Class C", "Class D", "Class E"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Class C is strongly favored
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(-2.0),  # A
            _create_model_output(-1.0),  # B
            _create_model_output(5.0),  # C (winner)
            _create_model_output(-1.0),  # D
            _create_model_output(-2.0),  # E
        ],
    )

    rng = np.random.RandomState(42)

    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly Class C
    class_c_count = sum(1 for r in responses if r == "Class C")
    assert class_c_count > 95


def test_simulate_response_with_item_metadata() -> None:
    """Test extraction from item_metadata."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_nli",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="What is the relationship?",
            options=["Entailment", "Neutral", "Contradiction"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Scores in item_metadata
    item = Item(
        item_template_id=uuid4(),
        item_metadata={
            "lm_score1": 5.0,
            "lm_score2": -2.0,
            "lm_score3": -2.0,
        },
    )

    rng = np.random.RandomState(42)

    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly Entailment
    entailment_count = sum(1 for r in responses if r == "Entailment")
    assert entailment_count > 95


def test_simulate_response_deterministic_with_seed() -> None:
    """Test that responses are deterministic with fixed seed."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_nli",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="What is the relationship?",
            options=["Entailment", "Neutral", "Contradiction"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(1.0),
            _create_model_output(2.0),
            _create_model_output(3.0),
        ],
    )

    # Generate responses with same seed
    rng1 = np.random.RandomState(123)
    response1 = strategy.simulate_response(item, template, "lm_score", rng1)

    rng2 = np.random.RandomState(123)
    response2 = strategy.simulate_response(item, template, "lm_score", rng2)

    # Should be identical
    assert response1 == response2


def test_simulate_response_returns_string() -> None:
    """Test that response is always a string from options."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_nli",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="What is the relationship?",
            options=["Entailment", "Neutral", "Contradiction"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(1.0),
            _create_model_output(2.0),
            _create_model_output(3.0),
        ],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert isinstance(response, str)
    assert response in ["Entailment", "Neutral", "Contradiction"]


def test_extract_model_outputs_correct_count() -> None:
    """Test extracting correct number of scores."""
    strategy = CategoricalStrategy()

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(1.0),
            _create_model_output(2.0),
            _create_model_output(3.0),
        ],
    )

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=3)

    assert scores is not None
    assert len(scores) == 3
    assert scores == [1.0, 2.0, 3.0]


def test_extract_model_outputs_wrong_count() -> None:
    """Test that wrong count returns None."""
    strategy = CategoricalStrategy()

    # Item has 3 scores but we need 2
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(1.0),
            _create_model_output(2.0),
            _create_model_output(3.0),
        ],
    )

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=2)

    # Should return None because count mismatch
    assert scores is None


def test_simulate_response_with_extreme_scores() -> None:
    """Test with extreme score differences."""
    strategy = CategoricalStrategy()

    template = ItemTemplate(
        name="test_nli",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="What is the relationship?",
            options=["Entailment", "Neutral", "Contradiction"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Extreme scores (test numerical stability)
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(100.0),
            _create_model_output(-100.0),
            _create_model_output(-100.0),
        ],
    )

    rng = np.random.RandomState(42)

    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should all be Entailment
    assert all(r == "Entailment" for r in responses)
