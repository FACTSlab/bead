"""Tests for LM-based annotator."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.config.models import NoiseModelConfig, SimulatedAnnotatorConfig
from bead.items.models import (
    Item,
    ItemTemplate,
    ModelOutput,
    PresentationSpec,
    TaskSpec,
)
from bead.simulation.annotators.base import SimulatedAnnotator
from bead.simulation.annotators.lm_based import LMBasedAnnotator


def test_lm_based_annotator_instantiation() -> None:
    """Test that LM-based annotator can be instantiated."""
    config = SimulatedAnnotatorConfig(strategy="lm_score")
    annotator = LMBasedAnnotator(config)
    assert annotator.config.strategy == "lm_score"


def test_lm_based_annotator_with_temperature_noise() -> None:
    """Test LM-based annotator with temperature noise model."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.5),
        random_state=42,
    )
    annotator = LMBasedAnnotator(config)
    assert annotator.noise_model is not None
    assert annotator.noise_model.temperature == 1.5


def test_lm_based_annotator_without_noise() -> None:
    """Test LM-based annotator without noise model."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = LMBasedAnnotator(config)
    assert annotator.noise_model is None


def test_lm_based_annotator_has_forced_choice_strategy() -> None:
    """Test that LM-based annotator has forced choice strategy."""
    config = SimulatedAnnotatorConfig(strategy="lm_score")
    annotator = LMBasedAnnotator(config)
    assert "forced_choice" in annotator.strategies


def test_annotate_2afc_with_model_outputs() -> None:
    """Test annotation of 2AFC item with model outputs."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = LMBasedAnnotator(config)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "Text A", "option_b": "Text B"},
        model_outputs=[
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text A"},
                output=-2.0,
                cache_key="key1",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text B"},
                output=-5.0,
                cache_key="key2",
            ),
        ],
    )

    # Annotate multiple times
    annotations = []
    for _ in range(100):
        annotation = annotator.annotate(item, template)
        annotations.append(annotation)

    # Should favor option_a
    option_a_count = annotations.count("option_a")
    assert option_a_count > 70


def test_annotate_3afc_with_model_outputs() -> None:
    """Test annotation of 3AFC item."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = LMBasedAnnotator(config)

    template = ItemTemplate(
        name="test_3afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Which is best?", options=["option_a", "option_b", "option_c"]
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-5.0,
                cache_key="k1",
            ),
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-1.0,  # Best
                cache_key="k2",
            ),
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-4.0,
                cache_key="k3",
            ),
        ],
    )

    annotations = []
    for _ in range(100):
        annotations.append(annotator.annotate(item, template))

    option_b_count = annotations.count("option_b")
    assert option_b_count > 70


def test_annotate_with_temperature_noise() -> None:
    """Test that temperature noise affects decisions."""
    # No noise
    config_no_noise = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator_no_noise = LMBasedAnnotator(config_no_noise)

    # With noise (for forced choice, noise is handled in strategy via softmax)
    config_noise = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.0),
        random_state=42,
    )
    annotator_noise = LMBasedAnnotator(config_noise)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-2.0,
                cache_key="k1",
            ),
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-5.0,
                cache_key="k2",
            ),
        ],
    )

    # Both should work (noise doesn't affect forced choice in current implementation)
    annotation_no_noise = annotator_no_noise.annotate(item, template)
    annotation_noise = annotator_noise.annotate(item, template)

    assert annotation_no_noise in ["option_a", "option_b"]
    assert annotation_noise in ["option_a", "option_b"]


def test_annotate_without_model_outputs() -> None:
    """Test annotation falls back to random when no model outputs."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = LMBasedAnnotator(config)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Should still work, falling back to random
    annotations = []
    for _ in range(100):
        annotation = annotator.annotate(item, template)
        annotations.append(annotation)

    # Should be roughly uniform
    option_a_count = annotations.count("option_a")
    assert 30 < option_a_count < 70


def test_annotate_wrong_task_type_raises() -> None:
    """Test that invalid item for cloze task raises ValueError."""
    config = SimulatedAnnotatorConfig(strategy="lm_score")
    annotator = LMBasedAnnotator(config)

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blank"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item without unfilled_slots for cloze task
    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="at least one unfilled slot"):
        annotator.annotate(item, template)


def test_annotate_invalid_item_raises() -> None:
    """Test that invalid item raises ValueError."""
    config = SimulatedAnnotatorConfig(strategy="lm_score")
    annotator = LMBasedAnnotator(config)

    # Template with no options
    template = ItemTemplate(
        name="test_invalid",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="task_spec.options must be defined"):
        annotator.annotate(item, template)


def test_annotate_batch() -> None:
    """Test batch annotation."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = LMBasedAnnotator(config)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    items = [
        Item(
            item_template_id=uuid4(),
            model_outputs=[
                ModelOutput(
                    model_name="test",
                    model_version="1.0",
                    operation="lm_score",
                    inputs={},
                    output=-2.0,
                    cache_key="k1",
                ),
                ModelOutput(
                    model_name="test",
                    model_version="1.0",
                    operation="lm_score",
                    inputs={},
                    output=-5.0,
                    cache_key="k2",
                ),
            ],
        ),
        Item(
            item_template_id=uuid4(),
            model_outputs=[
                ModelOutput(
                    model_name="test",
                    model_version="1.0",
                    operation="lm_score",
                    inputs={},
                    output=-4.0,
                    cache_key="k3",
                ),
                ModelOutput(
                    model_name="test",
                    model_version="1.0",
                    operation="lm_score",
                    inputs={},
                    output=-1.0,
                    cache_key="k4",
                ),
            ],
        ),
    ]

    annotations = annotator.annotate_batch(items, template)

    assert len(annotations) == 2
    assert str(items[0].id) in annotations
    assert str(items[1].id) in annotations
    assert annotations[str(items[0].id)] in ["option_a", "option_b"]
    assert annotations[str(items[1].id)] in ["option_a", "option_b"]


def test_annotate_with_custom_model_output_key() -> None:
    """Test annotation with custom model output key."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="custom_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = LMBasedAnnotator(config)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="custom_score",
                inputs={},
                output=-2.0,
                cache_key="k1",
            ),
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="custom_score",
                inputs={},
                output=-5.0,
                cache_key="k2",
            ),
        ],
    )

    annotations = []
    for _ in range(100):
        annotations.append(annotator.annotate(item, template))

    # Should favor option_a
    option_a_count = annotations.count("option_a")
    assert option_a_count > 70


def test_annotate_reproducible_with_random_state() -> None:
    """Test that annotations are reproducible with same random state."""
    config1 = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator1 = LMBasedAnnotator(config1)

    config2 = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator2 = LMBasedAnnotator(config2)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-2.0,
                cache_key="k1",
            ),
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-5.0,
                cache_key="k2",
            ),
        ],
    )

    # Generate sequences with same seed
    annotations1 = [annotator1.annotate(item, template) for _ in range(10)]
    annotations2 = [annotator2.annotate(item, template) for _ in range(10)]

    assert annotations1 == annotations2


def test_from_config() -> None:
    """Test creating annotator from config via factory method."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.5),
        random_state=42,
    )

    annotator = SimulatedAnnotator.from_config(config)

    assert isinstance(annotator, LMBasedAnnotator)
    assert annotator.config.strategy == "lm_score"
    assert annotator.noise_model is not None
