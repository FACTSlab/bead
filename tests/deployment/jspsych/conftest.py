"""Shared pytest fixtures for jsPsych deployment tests."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from sash.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from sash.items.models import Item
from sash.lists.constraints import OrderingConstraint
from sash.lists.models import ExperimentList


@pytest.fixture
def sample_experiment_config() -> ExperimentConfig:
    """Create a sample experiment configuration.

    Returns
    -------
    ExperimentConfig
        Sample experiment configuration for testing.
    """
    return ExperimentConfig(
        experiment_type="likert_rating",
        title="Test Acceptability Study",
        description="Test experiment for rating sentence acceptability",
        instructions="Please rate each sentence on a scale from 1 to 7.",
        randomize_trial_order=True,
        show_progress_bar=True,
        ui_theme="light",
    )


@pytest.fixture
def sample_rating_config() -> RatingScaleConfig:
    """Create a sample rating scale configuration.

    Returns
    -------
    RatingScaleConfig
        Sample rating scale configuration.
    """
    return RatingScaleConfig(
        min_value=1,
        max_value=7,
        min_label="Not natural at all",
        max_label="Very natural",
        step=1,
        show_numeric_labels=True,
        required=True,
    )


@pytest.fixture
def sample_choice_config() -> ChoiceConfig:
    """Create a sample choice configuration.

    Returns
    -------
    ChoiceConfig
        Sample choice configuration.
    """
    return ChoiceConfig(
        required=True,
        randomize_choice_order=False,
    )


@pytest.fixture
def sample_item() -> Item:
    """Create a single sample item for testing.

    Returns
    -------
    Item
        Sample item with test data.
    """
    return Item(
        item_template_id=uuid4(),
        rendered_elements={"sentence": "The cat broke the vase."},
        item_metadata={"condition": "A", "is_practice": False},
    )


@pytest.fixture
def sample_items() -> dict[UUID, Item]:
    """Create sample items for testing.

    Returns
    -------
    dict[UUID, Item]
        Dictionary of sample items keyed by UUID.
    """
    items = {}

    # Create 5 sample items
    for i in range(5):
        item_id = uuid4()
        item = Item(
            id=item_id,
            item_template_id=uuid4(),
            rendered_elements={
                "sentence": f"This is test sentence number {i + 1}."
            },
            item_metadata={
                "condition": "A" if i % 2 == 0 else "B",
                "is_practice": i < 2,
                "item_number": i,
            },
        )
        items[item_id] = item

    return items


@pytest.fixture
def sample_experiment_list(sample_items: dict[UUID, Item]) -> ExperimentList:
    """Create a sample experiment list.

    Parameters
    ----------
    sample_items : dict[UUID, Item]
        Sample items fixture.

    Returns
    -------
    ExperimentList
        Sample experiment list with ordering constraints.
    """
    # Create ordering constraint
    constraint = OrderingConstraint(
        practice_item_property="item_metadata.is_practice",
        no_adjacent_property="item_metadata.condition",
    )

    # Create experiment list
    exp_list = ExperimentList(name="test_list", list_number=0, list_constraints=[constraint])
    for item_id in sample_items.keys():
        exp_list.add_item(item_id)

    return exp_list


@pytest.fixture
def sample_precedence_constraint() -> OrderingConstraint:
    """Create a sample precedence constraint.

    Returns
    -------
    OrderingConstraint
        Ordering constraint with precedence pairs.
    """
    item1 = UUID("12345678-1234-5678-1234-567812345678")
    item2 = UUID("87654321-4321-8765-4321-876543218765")

    return OrderingConstraint(
        precedence_pairs=[(item1, item2)],
    )


@pytest.fixture
def sample_blocking_constraint() -> OrderingConstraint:
    """Create a sample blocking constraint.

    Returns
    -------
    OrderingConstraint
        Ordering constraint with blocking.
    """
    return OrderingConstraint(
        block_by_property="item_metadata.block_type",
        randomize_within_blocks=True,
    )


@pytest.fixture
def sample_distance_constraint() -> OrderingConstraint:
    """Create a sample distance constraint.

    Returns
    -------
    OrderingConstraint
        Ordering constraint with distance specifications.
    """
    return OrderingConstraint(
        no_adjacent_property="item_metadata.condition",
        min_distance=2,
        max_distance=10,
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest tmp_path fixture.

    Returns
    -------
    Path
        Temporary output directory.
    """
    output_dir = tmp_path / "experiment_output"
    output_dir.mkdir()
    return output_dir
