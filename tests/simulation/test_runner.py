"""Tests for simulation runner."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from bead.config.models import (
    NoiseModelConfig,
    SimulatedAnnotatorConfig,
    SimulationRunnerConfig,
)
from bead.items.models import (
    Item,
    ItemTemplate,
    ModelOutput,
    PresentationSpec,
    TaskSpec,
)
from bead.simulation.runner import SimulationRunner


def test_runner_instantiation() -> None:
    """Test that runner can be instantiated."""
    config = SimulationRunnerConfig(
        annotator_configs=[SimulatedAnnotatorConfig(strategy="lm_score")],
        n_annotators=1,
    )
    runner = SimulationRunner(config)
    assert len(runner.annotators) == 1


def test_runner_single_annotator() -> None:
    """Test runner with single annotator."""
    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            )
        ],
        n_annotators=1,
    )
    runner = SimulationRunner(config)

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
        )
    ]

    results = runner.run(items, template)

    assert "item_ids" in results
    assert "annotator_0" in results
    assert len(results["item_ids"]) == 1
    assert len(results["annotator_0"]) == 1
    assert results["annotator_0"][0] in ["option_a", "option_b"]


def test_runner_multiple_annotators() -> None:
    """Test runner with multiple annotators."""
    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            ),
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=43,
            ),
        ],
        n_annotators=2,
    )
    runner = SimulationRunner(config)
    assert len(runner.annotators) == 2

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
        )
    ]

    results = runner.run(items, template)

    assert "item_ids" in results
    assert "annotator_0" in results
    assert "annotator_1" in results
    assert len(results["annotator_0"]) == 1
    assert len(results["annotator_1"]) == 1


def test_runner_replicates_config_when_n_annotators_exceeds_configs() -> None:
    """Test that runner replicates configs when n_annotators > len(configs)."""
    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            )
        ],
        n_annotators=3,  # More than configs
    )
    runner = SimulationRunner(config)
    assert len(runner.annotators) == 3

    # Check that different random states are used
    assert runner.annotators[0].random_state == 42
    assert runner.annotators[1].random_state == 43
    assert runner.annotators[2].random_state == 44


def test_runner_with_multiple_items() -> None:
    """Test runner with multiple items."""
    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            )
        ],
        n_annotators=1,
    )
    runner = SimulationRunner(config)

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
                    output=-3.0,
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

    results = runner.run(items, template)

    assert len(results["item_ids"]) == 2
    assert len(results["annotator_0"]) == 2


def test_runner_with_list_of_templates() -> None:
    """Test runner with list of templates (one per item)."""
    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            )
        ],
        n_annotators=1,
    )
    runner = SimulationRunner(config)

    template1 = ItemTemplate(
        name="test_2afc_1",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    template2 = ItemTemplate(
        name="test_2afc_2",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is best?", options=["option_a", "option_b"]),
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
                    output=-3.0,
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

    results = runner.run(items, [template1, template2])

    assert len(results["item_ids"]) == 2
    assert len(results["annotator_0"]) == 2


def test_save_results_jsonl(tmp_path: Path) -> None:
    """Test saving results in JSONL format."""
    save_path = tmp_path / "results.jsonl"

    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            )
        ],
        n_annotators=1,
        output_format="jsonl",
        save_path=save_path,
    )
    runner = SimulationRunner(config)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item_id = uuid4()
    items = [
        Item(
            id=item_id,
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
    ]

    runner.run(items, template)

    # Check file was created
    assert save_path.exists()

    # Read and verify content
    with open(save_path) as f:
        lines = f.readlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["item_id"] == str(item_id)
        assert "annotator_0" in row


def test_save_results_dict(tmp_path: Path) -> None:
    """Test saving results in JSON dict format."""
    save_path = tmp_path / "results.json"

    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            )
        ],
        n_annotators=1,
        output_format="dict",
        save_path=save_path,
    )
    runner = SimulationRunner(config)

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
        )
    ]

    runner.run(items, template)

    # Check file was created
    assert save_path.exists()

    # Read and verify content
    with open(save_path) as f:
        data = json.load(f)
        assert "item_ids" in data
        assert "annotator_0" in data


def test_save_results_dataframe(tmp_path: Path) -> None:
    """Test saving results in CSV (dataframe) format."""
    save_path = tmp_path / "results.csv"

    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            )
        ],
        n_annotators=1,
        output_format="dataframe",
        save_path=save_path,
    )
    runner = SimulationRunner(config)

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
        )
    ]

    runner.run(items, template)

    # Check file was created
    assert save_path.exists()

    # Verify it's a valid CSV (optional test dependency)
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(save_path)
    assert "item_ids" in df.columns
    assert "annotator_0" in df.columns


def test_save_results_without_save_path_raises() -> None:
    """Test that save_results raises if save_path not configured."""
    config = SimulationRunnerConfig(
        annotator_configs=[SimulatedAnnotatorConfig(strategy="lm_score")],
        n_annotators=1,
        # No save_path
    )
    runner = SimulationRunner(config)

    results = {"item_ids": ["1", "2"], "annotator_0": ["a", "b"]}

    with pytest.raises(ValueError, match="save_path not configured"):
        runner.save_results(results)


def test_runner_result_structure() -> None:
    """Test that runner results have correct structure."""
    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=42,
            ),
            SimulatedAnnotatorConfig(
                strategy="lm_score",
                noise_model=NoiseModelConfig(noise_type="none"),
                random_state=43,
            ),
        ],
        n_annotators=2,
    )
    runner = SimulationRunner(config)

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
                    output=-3.0,
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

    results = runner.run(items, template)

    # Check structure
    assert isinstance(results, dict)
    assert "item_ids" in results
    assert all(f"annotator_{i}" in results for i in range(2))

    # Check all lists have same length
    n_items = len(results["item_ids"])
    assert all(len(results[key]) == n_items for key in results)

    # Check item IDs match
    assert results["item_ids"] == [str(item.id) for item in items]
