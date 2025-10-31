"""Tests for the simulation framework.

This module tests the SimulatedHumanAnnotator and simulation pipeline.
"""

from __future__ import annotations

import json

# Import from parent directory
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from bead.items.models import Item

sys.path.insert(0, str(Path(__file__).parent.parent))
from simulate_pipeline import (
    SimulatedHumanAnnotator,
    get_forced_choice_template,
    load_2afc_pairs,
    run_simulation,
)


class TestSimulatedHumanAnnotator:
    """Test suite for SimulatedHumanAnnotator."""

    def test_sigmoid_function(self):
        """Test sigmoid activation function."""
        annotator = SimulatedHumanAnnotator()

        # Test basic sigmoid properties
        assert abs(annotator.sigmoid(0) - 0.5) < 0.001
        assert annotator.sigmoid(10) > 0.99
        assert annotator.sigmoid(-10) < 0.01
        assert 0 < annotator.sigmoid(1) < 1

    def test_annotator_initialization(self):
        """Test annotator initialization."""
        # Default initialization
        annotator1 = SimulatedHumanAnnotator()
        assert annotator1.temperature == 1.0
        assert annotator1.random_state is None

        # Custom initialization
        annotator2 = SimulatedHumanAnnotator(temperature=2.0, random_state=42)
        assert annotator2.temperature == 2.0
        assert annotator2.random_state == 42

    def test_annotate_prefers_higher_score(self):
        """Test that annotator prefers higher-scoring options."""
        annotator = SimulatedHumanAnnotator(temperature=0.1, random_state=42)

        # Create item with clear preference (score_a >> score_b)
        item = Item(
            id=uuid4(),
            item_template_id=uuid4(),
            rendered_elements={"option_a": "A", "option_b": "B"},
            item_metadata={"lm_score1": 10.0, "lm_score2": -10.0},
        )

        # With low temperature and big score difference, should almost always choose A
        choices = [annotator.annotate(item) for _ in range(100)]
        assert choices.count("option_a") > 95

    def test_annotate_temperature_effect(self):
        """Test that temperature affects decision randomness."""
        item = Item(
            id=uuid4(),
            item_template_id=uuid4(),
            rendered_elements={"option_a": "A", "option_b": "B"},
            item_metadata={"lm_score1": 2.0, "lm_score2": 1.0},
        )

        # Low temperature: more deterministic
        annotator_cold = SimulatedHumanAnnotator(temperature=0.1, random_state=42)
        choices_cold = [annotator_cold.annotate(item) for _ in range(100)]
        prop_a_cold = choices_cold.count("option_a") / 100

        # High temperature: more random
        annotator_hot = SimulatedHumanAnnotator(temperature=10.0, random_state=42)
        choices_hot = [annotator_hot.annotate(item) for _ in range(100)]
        prop_a_hot = choices_hot.count("option_a") / 100

        # Cold should be more biased toward higher score
        assert prop_a_cold > prop_a_hot
        # Hot should be closer to 50/50
        assert 0.4 < prop_a_hot < 0.6

    def test_annotate_with_seed_reproducible(self):
        """Test that annotations are reproducible with seed."""
        item = Item(
            id=uuid4(),
            item_template_id=uuid4(),
            rendered_elements={"option_a": "A", "option_b": "B"},
            item_metadata={"lm_score1": 1.0, "lm_score2": 0.0},
        )

        annotator1 = SimulatedHumanAnnotator(random_state=42)
        annotator2 = SimulatedHumanAnnotator(random_state=42)

        choices1 = [annotator1.annotate(item) for _ in range(20)]
        choices2 = [annotator2.annotate(item) for _ in range(20)]

        assert choices1 == choices2

    def test_annotate_batch(self):
        """Test batch annotation."""
        items = [
            Item(
                id=uuid4(),
                item_template_id=uuid4(),
                rendered_elements={"option_a": f"A{i}", "option_b": f"B{i}"},
                item_metadata={"lm_score1": float(i), "lm_score2": 0.0},
            )
            for i in range(5)
        ]

        annotator = SimulatedHumanAnnotator(random_state=42)
        judgments = annotator.annotate_batch(items)

        # Check structure
        assert len(judgments) == 5
        for item in items:
            assert str(item.id) in judgments
            assert judgments[str(item.id)] in ["option_a", "option_b"]

    def test_annotate_equal_scores_random(self):
        """Test that equal scores give ~50/50 probability."""
        item = Item(
            id=uuid4(),
            item_template_id=uuid4(),
            rendered_elements={"option_a": "A", "option_b": "B"},
            item_metadata={"lm_score1": 5.0, "lm_score2": 5.0},
        )

        annotator = SimulatedHumanAnnotator(random_state=42)
        choices = [annotator.annotate(item) for _ in range(200)]

        prop_a = choices.count("option_a") / 200
        # Should be close to 0.5 (within reasonable margin)
        assert 0.4 < prop_a < 0.6


class TestGetForcedChoiceTemplate:
    """Test suite for get_forced_choice_template function."""

    def test_returns_valid_template(self):
        """Test that function returns valid ItemTemplate."""
        template = get_forced_choice_template()

        assert template.name == "2AFC Forced Choice"
        assert template.task_type.value == "forced_choice"
        assert template.language_code == "eng"
        assert len(template.slots) == 2

    def test_template_has_required_slots(self):
        """Test that template has option_a and option_b slots."""
        template = get_forced_choice_template()

        slot_names = {slot["name"] for slot in template.slots}
        assert "option_a" in slot_names
        assert "option_b" in slot_names


class TestLoad2AFCPairs:
    """Test suite for load_2afc_pairs function."""

    def test_load_with_limit(self):
        """Test loading with limit parameter."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                item = Item(
                    id=uuid4(),
                    item_template_id=uuid4(),
                    rendered_elements={"option_a": f"A{i}", "option_b": f"B{i}"},
                    item_metadata={"lm_score1": float(i), "lm_score2": 0.0},
                )
                f.write(json.dumps(item.model_dump()) + "\n")
            temp_path = Path(f.name)

        try:
            # Load with limit
            items = load_2afc_pairs(temp_path, limit=5)
            assert len(items) == 5

            # Load all
            items_all = load_2afc_pairs(temp_path)
            assert len(items_all) == 10
        finally:
            temp_path.unlink()

    def test_load_with_skip(self):
        """Test loading with skip parameter."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                item = Item(
                    id=uuid4(),
                    item_template_id=uuid4(),
                    rendered_elements={"option_a": f"A{i}", "option_b": f"B{i}"},
                    item_metadata={"lm_score1": float(i), "lm_score2": 0.0},
                )
                f.write(json.dumps(item.model_dump()) + "\n")
            temp_path = Path(f.name)

        try:
            # Skip first 3, load next 5
            items = load_2afc_pairs(temp_path, limit=5, skip=3)
            assert len(items) == 5

            # Check that we skipped the first 3
            first_item = items[0]
            assert "A3" in str(first_item.rendered_elements["option_a"])
        finally:
            temp_path.unlink()


class TestRunSimulation:
    """Test suite for run_simulation function."""

    @pytest.fixture
    def temp_items_dir(self):
        """Create temporary directory with sample 2AFC pairs."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create items subdirectory
            items_dir = tmpdir_path / "items"
            items_dir.mkdir()

            # Create sample 2AFC pairs
            pairs_path = items_dir / "2afc_pairs.jsonl"
            with open(pairs_path, "w") as f:
                for i in range(100):
                    item = Item(
                        id=uuid4(),
                        item_template_id=uuid4(),
                        rendered_elements={
                            "option_a": f"Option A {i}",
                            "option_b": f"Option B {i}",
                        },
                        item_metadata={
                            "lm_score1": float(i % 10),
                            "lm_score2": float((i + 5) % 10),
                        },
                    )
                    f.write(json.dumps(item.model_dump()) + "\n")

            # Change to temp directory
            import os

            original_dir = os.getcwd()
            os.chdir(tmpdir_path)

            yield tmpdir_path

            # Restore original directory
            os.chdir(original_dir)

    def test_simulation_completes(self, temp_items_dir):
        """Test that simulation runs to completion."""
        output_dir = temp_items_dir / "simulation_output"

        results = run_simulation(
            initial_size=20,
            budget_per_iteration=10,
            max_iterations=3,
            temperature=1.0,
            random_state=42,
            output_dir=output_dir,
            max_items=50,
        )

        # Check results structure
        assert "config" in results
        assert "human_agreement" in results
        assert "iterations" in results
        assert "converged" in results
        assert "total_annotations" in results

        # Check iterations
        assert len(results["iterations"]) <= 3
        assert results["total_annotations"] >= 20

    def test_simulation_creates_output_file(self, temp_items_dir):
        """Test that simulation creates output JSON file."""
        output_dir = temp_items_dir / "simulation_output"

        run_simulation(
            initial_size=20,
            budget_per_iteration=10,
            max_iterations=2,
            random_state=42,
            output_dir=output_dir,
            max_items=50,
        )

        results_path = output_dir / "simulation_results.json"
        assert results_path.exists()

        # Load and validate JSON
        with open(results_path) as f:
            data = json.load(f)
            assert "config" in data
            assert "iterations" in data

    def test_simulation_reproducible(self, temp_items_dir):
        """Test that simulation is reproducible with seed."""
        output_dir1 = temp_items_dir / "output1"
        output_dir2 = temp_items_dir / "output2"

        results1 = run_simulation(
            initial_size=20,
            budget_per_iteration=10,
            max_iterations=2,
            random_state=42,
            output_dir=output_dir1,
            max_items=50,
        )

        results2 = run_simulation(
            initial_size=20,
            budget_per_iteration=10,
            max_iterations=2,
            random_state=42,
            output_dir=output_dir2,
            max_items=50,
        )

        # Compare final accuracies
        assert (
            results1["iterations"][-1]["test_accuracy"]
            == results2["iterations"][-1]["test_accuracy"]
        )
        assert results1["human_agreement"] == results2["human_agreement"]
