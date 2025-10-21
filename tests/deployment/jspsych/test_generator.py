"""Tests for JsPsychExperimentGenerator."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from sash.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from sash.deployment.jspsych.generator import JsPsychExperimentGenerator
from sash.lists.models import ExperimentList


class TestGeneratorInitialization:
    """Tests for JsPsychExperimentGenerator initialization."""

    def test_basic_initialization(
        self,
        sample_experiment_config: ExperimentConfig,
        tmp_output_dir: Path,
    ) -> None:
        """Test generator initialization."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        assert generator.config == sample_experiment_config
        assert generator.output_dir == tmp_output_dir
        assert generator.rating_config is not None
        assert generator.choice_config is not None

    def test_initialization_with_custom_configs(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_rating_config: RatingScaleConfig,
        sample_choice_config: ChoiceConfig,
        tmp_output_dir: Path,
    ) -> None:
        """Test initialization with custom rating and choice configs."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
            rating_config=sample_rating_config,
            choice_config=sample_choice_config,
        )

        assert generator.rating_config == sample_rating_config
        assert generator.choice_config == sample_choice_config


class TestDirectoryStructure:
    """Tests for directory structure creation."""

    def test_create_directory_structure(
        self,
        sample_experiment_config: ExperimentConfig,
        tmp_output_dir: Path,
    ) -> None:
        """Test directory structure creation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator._create_directory_structure()

        assert (tmp_output_dir / "css").exists()
        assert (tmp_output_dir / "js").exists()
        assert (tmp_output_dir / "data").exists()


class TestTimelineGeneration:
    """Tests for timeline data generation."""

    def test_generate_timeline_data(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test timeline data generation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        timeline_data = generator._generate_timeline_data(
            sample_experiment_list,
            sample_items,
            sample_templates,
        )

        assert "trials" in timeline_data
        assert len(timeline_data["trials"]) == len(sample_experiment_list.item_refs)

        first_trial = timeline_data["trials"][0]
        assert "type" in first_trial
        assert "stimulus" in first_trial
        assert "data" in first_trial

    def test_missing_item_raises_error(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test timeline data generation with missing item."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        sample_experiment_list.add_item(uuid4())

        with pytest.raises(ValueError, match="not found in items dictionary"):
            generator._generate_timeline_data(
                sample_experiment_list,
                sample_items,
                sample_templates,
            )

    def test_trial_order_preserved(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test that trial numbers are correctly assigned."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        timeline_data = generator._generate_timeline_data(
            sample_experiment_list,
            sample_items,
            sample_templates,
        )

        for i, trial in enumerate(timeline_data["trials"]):
            assert trial["data"]["trial_number"] == i


class TestConstraintExtraction:
    """Tests for constraint and metadata extraction."""

    def test_extract_ordering_constraints(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        tmp_output_dir: Path,
    ) -> None:
        """Test ordering constraint extraction."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        constraints = generator._extract_ordering_constraints(sample_experiment_list)

        assert len(constraints) > 0
        assert any(c.practice_item_property is not None for c in constraints)

    def test_extract_item_metadata(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test item metadata extraction."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        metadata = generator._extract_item_metadata(
            sample_experiment_list,
            sample_items,
        )

        assert len(metadata) == len(sample_experiment_list.item_refs)

        first_item_id = sample_experiment_list.item_refs[0]
        assert first_item_id in metadata
        assert "condition" in metadata[first_item_id]


class TestCompleteGeneration:
    """Tests for complete experiment generation."""

    def test_generate_experiment(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test complete experiment generation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        output_path = generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        assert output_path == tmp_output_dir
        assert (tmp_output_dir / "index.html").exists()
        assert (tmp_output_dir / "css" / "experiment.css").exists()
        assert (tmp_output_dir / "js" / "experiment.js").exists()
        assert (tmp_output_dir / "data" / "config.json").exists()

    def test_no_lists_raises_error(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test error when no lists provided."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        with pytest.raises(ValueError, match="At least one ExperimentList is required"):
            generator.generate(
                lists=[],
                items=sample_items,
                templates=sample_templates,
            )


class TestFileGeneration:
    """Tests for individual file generation."""

    def test_html_content(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test HTML file content generation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        html_content = (tmp_output_dir / "index.html").read_text()

        assert sample_experiment_config.title in html_content
        assert "jspsych" in html_content.lower()
        assert "seedrandom" in html_content.lower()

    def test_js_content(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test JavaScript file content generation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        js_content = (tmp_output_dir / "js" / "experiment.js").read_text()

        assert "initJsPsych" in js_content
        assert "timeline" in js_content
        assert sample_experiment_config.title in js_content
        assert sample_experiment_config.instructions in js_content

    def test_css_content(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test CSS file generation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        css_file = tmp_output_dir / "css" / "experiment.css"
        assert css_file.exists()

        css_content = css_file.read_text()
        assert "--primary-color" in css_content or "primary" in css_content.lower()
        assert ".jspsych-btn" in css_content

    def test_config_json(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test config.json generation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        config_file = tmp_output_dir / "data" / "config.json"
        config_data = json.loads(config_file.read_text())

        assert (
            config_data["experiment_type"] == sample_experiment_config.experiment_type
        )
        assert config_data["title"] == sample_experiment_config.title
        assert (
            config_data["randomize_trial_order"]
            == sample_experiment_config.randomize_trial_order
        )


class TestRandomization:
    """Tests for randomization handling."""

    def test_with_randomization(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test experiment generation with randomization enabled."""
        sample_experiment_config.randomize_trial_order = True

        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        js_content = (tmp_output_dir / "js" / "experiment.js").read_text()

        assert "randomizeTrials" in js_content or "function shuffle" in js_content

    def test_without_randomization(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test experiment generation without randomization."""
        sample_experiment_config.randomize_trial_order = False

        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        js_content = (tmp_output_dir / "js" / "experiment.js").read_text()

        assert (
            "original order" in js_content.lower()
            or "timeline.push(...experimentTrials)" in js_content
        )
