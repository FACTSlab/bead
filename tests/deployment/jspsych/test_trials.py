"""Tests for trial generation."""

from __future__ import annotations

from uuid import uuid4

import pytest

from sash.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from sash.deployment.jspsych.trials import (
    _generate_stimulus_html,
    create_completion_trial,
    create_consent_trial,
    create_instruction_trial,
    create_trial,
)
from sash.items.models import Item


class TestCreateTrial:
    """Tests for create_trial() with different experiment types."""

    def test_likert_rating(
        self,
        sample_item: Item,
        sample_experiment_config: ExperimentConfig,
        sample_rating_config: RatingScaleConfig,
    ) -> None:
        """Test Likert rating trial creation."""
        trial = create_trial(
            item=sample_item,
            experiment_config=sample_experiment_config,
            trial_number=0,
            rating_config=sample_rating_config,
        )

        assert trial["type"] == "html-button-response"
        assert len(trial["choices"]) == 7
        assert trial["data"]["item_id"] == str(sample_item.id)
        assert trial["data"]["trial_type"] == "likert_rating"

    def test_slider_rating(self, sample_item: Item) -> None:
        """Test slider rating trial creation."""
        config = ExperimentConfig(
            experiment_type="slider_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
        )
        rating_config = RatingScaleConfig(min_value=1, max_value=7)

        trial = create_trial(
            item=sample_item,
            experiment_config=config,
            trial_number=0,
            rating_config=rating_config,
        )

        assert trial["type"] == "html-slider-response"
        assert trial["min"] == 1
        assert trial["max"] == 7
        assert trial["data"]["trial_type"] == "slider_rating"

    def test_binary_choice(self, sample_item: Item) -> None:
        """Test binary choice trial creation."""
        config = ExperimentConfig(
            experiment_type="binary_choice",
            title="Test",
            description="Test",
            instructions="Test instructions",
        )
        choice_config = ChoiceConfig()

        trial = create_trial(
            item=sample_item,
            experiment_config=config,
            trial_number=0,
            choice_config=choice_config,
        )

        assert trial["type"] == "html-button-response"
        assert trial["choices"] == ["Yes", "No"]
        assert trial["data"]["trial_type"] == "binary_choice"

    def test_forced_choice(self) -> None:
        """Test forced choice trial creation."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "sentence": "Which is more natural?",
                "choice_0": "The cat broke the vase.",
                "choice_1": "The vase was broken by the cat.",
            },
        )

        config = ExperimentConfig(
            experiment_type="forced_choice",
            title="Test",
            description="Test",
            instructions="Test instructions",
        )
        choice_config = ChoiceConfig()

        trial = create_trial(
            item=item,
            experiment_config=config,
            trial_number=0,
            choice_config=choice_config,
        )

        assert trial["type"] == "html-button-response"
        assert len(trial["choices"]) == 2
        assert trial["data"]["trial_type"] == "forced_choice"

    def test_missing_config_raises_error(self) -> None:
        """Test trial creation with missing required config."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
        )

        with pytest.raises(ValueError, match="rating_config required"):
            create_trial(
                item=item,
                experiment_config=config,
                trial_number=0,
            )

    def test_unknown_type_raises_error(self) -> None:
        """Test that Pydantic validates experiment type."""
        Item(
            item_template_id=uuid4(),
            rendered_elements={"sentence": "Test sentence."},
        )

        # Test that Pydantic validation prevents invalid experiment types
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExperimentConfig(
                experiment_type="invalid_type",  # type: ignore
                title="Test",
                description="Test",
                instructions="Test instructions",
            )

    def test_metadata_inclusion(self, sample_item: Item) -> None:
        """Test that item metadata is included in trial data."""
        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
        )
        rating_config = RatingScaleConfig()

        trial = create_trial(
            item=sample_item,
            experiment_config=config,
            trial_number=5,
            rating_config=rating_config,
        )

        assert trial["data"]["trial_number"] == 5
        assert trial["data"]["item_metadata"] == sample_item.item_metadata


class TestLikertConfiguration:
    """Tests for Likert trial configuration."""

    def test_custom_labels(self) -> None:
        """Test Likert trial with custom labels."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
        )

        rating_config = RatingScaleConfig(
            min_value=1,
            max_value=5,
            min_label="Strongly disagree",
            max_label="Strongly agree",
        )

        trial = create_trial(
            item=item,
            experiment_config=config,
            trial_number=0,
            rating_config=rating_config,
        )

        assert "Strongly disagree" in trial["prompt"]
        assert "Strongly agree" in trial["prompt"]
        assert len(trial["choices"]) == 5


class TestSliderConfiguration:
    """Tests for slider trial configuration."""

    def test_require_movement(self) -> None:
        """Test slider trial with require_movement setting."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="slider_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
        )

        rating_config = RatingScaleConfig(required=True)

        trial = create_trial(
            item=item,
            experiment_config=config,
            trial_number=0,
            rating_config=rating_config,
        )

        assert trial["require_movement"] is True


class TestStimulusGeneration:
    """Tests for stimulus HTML generation."""

    def test_single_element(self) -> None:
        """Test stimulus HTML generation with single element."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"sentence": "The cat broke the vase."},
        )

        html = _generate_stimulus_html(item)

        assert "The cat broke the vase." in html
        assert "stimulus-container" in html

    def test_multiple_elements(self) -> None:
        """Test stimulus HTML generation with multiple elements."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "sentence1": "First sentence.",
                "sentence2": "Second sentence.",
            },
        )

        html = _generate_stimulus_html(item, include_all=True)

        assert "First sentence." in html
        assert "Second sentence." in html

    def test_first_element_only(self) -> None:
        """Test stimulus HTML generation with only first element."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "sentence": "Main sentence.",
                "choice_0": "Choice A",
                "choice_1": "Choice B",
            },
        )

        html = _generate_stimulus_html(item, include_all=False)

        assert html.count("<p>") == 1

    def test_empty_elements(self) -> None:
        """Test stimulus HTML generation with no elements."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={},
        )

        html = _generate_stimulus_html(item)

        assert "No stimulus available" in html


class TestSpecialTrials:
    """Tests for instruction, consent, and completion trials."""

    def test_instruction_trial(self) -> None:
        """Test instruction trial creation."""
        trial = create_instruction_trial("Please follow these instructions carefully.")

        assert trial["type"] == "html-keyboard-response"
        assert "Please follow these instructions carefully." in trial["stimulus"]
        assert trial["data"]["trial_type"] == "instructions"
        assert "Press any key" in trial["stimulus"]

    def test_consent_trial(self) -> None:
        """Test consent trial creation."""
        consent_text = "This study involves rating sentences."

        trial = create_consent_trial(consent_text)

        assert trial["type"] == "html-button-response"
        assert consent_text in trial["stimulus"]
        assert trial["choices"] == ["I agree", "I do not agree"]
        assert trial["data"]["trial_type"] == "consent"

    def test_completion_trial_default(self) -> None:
        """Test completion trial creation with default message."""
        trial = create_completion_trial()

        assert trial["type"] == "html-keyboard-response"
        assert "Thank you for participating!" in trial["stimulus"]
        assert trial["choices"] == "NO_KEYS"
        assert trial["data"]["trial_type"] == "completion"

    def test_completion_trial_custom_message(self) -> None:
        """Test completion trial with custom message."""
        custom_message = "Great job! Your responses have been recorded."

        trial = create_completion_trial(completion_message=custom_message)

        assert custom_message in trial["stimulus"]
