"""Tests for trial generation."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from bead.data.range import Range
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    DemographicsConfig,
    DemographicsFieldConfig,
    ExperimentConfig,
    InstructionPage,
    InstructionsConfig,
    RatingScaleConfig,
)
from bead.deployment.jspsych.trials import (
    _generate_stimulus_html,
    create_completion_trial,
    create_consent_trial,
    create_demographics_trial,
    create_instructions_trial,
    create_trial,
)
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec


class TestCreateTrial:
    """Tests for create_trial() with different experiment types."""

    def test_likert_rating(
        self,
        sample_item: Item,
        sample_item_template: ItemTemplate,
        sample_experiment_config: ExperimentConfig,
        sample_rating_config: RatingScaleConfig,
    ) -> None:
        """Test Likert rating trial creation."""
        trial = create_trial(
            item=sample_item,
            template=sample_item_template,
            experiment_config=sample_experiment_config,
            trial_number=0,
            rating_config=sample_rating_config,
        )

        assert trial["type"] == "html-button-response"
        assert len(trial["choices"]) == 7
        assert trial["data"]["item_id"] == str(sample_item.id)
        assert trial["data"]["trial_type"] == "likert_rating"

    def test_slider_rating(
        self, sample_item: Item, sample_item_template: ItemTemplate
    ) -> None:
        """Test slider rating trial creation."""
        config = ExperimentConfig(
            experiment_type="slider_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )
        rating_config = RatingScaleConfig(scale=Range[int](min=1, max=7))

        trial = create_trial(
            item=sample_item,
            template=sample_item_template,
            experiment_config=config,
            trial_number=0,
            rating_config=rating_config,
        )

        assert trial["type"] == "html-slider-response"
        assert trial["min"] == 1
        assert trial["max"] == 7
        assert trial["data"]["trial_type"] == "slider_rating"

    def test_binary_choice(
        self, sample_item: Item, sample_item_template: ItemTemplate
    ) -> None:
        """Test binary choice trial creation."""
        config = ExperimentConfig(
            experiment_type="binary_choice",
            title="Test",
            description="Test",
            instructions="Test instructions",
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )
        choice_config = ChoiceConfig()

        trial = create_trial(
            item=sample_item,
            template=sample_item_template,
            experiment_config=config,
            trial_number=0,
            choice_config=choice_config,
        )

        assert trial["type"] == "html-button-response"
        assert trial["choices"] == ["Yes", "No"]
        assert trial["data"]["trial_type"] == "binary_choice"

    def test_forced_choice(self) -> None:
        """Test forced choice trial creation."""
        template = ItemTemplate(
            name="test_template",
            description="Test item template",
            judgment_type="preference",
            task_type="forced_choice",
            task_spec=TaskSpec(
                prompt="Which is more natural?",
            ),
            presentation_spec=PresentationSpec(mode="static"),
        )

        item = Item(
            item_template_id=template.id,
            options=[
                "The cat broke the vase.",
                "The vase was broken by the cat.",
            ],
        )

        config = ExperimentConfig(
            experiment_type="forced_choice",
            title="Test",
            description="Test",
            instructions="Test instructions",
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )
        choice_config = ChoiceConfig()

        trial = create_trial(
            item=item,
            template=template,
            experiment_config=config,
            trial_number=0,
            choice_config=choice_config,
        )

        assert trial["type"] == "html-button-response"
        assert len(trial["choices"]) == 2
        assert trial["data"]["trial_type"] == "forced_choice"

    def test_missing_config_raises_error(self) -> None:
        """Test trial creation with missing required config."""
        template = ItemTemplate(
            name="test_template",
            description="Test item template",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(
                prompt="How natural is this sentence?",
                scale_bounds=(1, 7),
            ),
            presentation_spec=PresentationSpec(mode="static"),
        )

        item = Item(
            item_template_id=template.id,
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        with pytest.raises(ValueError, match="rating_config required"):
            create_trial(
                item=item,
                template=template,
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
        with pytest.raises(ValidationError):
            ExperimentConfig(
                experiment_type="invalid_type",  # type: ignore
                title="Test",
                description="Test",
                instructions="Test instructions",
                distribution_strategy=ListDistributionStrategy(
                    strategy_type=DistributionStrategyType.BALANCED
                ),
            )

    def test_metadata_inclusion(
        self, sample_item: Item, sample_item_template: ItemTemplate
    ) -> None:
        """Test that item metadata is included in trial data."""
        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )
        rating_config = RatingScaleConfig()

        trial = create_trial(
            item=sample_item,
            template=sample_item_template,
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
        template = ItemTemplate(
            name="test_template",
            description="Test item template",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(
                prompt="How natural is this sentence?",
                scale_bounds=(1, 5),
            ),
            presentation_spec=PresentationSpec(mode="static"),
        )

        item = Item(
            item_template_id=template.id,
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        rating_config = RatingScaleConfig(
            scale=Range[int](min=1, max=5),
            min_label="Strongly disagree",
            max_label="Strongly agree",
        )

        trial = create_trial(
            item=item,
            template=template,
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
        template = ItemTemplate(
            name="test_template",
            description="Test item template",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(
                prompt="How natural is this sentence?",
                scale_bounds=(1, 7),
            ),
            presentation_spec=PresentationSpec(mode="static"),
        )

        item = Item(
            item_template_id=template.id,
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="slider_rating",
            title="Test",
            description="Test",
            instructions="Test instructions",
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        rating_config = RatingScaleConfig(required=True)

        trial = create_trial(
            item=item,
            template=template,
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

    def test_instruction_trial_simple_string(self) -> None:
        """Test instruction trial creation with simple string."""
        trial = create_instructions_trial("Please follow these instructions carefully.")

        assert trial["type"] == "html-keyboard-response"
        assert "Please follow these instructions carefully." in trial["stimulus"]
        assert trial["data"]["trial_type"] == "instructions"
        assert "Press any key" in trial["stimulus"]

    def test_instruction_trial_multi_page(self) -> None:
        """Test instruction trial creation with multi-page config."""
        config = InstructionsConfig(
            pages=[
                InstructionPage(
                    title="Welcome", content="<p>Welcome to the study!</p>"
                ),
                InstructionPage(
                    title="Task", content="<p>Your task is to rate sentences.</p>"
                ),
            ],
            allow_backwards=True,
            button_label_next="Continue",
            button_label_finish="Start Experiment",
        )
        trial = create_instructions_trial(config)

        assert trial["type"] == "instructions"
        assert len(trial["pages"]) == 2
        assert trial["allow_backward"] is True
        assert trial["button_label_next"] == "Continue"
        assert trial["button_label_finish"] == "Start Experiment"
        assert trial["data"]["trial_type"] == "instructions"

    def test_demographics_trial(self) -> None:
        """Test demographics trial creation."""
        config = DemographicsConfig(
            enabled=True,
            title="About You",
            fields=[
                DemographicsFieldConfig(
                    name="age",
                    field_type="number",
                    label="Your Age",
                    required=True,
                ),
                DemographicsFieldConfig(
                    name="education",
                    field_type="dropdown",
                    label="Education Level",
                    options=["High School", "Bachelors", "Masters", "PhD"],
                ),
            ],
            submit_button_text="Next",
        )
        trial = create_demographics_trial(config)

        assert trial["type"] == "survey"
        assert trial["title"] == "About You"
        assert trial["button_label_finish"] == "Next"
        assert trial["data"]["trial_type"] == "demographics"

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
