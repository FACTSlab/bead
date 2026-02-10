"""Configuration models for jsPsych experiment generation.

This module provides Pydantic models for configuring jsPsych experiment
generation, including experiment types, UI settings, and display options.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bead.config.deployment import SlopitIntegrationConfig
from bead.data.range import Range
from bead.deployment.distribution import ListDistributionStrategy

# Type alias for experiment types
type ExperimentType = Literal[
    "likert_rating",
    "slider_rating",
    "binary_choice",
    "forced_choice",
    "span_labeling",
]

# Type alias for UI themes
type UITheme = Literal["light", "dark", "auto"]


# Factory functions for default lists
def _empty_demographics_fields() -> list[DemographicsFieldConfig]:
    """Return empty demographics field list."""
    return []


def _empty_instruction_pages() -> list[InstructionPage]:
    """Return empty instruction pages list."""
    return []


def _default_span_color_palette() -> list[str]:
    """Return default span highlight color palette."""
    return [
        "#BBDEFB",
        "#C8E6C9",
        "#FFE0B2",
        "#F8BBD0",
        "#D1C4E9",
        "#B2EBF2",
        "#DCEDC8",
        "#FFD54F",
    ]


def _default_span_dark_palette() -> list[str]:
    """Return default dark color palette for span subscript badges."""
    return [
        "#1565C0",
        "#2E7D32",
        "#E65100",
        "#AD1457",
        "#4527A0",
        "#00838F",
        "#558B2F",
        "#F9A825",
    ]


class SpanDisplayConfig(BaseModel):
    """Visual configuration for span rendering in experiments.

    Attributes
    ----------
    highlight_style : Literal["background", "underline", "border"]
        How to visually indicate spans.
    color_palette : list[str]
        CSS color values for span highlighting (light backgrounds).
    dark_color_palette : list[str]
        CSS color values for subscript label badges (dark, index-aligned
        with color_palette).
    show_labels : bool
        Whether to show span labels inline.
    show_tooltips : bool
        Whether to show tooltips on hover.
    token_delimiter : str
        Delimiter between tokens in display.
    label_position : Literal["inline", "below", "tooltip"]
        Where to display span labels.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    highlight_style: Literal["background", "underline", "border"] = "background"
    color_palette: list[str] = Field(
        default_factory=_default_span_color_palette
    )
    dark_color_palette: list[str] = Field(
        default_factory=_default_span_dark_palette
    )
    show_labels: bool = True
    show_tooltips: bool = True
    token_delimiter: str = " "
    label_position: Literal["inline", "below", "tooltip"] = "inline"


class DemographicsFieldConfig(BaseModel):
    """Configuration for a single demographics form field.

    Used to configure fields in a demographics form that appears before
    the experiment instructions. Supports various input types including
    text, number, dropdown, radio buttons, and checkboxes.

    Attributes
    ----------
    name : str
        Field name (used as key in collected data).
    field_type : Literal["text", "number", "dropdown", "radio", "checkbox"]
        Type of form input.
    label : str
        Display label for the field.
    required : bool
        Whether this field is required (default: False).
    options : list[str] | None
        Options for dropdown/radio fields (default: None).
    range : Range[int] | Range[float] | None
        Numeric range constraint for number fields (default: None).
    placeholder : str | None
        Placeholder text for text/number inputs (default: None).
    help_text : str | None
        Help text displayed below the field (default: None).

    Examples
    --------
    >>> age_field = DemographicsFieldConfig(
    ...     name="age",
    ...     field_type="number",
    ...     label="Your Age",
    ...     required=True,
    ...     range=Range[int](min=18, max=100),
    ... )
    >>> education_field = DemographicsFieldConfig(
    ...     name="education",
    ...     field_type="dropdown",
    ...     label="Highest Education Level",
    ...     required=True,
    ...     options=["High School", "Bachelor's", "Master's", "PhD"],
    ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    field_type: Literal["text", "number", "dropdown", "radio", "checkbox"]
    label: str
    required: bool = False
    options: list[str] | None = None
    range: Range[int] | Range[float] | None = None
    placeholder: str | None = None
    help_text: str | None = None


class DemographicsConfig(BaseModel):
    """Configuration for participant demographics form.

    Defines a demographics form that appears before experiment instructions.
    When enabled, participants must complete this form before proceeding.

    Attributes
    ----------
    enabled : bool
        Whether to show the demographics form (default: False).
    title : str
        Title displayed at the top of the form (default: "Participant Information").
    fields : list[DemographicsFieldConfig]
        List of fields to include in the form.
    submit_button_text : str
        Text for the submit button (default: "Continue").

    Examples
    --------
    >>> config = DemographicsConfig(
    ...     enabled=True,
    ...     title="About You",
    ...     fields=[
    ...         DemographicsFieldConfig(
    ...             name="age",
    ...             field_type="number",
    ...             label="Age",
    ...             required=True,
    ...         ),
    ...     ],
    ... )
    >>> config.enabled
    True
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    title: str = "Participant Information"
    fields: list[DemographicsFieldConfig] = Field(
        default_factory=_empty_demographics_fields
    )
    submit_button_text: str = "Continue"


class InstructionPage(BaseModel):
    """A single instruction page for multi-page instructions.

    Attributes
    ----------
    content : str
        HTML content for this page.
    title : str | None
        Optional title for this page (displayed above content).

    Examples
    --------
    >>> page = InstructionPage(
    ...     title="Welcome",
    ...     content="<p>Thank you for participating in this study.</p>",
    ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    title: str | None = None


class InstructionsConfig(BaseModel):
    """Configuration for multi-page experiment instructions.

    Allows creating rich, multi-page instructions with navigation controls.
    Participants can optionally navigate backwards through pages.

    Attributes
    ----------
    pages : list[InstructionPage]
        List of instruction pages to display.
    show_page_numbers : bool
        Whether to show page numbers (default: True).
    allow_backwards : bool
        Whether to allow navigating to previous pages (default: True).
    button_label_next : str
        Label for the next button (default: "Next").
    button_label_finish : str
        Label for the final button (default: "Begin Experiment").

    Examples
    --------
    >>> config = InstructionsConfig(
    ...     pages=[
    ...         InstructionPage(title="Welcome", content="<p>Welcome!</p>"),
    ...         InstructionPage(title="Task", content="<p>Your task is...</p>"),
    ...     ],
    ...     allow_backwards=True,
    ... )
    >>> len(config.pages)
    2

    >>> # Create from plain text (single page)
    >>> config = InstructionsConfig.from_text("Please rate each sentence.")
    >>> len(config.pages)
    1
    """

    model_config = ConfigDict(extra="forbid")

    pages: list[InstructionPage] = Field(default_factory=_empty_instruction_pages)
    show_page_numbers: bool = True
    allow_backwards: bool = True
    button_label_next: str = "Next"
    button_label_finish: str = "Begin Experiment"

    @classmethod
    def from_text(cls, text: str) -> InstructionsConfig:
        """Create single-page instructions from plain text.

        Provides backward compatibility with simple string instructions.

        Parameters
        ----------
        text : str
            Plain text or HTML content for a single instruction page.

        Returns
        -------
        InstructionsConfig
            Instructions config with a single page.

        Examples
        --------
        >>> config = InstructionsConfig.from_text("Rate each item from 1-7.")
        >>> config.pages[0].content
        'Rate each item from 1-7.'
        """
        return cls(pages=[InstructionPage(content=text)])


class ExperimentConfig(BaseModel):
    """Configuration for jsPsych experiment generation.

    Defines all configurable aspects of a jsPsych experiment, including experiment
    type, UI settings, trial presentation options, and list distribution strategy.

    Attributes
    ----------
    experiment_type : ExperimentType
        Type of experiment (likert_rating, slider_rating, binary_choice, forced_choice)
    title : str
        Experiment title displayed to participants
    description : str
        Brief description of the experiment
    instructions : str | InstructionsConfig
        Instructions shown to participants before the experiment. Can be a simple
        string (single page) or InstructionsConfig for multi-page instructions.
    demographics : DemographicsConfig | None
        Optional demographics form shown before instructions (default: None).
        When provided and enabled, participants must complete this form first.
    distribution_strategy : ListDistributionStrategy
        List distribution strategy for batch mode (required, no default).
        Specifies how participants are assigned to experiment lists using JATOS
        batch sessions. See bead.deployment.distribution for available strategies.
    randomize_trial_order : bool
        Whether to randomize trial order (default: True)
    show_progress_bar : bool
        Whether to show a progress bar during the experiment (default: True)
    ui_theme : UITheme
        UI theme for the experiment (light, dark, auto; default: light)
    on_finish_url : str | None
        URL to redirect to after experiment completion (default: None)
        If prolific_completion_code is set, this will be auto-generated
    allow_backwards : bool
        Whether participants can go back to previous trials (default: False)
    show_click_target : bool
        Whether to show click target for accuracy tracking (default: False)
    minimum_duration_ms : int
        Minimum trial duration in milliseconds (default: 0)
    use_jatos : bool
        Whether to enable JATOS integration (default: True)
    prolific_completion_code : str | None
        Prolific completion code for automatic redirect URL generation (default: None)
        When set, on_finish_url will be auto-generated as:
        https://app.prolific.co/submissions/complete?cc=<code>
    slopit : SlopitIntegrationConfig
        Slopit behavioral capture integration configuration (default: disabled).
        When enabled, captures keystroke dynamics, focus patterns, and paste events
        during experiment trials for AI-assisted response detection.

    Examples
    --------
    >>> from bead.deployment.distribution import (
    ...     ListDistributionStrategy,
    ...     DistributionStrategyType
    ... )
    >>> strategy = ListDistributionStrategy(
    ...     strategy_type=DistributionStrategyType.BALANCED,
    ...     max_participants=100
    ... )
    >>> config = ExperimentConfig(
    ...     experiment_type="likert_rating",
    ...     title="Sentence Acceptability Study",
    ...     description="Rate the acceptability of sentences",
    ...     instructions="Please rate each sentence on a scale from 1 to 7.",
    ...     distribution_strategy=strategy
    ... )
    >>> config.randomize_trial_order
    True
    >>> config.ui_theme
    'light'
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
    )

    experiment_type: ExperimentType
    title: str
    description: str
    instructions: str | InstructionsConfig
    distribution_strategy: ListDistributionStrategy
    demographics: DemographicsConfig | None = Field(
        default=None,
        description="Demographics form shown before instructions",
    )
    randomize_trial_order: bool = Field(default=True)
    show_progress_bar: bool = Field(default=True)
    ui_theme: UITheme = Field(default="light")
    on_finish_url: str | None = Field(default=None)
    allow_backwards: bool = Field(default=False)
    show_click_target: bool = Field(default=False)
    minimum_duration_ms: int = Field(default=0, ge=0)
    use_jatos: bool = Field(default=True)
    prolific_completion_code: str | None = Field(default=None)
    slopit: SlopitIntegrationConfig = Field(
        default_factory=SlopitIntegrationConfig,
        description="Slopit behavioral capture integration (opt-in, disabled)",
    )
    span_display: SpanDisplayConfig | None = Field(
        default=None,
        description="Span display config (auto-enabled when items have spans)",
    )


class RatingScaleConfig(BaseModel):
    """Configuration for rating scale trials.

    Attributes
    ----------
    scale
        Numeric range for the rating scale with min and max values.
        Default is Range(min=1, max=7) for a standard 7-point Likert scale.
    min_label
        Label for the minimum value (default: "Not at all").
    max_label
        Label for the maximum value (default: "Very much").
    step
        Step size between values (default: 1).
    show_numeric_labels
        Whether to show numeric labels on the scale (default: True).
    required
        Whether a response is required (default: True).

    Examples
    --------
    >>> config = RatingScaleConfig()
    >>> config.scale.min
    1
    >>> config.scale.max
    7
    >>> config.scale.contains(4)
    True

    >>> # Custom 5-point scale
    >>> config = RatingScaleConfig(scale=Range[int](min=1, max=5))
    >>> config.scale.max
    5
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
    )

    scale: Range[int] = Field(
        default_factory=lambda: Range[int](min=1, max=7),
        description="Numeric range for the rating scale",
    )
    min_label: str = Field(default="Not at all")
    max_label: str = Field(default="Very much")
    step: int = Field(default=1, ge=1)
    show_numeric_labels: bool = Field(default=True)
    required: bool = Field(default=True)


class ChoiceConfig(BaseModel):
    """Configuration for choice trials.

    Attributes
    ----------
    button_html : str | None
        Custom HTML for choice buttons (default: None)
    required : bool
        Whether a response is required (default: True)
    randomize_choice_order : bool
        Whether to randomize the order of choices (default: False)
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
    )

    button_html: str | None = Field(default=None)
    required: bool = Field(default=True)
    randomize_choice_order: bool = Field(default=False)
    layout: Literal["horizontal", "vertical"] = Field(default="horizontal")
