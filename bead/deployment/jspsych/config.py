"""Configuration models for jsPsych experiment generation.

This module provides Pydantic models for configuring jsPsych experiment
generation, including experiment types, UI settings, and display options.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Type alias for experiment types
type ExperimentType = Literal[
    "likert_rating",
    "slider_rating",
    "binary_choice",
    "forced_choice",
]

# Type alias for UI themes
type UITheme = Literal["light", "dark", "auto"]


class ExperimentConfig(BaseModel):
    """Configuration for jsPsych experiment generation.

    This model defines all configurable aspects of a jsPsych experiment,
    including the experiment type, UI settings, and trial presentation options.

    Attributes
    ----------
    experiment_type : ExperimentType
        Type of experiment (likert_rating, slider_rating, binary_choice, forced_choice)
    title : str
        Experiment title displayed to participants
    description : str
        Brief description of the experiment
    instructions : str
        Instructions shown to participants before the experiment
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

    Examples
    --------
    >>> config = ExperimentConfig(
    ...     experiment_type="likert_rating",
    ...     title="Sentence Acceptability Study",
    ...     description="Rate the acceptability of sentences",
    ...     instructions="Please rate each sentence on a scale from 1 to 7."
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
    instructions: str
    randomize_trial_order: bool = Field(default=True)
    show_progress_bar: bool = Field(default=True)
    ui_theme: UITheme = Field(default="light")
    on_finish_url: str | None = Field(default=None)
    allow_backwards: bool = Field(default=False)
    show_click_target: bool = Field(default=False)
    minimum_duration_ms: int = Field(default=0, ge=0)
    use_jatos: bool = Field(default=True)
    prolific_completion_code: str | None = Field(default=None)


class RatingScaleConfig(BaseModel):
    """Configuration for rating scale trials.

    Attributes
    ----------
    min_value : int
        Minimum value on the scale (default: 1)
    max_value : int
        Maximum value on the scale (default: 7)
    min_label : str
        Label for the minimum value (default: "Not at all")
    max_label : str
        Label for the maximum value (default: "Very much")
    step : int
        Step size between values (default: 1)
    show_numeric_labels : bool
        Whether to show numeric labels on the scale (default: True)
    required : bool
        Whether a response is required (default: True)
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
    )

    min_value: int = Field(default=1)
    max_value: int = Field(default=7)
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
