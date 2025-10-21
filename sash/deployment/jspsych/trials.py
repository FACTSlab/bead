"""Trial generators for jsPsych experiments.

This module provides functions to generate jsPsych trial objects from
Item models. It supports various trial types including rating scales,
forced choice, and binary choice trials.
"""

from __future__ import annotations

from typing import Any

from sash.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from sash.items.models import Item


def create_trial(
    item: Item,
    experiment_config: ExperimentConfig,
    trial_number: int,
    rating_config: RatingScaleConfig | None = None,
    choice_config: ChoiceConfig | None = None,
) -> dict[str, Any]:
    """Create a jsPsych trial object from an Item.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    experiment_config : ExperimentConfig
        The experiment configuration.
    trial_number : int
        The trial number (for tracking).
    rating_config : RatingScaleConfig | None
        Configuration for rating scale trials (required for rating types).
    choice_config : ChoiceConfig | None
        Configuration for choice trials (required for choice types).

    Returns
    -------
    dict[str, Any]
        A jsPsych trial object.

    Raises
    ------
    ValueError
        If required configuration is missing for the experiment type.

    Examples
    --------
    >>> from uuid import UUID
    >>> item = Item(
    ...     item_template_id=UUID("12345678-1234-5678-1234-567812345678"),
    ...     rendered_elements={"sentence": "The cat broke the vase"}
    ... )
    >>> config = ExperimentConfig(
    ...     experiment_type="likert_rating",
    ...     title="Test",
    ...     description="Test",
    ...     instructions="Test"
    ... )
    >>> rating_config = RatingScaleConfig()
    >>> trial = create_trial(item, config, 0, rating_config=rating_config)
    >>> trial["type"]
    'html-slider-response'
    """
    if experiment_config.experiment_type == "likert_rating":
        if rating_config is None:
            raise ValueError("rating_config required for likert_rating experiments")
        return _create_likert_trial(item, rating_config, trial_number)
    elif experiment_config.experiment_type == "slider_rating":
        if rating_config is None:
            raise ValueError("rating_config required for slider_rating experiments")
        return _create_slider_trial(item, rating_config, trial_number)
    elif experiment_config.experiment_type == "binary_choice":
        if choice_config is None:
            raise ValueError("choice_config required for binary_choice experiments")
        return _create_binary_choice_trial(item, choice_config, trial_number)
    elif experiment_config.experiment_type == "forced_choice":
        if choice_config is None:
            raise ValueError("choice_config required for forced_choice experiments")
        return _create_forced_choice_trial(item, choice_config, trial_number)
    else:
        raise ValueError(
            f"Unknown experiment type: {experiment_config.experiment_type}"
        )


def _create_likert_trial(
    item: Item,
    config: RatingScaleConfig,
    trial_number: int,
) -> dict[str, Any]:
    """Create a Likert rating trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    config : RatingScaleConfig
        Rating scale configuration.
    trial_number : int
        The trial number.

    Returns
    -------
    dict[str, Any]
        A jsPsych html-button-response trial object.
    """
    # Generate stimulus HTML from rendered elements
    stimulus_html = _generate_stimulus_html(item)

    # Generate button labels for Likert scale
    labels: list[str] = []
    for i in range(config.min_value, config.max_value + 1, config.step):
        if config.show_numeric_labels:
            labels.append(str(i))
        else:
            labels.append("")

    prompt_html = (
        f'<p style="margin-top: 20px;">'
        f'<span style="float: left;">{config.min_label}</span>'
        f'<span style="float: right;">{config.max_label}</span>'
        f"</p>"
    )

    return {
        "type": "html-button-response",
        "stimulus": stimulus_html,
        "choices": labels,
        "prompt": prompt_html,
        "data": {
            "item_id": str(item.id),
            "trial_number": trial_number,
            "trial_type": "likert_rating",
            "item_metadata": item.item_metadata,
        },
        "button_html": '<button class="jspsych-btn likert-button">%choice%</button>',
    }


def _create_slider_trial(
    item: Item,
    config: RatingScaleConfig,
    trial_number: int,
) -> dict[str, Any]:
    """Create a slider rating trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    config : RatingScaleConfig
        Rating scale configuration.
    trial_number : int
        The trial number.

    Returns
    -------
    dict[str, Any]
        A jsPsych html-slider-response trial object.
    """
    stimulus_html = _generate_stimulus_html(item)

    return {
        "type": "html-slider-response",
        "stimulus": stimulus_html,
        "labels": [config.min_label, config.max_label],
        "min": config.min_value,
        "max": config.max_value,
        "step": config.step,
        "slider_start": (config.min_value + config.max_value) // 2,
        "require_movement": config.required,
        "data": {
            "item_id": str(item.id),
            "trial_number": trial_number,
            "trial_type": "slider_rating",
            "item_metadata": item.item_metadata,
        },
    }


def _create_binary_choice_trial(
    item: Item,
    config: ChoiceConfig,
    trial_number: int,
) -> dict[str, Any]:
    """Create a binary choice trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    config : ChoiceConfig
        Choice configuration.
    trial_number : int
        The trial number.

    Returns
    -------
    dict[str, Any]
        A jsPsych html-button-response trial object.
    """
    stimulus_html = _generate_stimulus_html(item)

    return {
        "type": "html-button-response",
        "stimulus": stimulus_html,
        "choices": ["Yes", "No"],
        "data": {
            "item_id": str(item.id),
            "trial_number": trial_number,
            "trial_type": "binary_choice",
            "item_metadata": item.item_metadata,
        },
        "button_html": config.button_html
        or '<button class="jspsych-btn">%choice%</button>',
    }


def _create_forced_choice_trial(
    item: Item,
    config: ChoiceConfig,
    trial_number: int,
) -> dict[str, Any]:
    """Create a forced choice trial.

    For forced choice trials, the item should have multiple rendered elements
    that represent the different choices. The choices are extracted from the
    rendered_elements and presented as buttons.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    config : ChoiceConfig
        Choice configuration.
    trial_number : int
        The trial number.

    Returns
    -------
    dict[str, Any]
        A jsPsych html-button-response trial object.
    """
    # For forced choice, we expect the item to have a primary stimulus
    # and multiple choice options in rendered_elements
    stimulus_html = _generate_stimulus_html(item, include_all=False)

    # Extract choices from rendered elements (excluding the main stimulus)
    # This assumes element names like "choice_0", "choice_1", etc.
    # or "option_a", "option_b", etc.
    choices = []
    choice_keys = sorted(
        [
            k
            for k in item.rendered_elements.keys()
            if k.startswith(("choice_", "option_"))
        ]
    )

    if choice_keys:
        choices = [item.rendered_elements[k] for k in choice_keys]
    else:
        # Fallback: use all elements except the first one as choices
        all_keys = sorted(item.rendered_elements.keys())
        if len(all_keys) > 1:
            choices = [item.rendered_elements[k] for k in all_keys[1:]]
        else:
            # No choices found, create generic yes/no
            choices = ["Choice A", "Choice B"]

    return {
        "type": "html-button-response",
        "stimulus": stimulus_html,
        "choices": choices,
        "data": {
            "item_id": str(item.id),
            "trial_number": trial_number,
            "trial_type": "forced_choice",
            "item_metadata": item.item_metadata,
        },
        "button_html": config.button_html
        or '<button class="jspsych-btn">%choice%</button>',
    }


def _generate_stimulus_html(item: Item, include_all: bool = True) -> str:
    """Generate HTML for stimulus presentation.

    Parameters
    ----------
    item : Item
        The item to generate HTML for.
    include_all : bool
        Whether to include all rendered elements (True) or just the first one (False).

    Returns
    -------
    str
        HTML string for the stimulus.
    """
    if not item.rendered_elements:
        return "<p>No stimulus available</p>"

    # Get rendered elements in a consistent order
    sorted_keys = sorted(item.rendered_elements.keys())

    if include_all:
        # Include all rendered elements
        elements = [
            f'<div class="stimulus-element"><p>{item.rendered_elements[k]}</p></div>'
            for k in sorted_keys
        ]
        return '<div class="stimulus-container">' + "".join(elements) + "</div>"
    else:
        # Include only the first element (for forced choice where others are options)
        first_key = sorted_keys[0]
        element_html = item.rendered_elements[first_key]
        return f'<div class="stimulus-container"><p>{element_html}</p></div>'


def create_instruction_trial(instructions: str) -> dict[str, Any]:
    """Create an instruction trial.

    Parameters
    ----------
    instructions : str
        The instruction text to display.

    Returns
    -------
    dict[str, Any]
        A jsPsych html-keyboard-response trial object.
    """
    stimulus_html = (
        f'<div class="instructions">'
        f"<h2>Instructions</h2>"
        f"<p>{instructions}</p>"
        f"<p><em>Press any key to continue</em></p>"
        f"</div>"
    )

    return {
        "type": "html-keyboard-response",
        "stimulus": stimulus_html,
        "data": {
            "trial_type": "instructions",
        },
    }


def create_consent_trial(consent_text: str) -> dict[str, Any]:
    """Create a consent trial.

    Parameters
    ----------
    consent_text : str
        The consent text to display.

    Returns
    -------
    dict[str, Any]
        A jsPsych html-button-response trial object.
    """
    stimulus_html = (
        f'<div class="consent"><h2>Consent</h2><div>{consent_text}</div></div>'
    )

    return {
        "type": "html-button-response",
        "stimulus": stimulus_html,
        "choices": ["I agree", "I do not agree"],
        "data": {
            "trial_type": "consent",
        },
    }


def create_completion_trial(
    completion_message: str = "Thank you for participating!",
) -> dict[str, Any]:
    """Create a completion trial.

    Parameters
    ----------
    completion_message : str
        The completion message to display.

    Returns
    -------
    dict[str, Any]
        A jsPsych html-keyboard-response trial object.
    """
    stimulus_html = (
        f'<div class="completion"><h2>Complete</h2><p>{completion_message}</p></div>'
    )

    return {
        "type": "html-keyboard-response",
        "stimulus": stimulus_html,
        "choices": "NO_KEYS",
        "data": {
            "trial_type": "completion",
        },
    }
