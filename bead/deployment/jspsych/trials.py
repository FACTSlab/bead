"""Trial generators for jsPsych experiments.

This module provides functions to generate jsPsych trial objects from
Item models. It supports various trial types including rating scales,
forced choice, and binary choice trials.
"""

from __future__ import annotations

from bead.data.base import JsonValue
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from bead.items.item import Item
from bead.items.item_template import ItemTemplate


def _serialize_item_metadata(
    item: Item, template: ItemTemplate
) -> dict[str, JsonValue]:
    """Serialize complete item and template metadata for trial data.

    Parameters
    ----------
    item : Item
        The item to serialize metadata from.
    template : ItemTemplate
        The item template to serialize metadata from.

    Returns
    -------
    dict[str, JsonValue]
        Metadata dictionary containing all item and template fields.
    """
    return {
        # Item identification
        "item_id": str(item.id),
        "item_created": item.created_at.isoformat(),
        "item_modified": item.modified_at.isoformat(),
        # Item template reference
        "item_template_id": str(item.item_template_id),
        # Filled template references
        "filled_template_refs": [str(ref) for ref in item.filled_template_refs],
        # Rendered elements
        "rendered_elements": dict(item.rendered_elements),
        # Unfilled slots (for cloze tasks)
        "unfilled_slots": [
            {
                "slot_name": slot.slot_name,
                "position": slot.position,
                "constraint_ids": [str(cid) for cid in slot.constraint_ids],
            }
            for slot in item.unfilled_slots
        ],
        # Model outputs
        "model_outputs": [
            {
                "model_name": output.model_name,
                "model_version": output.model_version,
                "operation": output.operation,
                "inputs": output.inputs,
                "output": output.output,
                "cache_key": output.cache_key,
                "computation_metadata": output.computation_metadata,
            }
            for output in item.model_outputs
        ],
        # Constraint satisfaction
        "constraint_satisfaction": {
            str(k): v for k, v in item.constraint_satisfaction.items()
        },
        # Item-specific metadata
        "item_metadata": dict(item.item_metadata),
        # Template information
        "template_name": template.name,
        "template_description": template.description,
        "judgment_type": template.judgment_type,
        "task_type": template.task_type,
        # Template elements
        "template_elements": [
            {
                "element_type": elem.element_type,
                "element_name": elem.element_name,
                "content": elem.content,
                "filled_template_ref_id": (
                    str(elem.filled_template_ref_id)
                    if elem.filled_template_ref_id
                    else None
                ),
                "element_metadata": elem.element_metadata,
                "order": elem.order,
            }
            for elem in template.elements
        ],
        # Template constraints
        "template_constraints": [str(c) for c in template.constraints],
        # Task specification
        "task_spec": {
            "prompt": template.task_spec.prompt,
            "scale_bounds": template.task_spec.scale_bounds,
            "scale_labels": template.task_spec.scale_labels,
            "options": template.task_spec.options,
            "min_selections": template.task_spec.min_selections,
            "max_selections": template.task_spec.max_selections,
            "text_validation_pattern": template.task_spec.text_validation_pattern,
            "max_length": template.task_spec.max_length,
        },
        # Presentation specification
        "presentation_spec": {
            "mode": template.presentation_spec.mode,
            "chunking": (
                {
                    "unit": template.presentation_spec.chunking.unit,
                    "parse_type": (template.presentation_spec.chunking.parse_type),
                    "constituent_labels": (
                        template.presentation_spec.chunking.constituent_labels
                    ),
                    "parser": template.presentation_spec.chunking.parser,
                    "parse_language": (
                        template.presentation_spec.chunking.parse_language
                    ),
                    "custom_boundaries": (
                        template.presentation_spec.chunking.custom_boundaries
                    ),
                }
                if template.presentation_spec.chunking
                else None
            ),
            "timing": (
                {
                    "duration_ms": template.presentation_spec.timing.duration_ms,
                    "isi_ms": template.presentation_spec.timing.isi_ms,
                    "timeout_ms": template.presentation_spec.timing.timeout_ms,
                    "mask_char": template.presentation_spec.timing.mask_char,
                    "cumulative": template.presentation_spec.timing.cumulative,
                }
                if template.presentation_spec.timing
                else None
            ),
            "display_format": template.presentation_spec.display_format,
        },
        # Presentation order
        "presentation_order": template.presentation_order,
        # Template metadata
        "template_metadata": dict(template.template_metadata),
    }


def create_trial(
    item: Item,
    template: ItemTemplate,
    experiment_config: ExperimentConfig,
    trial_number: int,
    rating_config: RatingScaleConfig | None = None,
    choice_config: ChoiceConfig | None = None,
) -> dict[str, JsonValue]:
    """Create a jsPsych trial object from an Item.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template for this item.
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
    dict[str, JsonValue]
        A jsPsych trial object with item and template metadata.

    Raises
    ------
    ValueError
        If required configuration is missing for the experiment type.

    Examples
    --------
    >>> from uuid import UUID
    >>> from bead.items.item_template import TaskSpec, PresentationSpec
    >>> item = Item(
    ...     item_template_id=UUID("12345678-1234-5678-1234-567812345678"),
    ...     rendered_elements={"sentence": "The cat broke the vase"}
    ... )
    >>> template = ItemTemplate(
    ...     name="test",
    ...     judgment_type="acceptability",
    ...     task_type="ordinal_scale",
    ...     task_spec=TaskSpec(prompt="Rate this"),
    ...     presentation_spec=PresentationSpec(mode="static")
    ... )
    >>> config = ExperimentConfig(
    ...     experiment_type="likert_rating",
    ...     title="Test",
    ...     description="Test",
    ...     instructions="Test"
    ... )
    >>> rating_config = RatingScaleConfig()
    >>> trial = create_trial(item, template, config, 0, rating_config=rating_config)
    >>> trial["type"]
    'html-slider-response'
    """
    if experiment_config.experiment_type == "likert_rating":
        if rating_config is None:
            raise ValueError("rating_config required for likert_rating experiments")
        return _create_likert_trial(item, template, rating_config, trial_number)
    elif experiment_config.experiment_type == "slider_rating":
        if rating_config is None:
            raise ValueError("rating_config required for slider_rating experiments")
        return _create_slider_trial(item, template, rating_config, trial_number)
    elif experiment_config.experiment_type == "binary_choice":
        if choice_config is None:
            raise ValueError("choice_config required for binary_choice experiments")
        return _create_binary_choice_trial(item, template, choice_config, trial_number)
    elif experiment_config.experiment_type == "forced_choice":
        if choice_config is None:
            raise ValueError("choice_config required for forced_choice experiments")
        return _create_forced_choice_trial(item, template, choice_config, trial_number)
    else:
        raise ValueError(
            f"Unknown experiment type: {experiment_config.experiment_type}"
        )


def _create_likert_trial(
    item: Item,
    template: ItemTemplate,
    config: RatingScaleConfig,
    trial_number: int,
) -> dict[str, JsonValue]:
    """Create a Likert rating trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template.
    config : RatingScaleConfig
        Rating scale configuration.
    trial_number : int
        The trial number.

    Returns
    -------
    dict[str, JsonValue]
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

    # Serialize complete metadata
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "likert_rating"

    return {
        "type": "html-button-response",
        "stimulus": stimulus_html,
        "choices": labels,
        "prompt": prompt_html,
        "data": metadata,
        "button_html": '<button class="jspsych-btn likert-button">%choice%</button>',
    }


def _create_slider_trial(
    item: Item,
    template: ItemTemplate,
    config: RatingScaleConfig,
    trial_number: int,
) -> dict[str, JsonValue]:
    """Create a slider rating trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template.
    config : RatingScaleConfig
        Rating scale configuration.
    trial_number : int
        The trial number.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych html-slider-response trial object.
    """
    stimulus_html = _generate_stimulus_html(item)

    # Serialize complete metadata
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "slider_rating"

    return {
        "type": "html-slider-response",
        "stimulus": stimulus_html,
        "labels": [config.min_label, config.max_label],
        "min": config.min_value,
        "max": config.max_value,
        "step": config.step,
        "slider_start": (config.min_value + config.max_value) // 2,
        "require_movement": config.required,
        "data": metadata,
    }


def _create_binary_choice_trial(
    item: Item,
    template: ItemTemplate,
    config: ChoiceConfig,
    trial_number: int,
) -> dict[str, JsonValue]:
    """Create a binary choice trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template.
    config : ChoiceConfig
        Choice configuration.
    trial_number : int
        The trial number.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych html-button-response trial object.
    """
    stimulus_html = _generate_stimulus_html(item)

    # Serialize complete metadata
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "binary_choice"

    return {
        "type": "html-button-response",
        "stimulus": stimulus_html,
        "choices": ["Yes", "No"],
        "data": metadata,
        "button_html": config.button_html
        or '<button class="jspsych-btn">%choice%</button>',
    }


def _create_forced_choice_trial(
    item: Item,
    template: ItemTemplate,
    config: ChoiceConfig,
    trial_number: int,
) -> dict[str, JsonValue]:
    """Create a forced choice trial.

    For forced choice trials, the item should have multiple rendered elements
    that represent the different choices. The choices are extracted from the
    rendered_elements and presented as buttons.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template.
    config : ChoiceConfig
        Choice configuration.
    trial_number : int
        The trial number.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych html-button-response trial object.
    """
    # For forced choice, use the prompt from the template as the stimulus
    # (not the choices themselves)
    prompt = (
        template.task_spec.prompt
        if template.task_spec
        else "Which option do you choose?"
    )
    stimulus_html = (
        f'<div class="stimulus-container"><p class="prompt">{prompt}</p></div>'
    )

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

    # Serialize complete metadata
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "forced_choice"

    return {
        "type": "html-button-response",
        "stimulus": stimulus_html,
        "choices": choices,
        "data": metadata,
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


def create_instruction_trial(instructions: str) -> dict[str, JsonValue]:
    """Create an instruction trial.

    Parameters
    ----------
    instructions : str
        The instruction text to display.

    Returns
    -------
    dict[str, JsonValue]
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


def create_consent_trial(consent_text: str) -> dict[str, JsonValue]:
    """Create a consent trial.

    Parameters
    ----------
    consent_text : str
        The consent text to display.

    Returns
    -------
    dict[str, JsonValue]
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
) -> dict[str, JsonValue]:
    """Create a completion trial.

    Parameters
    ----------
    completion_message : str
        The completion message to display.

    Returns
    -------
    dict[str, JsonValue]
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
