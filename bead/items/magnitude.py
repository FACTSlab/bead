"""Utilities for creating magnitude experimental items.

This module provides language-agnostic utilities for creating magnitude
items where participants enter numeric values (bounded or unbounded),
such as reading times, confidence ratings, or counts.

Integration Points
------------------
- Active Learning: bead/active_learning/models/magnitude.py
- Simulation: bead/simulation/strategies/magnitude.py
- Deployment: bead/deployment/jspsych/ (numeric input)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue


def create_magnitude_item(
    text: str,
    unit: str | None = None,
    bounds: tuple[int | float | None, int | float | None] = (None, None),
    prompt: str | None = None,
    step: int | float | None = None,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    *,
    text_element: str = "text",
    prompt_element: str = "prompt",
    unit_element: str = "unit",
) -> Item:
    """Create a magnitude (numeric input) item.

    Parameters
    ----------
    text : str
        The stimulus text or question.
    unit : str | None
        Optional unit for the value (e.g., "ms", "%", "count").
    bounds : tuple[int | float | None, int | float | None]
        Tuple of (min, max) bounds. None means unbounded in that direction.
        Default: (None, None) for fully unbounded.
    prompt : str | None
        Optional prompt for the numeric input.
        If None, uses "Enter a value:".
    step : int | float | None
        Optional step size for input validation (e.g., 1 for integers, 0.01 for
        hundredths).
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.
    text_element : str
        Name of the rendered element that holds the stimulus text
        (default: "text").
    prompt_element : str
        Name of the rendered element that holds the prompt
        (default: "prompt").
    unit_element : str
        Name of the rendered element that holds the unit, written only
        when a unit is given (default: "unit").

    Returns
    -------
    Item
        Magnitude item with the stimulus, the prompt, and any unit stored in
        rendered_elements under the declared element names.

    Raises
    ------
    ValueError
        If text is empty, if both bounds are provided and min >= max, or if
        two declared element names that are both written are the same.

    Examples
    --------
    >>> item = create_magnitude_item(
    ...     text="How long did it take to read this sentence?",
    ...     unit="ms",
    ...     bounds=(0, None),
    ...     prompt="Enter time in milliseconds:"
    ... )
    >>> item.rendered_elements["text"]
    'How long did it take to read this sentence?'
    >>> item.item_metadata["unit"]
    'ms'
    >>> item.item_metadata["min_value"]
    0
    >>> item.item_metadata["max_value"] is None
    True

    >>> # Confidence with bounded range
    >>> item = create_magnitude_item(
    ...     text="How confident are you in your answer?",
    ...     unit="%",
    ...     bounds=(0, 100),
    ...     step=1
    ... )
    >>> item.item_metadata["max_value"]
    100

    >>> # Declared element names
    >>> item = create_magnitude_item(
    ...     text="How long was the pause?",
    ...     unit="ms",
    ...     text_element="sentence",
    ...     unit_element="measure",
    ... )
    >>> sorted(item.rendered_elements)
    ['measure', 'prompt', 'sentence']
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    if text_element == prompt_element:
        raise ValueError("text_element and prompt_element must name different elements")

    if unit and unit_element in (text_element, prompt_element):
        raise ValueError(
            "unit_element must name an element other than text_element and "
            "prompt_element"
        )

    min_value, max_value = bounds

    # Validate bounds if both are provided
    if min_value is not None and max_value is not None:
        if min_value >= max_value:
            raise ValueError(
                f"min_value ({min_value}) must be less than max_value ({max_value})"
            )

    if item_template_id is None:
        item_template_id = uuid4()

    if prompt is None:
        prompt = "Enter a value:"

    rendered_elements: dict[str, str] = {
        text_element: text,
        prompt_element: prompt,
    }

    if unit:
        rendered_elements[unit_element] = unit

    # Build item metadata
    item_metadata: dict[str, MetadataValue] = {
        "min_value": min_value,
        "max_value": max_value,
    }

    if unit:
        item_metadata["unit"] = unit

    if step is not None:
        item_metadata["step"] = step

    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        rendered_elements=rendered_elements,
        item_metadata=item_metadata,
    )


def create_magnitude_items_from_texts(
    texts: list[str],
    unit: str | None = None,
    bounds: tuple[int | float | None, int | float | None] = (None, None),
    prompt: str | None = None,
    step: int | float | None = None,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: Callable[[str], dict[str, MetadataValue]] | None = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
    unit_element: str = "unit",
) -> list[Item]:
    """Create magnitude items from a list of texts.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    unit : str | None
        Optional unit for all items.
    bounds : tuple[int | float | None, int | float | None]
        Bounds (min, max) for all items.
    prompt : str | None
        The question/prompt for all items.
    step : int | float | None
        Step size for all items.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.
    metadata_fn : Callable[[str], dict[str, MetadataValue]] | None
        Function to generate metadata from each text.
    text_element : str
        Name of the rendered element that holds the stimulus text
        (default: "text").
    prompt_element : str
        Name of the rendered element that holds the prompt
        (default: "prompt").
    unit_element : str
        Name of the rendered element that holds the unit, written only
        when a unit is given (default: "unit").

    Returns
    -------
    list[Item]
        Magnitude items for each text.

    Examples
    --------
    >>> texts = ["Sentence 1", "Sentence 2", "Sentence 3"]
    >>> items = create_magnitude_items_from_texts(
    ...     texts,
    ...     unit="ms",
    ...     bounds=(0, None),
    ...     prompt="Reading time?",
    ...     metadata_fn=lambda t: {"text_length": len(t)}
    ... )
    >>> len(items)
    3
    >>> items[0].item_metadata["unit"]
    'ms'
    """
    magnitude_items: list[Item] = []

    for text in texts:
        item_metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            item_metadata = metadata_fn(text)

        item = create_magnitude_item(
            text=text,
            unit=unit,
            bounds=bounds,
            prompt=prompt,
            step=step,
            item_template_id=item_template_id,
            metadata=item_metadata,
            text_element=text_element,
            prompt_element=prompt_element,
            unit_element=unit_element,
        )
        magnitude_items.append(item)

    return magnitude_items


def create_magnitude_items_from_groups(
    items: list[Item],
    group_by: Callable[[Item], Any],
    unit: str | None = None,
    bounds: tuple[int | float | None, int | float | None] = (None, None),
    prompt: str | None = None,
    step: int | float | None = None,
    *,
    extract_text: Callable[[Item], str] | None = None,
    include_group_metadata: bool = True,
    item_template_id: UUID | None = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
    unit_element: str = "unit",
) -> list[Item]:
    """Create magnitude items from grouped source items.

    Groups items and creates one magnitude item per source item,
    preserving group information in metadata.

    Parameters
    ----------
    items : list[Item]
        Source items to process.
    group_by : Callable[[Item], Any]
        Function to extract grouping key from items.
    unit : str | None
        Optional unit for all items.
    bounds : tuple[int | float | None, int | float | None]
        Bounds (min, max) for all items.
    prompt : str | None
        The question/prompt for all items.
    step : int | float | None
        Step size for all items.
    extract_text : Callable[[Item], str] | None
        Function to extract text from item. If None, tries common keys.
    include_group_metadata : bool
        Whether to include group key in item metadata.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.
    text_element : str
        Name of the rendered element that holds the stimulus text
        (default: "text").
    prompt_element : str
        Name of the rendered element that holds the prompt
        (default: "prompt").
    unit_element : str
        Name of the rendered element that holds the unit, written only
        when a unit is given (default: "unit").

    Returns
    -------
    list[Item]
        Magnitude items from source items.

    Examples
    --------
    >>> source_items = [
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "The cat sat."},
    ...         item_metadata={"category": "simple"}
    ...     )
    ... ]
    >>> magnitude_items = create_magnitude_items_from_groups(
    ...     source_items,
    ...     group_by=lambda i: i.item_metadata["category"],
    ...     unit="ms",
    ...     bounds=(0, None),
    ...     prompt="Reading time:"
    ... )
    >>> len(magnitude_items)
    1
    """
    # Group items
    groups: dict[Any, list[Item]] = defaultdict(list)
    for item in items:
        group_key = group_by(item)
        groups[group_key].append(item)

    magnitude_items: list[Item] = []

    for group_key, group_items in groups.items():
        for item in group_items:
            # Extract text
            if extract_text:
                text: str = extract_text(item)
            else:
                text = _extract_text_from_item(item)

            # Build metadata
            item_metadata: dict[str, MetadataValue] = {
                "source_item_id": str(item.id),
            }
            if include_group_metadata:
                item_metadata["group_key"] = str(group_key)

            # Create magnitude item
            magnitude_item = create_magnitude_item(
                text=text,
                unit=unit,
                bounds=bounds,
                prompt=prompt,
                step=step,
                item_template_id=item_template_id,
                metadata=item_metadata,
                text_element=text_element,
                prompt_element=prompt_element,
                unit_element=unit_element,
            )
            magnitude_items.append(magnitude_item)

    return magnitude_items


def create_magnitude_items_cross_product(
    texts: list[str],
    prompts: list[str],
    unit: str | None = None,
    bounds: tuple[int | float | None, int | float | None] = (None, None),
    step: int | float | None = None,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: (Callable[[str, str], dict[str, MetadataValue]] | None) = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
    unit_element: str = "unit",
) -> list[Item]:
    """Create magnitude items from cross-product of texts and prompts.

    Useful when you want to apply multiple prompts to each text.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    prompts : list[str]
        List of prompts to apply.
    unit : str | None
        Optional unit for all items.
    bounds : tuple[int | float | None, int | float | None]
        Bounds (min, max) for all items.
    step : int | float | None
        Step size for all items.
    item_template_id : UUID | None
        Template ID for all created items.
    metadata_fn : Callable[[str, str], dict[str, MetadataValue]] | None
        Function to generate metadata from (text, prompt).
    text_element : str
        Name of the rendered element that holds the stimulus text
        (default: "text").
    prompt_element : str
        Name of the rendered element that holds the prompt
        (default: "prompt").
    unit_element : str
        Name of the rendered element that holds the unit, written only
        when a unit is given (default: "unit").

    Returns
    -------
    list[Item]
        Magnitude items from cross-product.

    Examples
    --------
    >>> texts = ["Sentence 1.", "Sentence 2."]
    >>> prompts = ["Reading time?", "Processing time?"]
    >>> items = create_magnitude_items_cross_product(
    ...     texts, prompts, unit="ms", bounds=(0, None)
    ... )
    >>> len(items)
    4
    """
    magnitude_items: list[Item] = []

    for text, prompt in product(texts, prompts):
        item_metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            item_metadata = metadata_fn(text, prompt)

        item = create_magnitude_item(
            text=text,
            unit=unit,
            bounds=bounds,
            prompt=prompt,
            step=step,
            item_template_id=item_template_id,
            metadata=item_metadata,
            text_element=text_element,
            prompt_element=prompt_element,
            unit_element=unit_element,
        )
        magnitude_items.append(item)

    return magnitude_items


def create_filtered_magnitude_items(
    items: list[Item],
    unit: str | None = None,
    bounds: tuple[int | float | None, int | float | None] = (None, None),
    prompt: str | None = None,
    step: int | float | None = None,
    *,
    item_filter: Callable[[Item], bool] | None = None,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
    unit_element: str = "unit",
) -> list[Item]:
    """Create magnitude items with filtering.

    Parameters
    ----------
    items : list[Item]
        Source items.
    unit : str | None
        Optional unit for all items.
    bounds : tuple[int | float | None, int | float | None]
        Bounds (min, max) for all items.
    prompt : str | None
        The question/prompt for all items.
    step : int | float | None
        Step size for all items.
    item_filter : Callable[[Item], bool] | None
        Filter individual items.
    extract_text : Callable[[Item], str] | None
        Text extraction function.
    item_template_id : UUID | None
        Template ID for created items.
    text_element : str
        Name of the rendered element that holds the stimulus text
        (default: "text").
    prompt_element : str
        Name of the rendered element that holds the prompt
        (default: "prompt").
    unit_element : str
        Name of the rendered element that holds the unit, written only
        when a unit is given (default: "unit").

    Returns
    -------
    list[Item]
        Filtered magnitude items.

    Examples
    --------
    >>> magnitude_items = create_filtered_magnitude_items(
    ...     items,
    ...     unit="ms",
    ...     bounds=(0, None),
    ...     prompt="Reading time:",
    ...     item_filter=lambda i: i.item_metadata.get("valid", True)
    ... )  # doctest: +SKIP
    """
    # Filter items
    filtered_items = items
    if item_filter:
        filtered_items = [item for item in items if item_filter(item)]

    magnitude_items: list[Item] = []

    for item in filtered_items:
        # Extract text
        if extract_text:
            text: str = extract_text(item)
        else:
            text = _extract_text_from_item(item)

        # Create magnitude item
        item_metadata: dict[str, MetadataValue] = {
            "source_item_id": str(item.id),
        }

        magnitude_item = create_magnitude_item(
            text=text,
            unit=unit,
            bounds=bounds,
            prompt=prompt,
            step=step,
            item_template_id=item_template_id,
            metadata=item_metadata,
            text_element=text_element,
            prompt_element=prompt_element,
            unit_element=unit_element,
        )
        magnitude_items.append(magnitude_item)

    return magnitude_items


def create_reading_time_item(
    text: str,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    *,
    text_element: str = "text",
    prompt_element: str = "prompt",
    unit_element: str = "unit",
) -> Item:
    """Create a reading time measurement item.

    Convenience function for reading time in milliseconds with a lower bound
    of 0 (no upper bound).

    Parameters
    ----------
    text : str
        The text to measure reading time for.
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.
    text_element : str
        Name of the rendered element that holds the stimulus text
        (default: "text").
    prompt_element : str
        Name of the rendered element that holds the prompt
        (default: "prompt").
    unit_element : str
        Name of the rendered element that holds the unit, written only
        when a unit is given (default: "unit").

    Returns
    -------
    Item
        Reading time magnitude item.

    Examples
    --------
    >>> item = create_reading_time_item("The cat sat on the mat.")
    >>> item.item_metadata["unit"]
    'ms'
    >>> item.item_metadata["min_value"]
    0
    """
    return create_magnitude_item(
        text,
        unit="ms",
        bounds=(0, None),
        prompt="How long did it take to read?",
        step=1,
        item_template_id=item_template_id,
        metadata=metadata,
        text_element=text_element,
        prompt_element=prompt_element,
        unit_element=unit_element,
    )


def create_confidence_item(
    text: str,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    *,
    text_element: str = "text",
    prompt_element: str = "prompt",
    unit_element: str = "unit",
) -> Item:
    """Create a confidence rating item (0-100%).

    Convenience function for confidence percentage with bounds (0, 100).

    Parameters
    ----------
    text : str
        The text or question to rate confidence for.
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.
    text_element : str
        Name of the rendered element that holds the stimulus text
        (default: "text").
    prompt_element : str
        Name of the rendered element that holds the prompt
        (default: "prompt").
    unit_element : str
        Name of the rendered element that holds the unit, written only
        when a unit is given (default: "unit").

    Returns
    -------
    Item
        Confidence magnitude item.

    Examples
    --------
    >>> item = create_confidence_item("Is this sentence grammatical?")
    >>> item.item_metadata["unit"]
    '%'
    >>> item.item_metadata["max_value"]
    100
    """
    return create_magnitude_item(
        text,
        unit="%",
        bounds=(0, 100),
        prompt="How confident are you?",
        step=1,
        item_template_id=item_template_id,
        metadata=metadata,
        text_element=text_element,
        prompt_element=prompt_element,
        unit_element=unit_element,
    )


def _extract_text_from_item(item: Item) -> str:
    """Extract text from item's rendered_elements.

    Tries common keys: "text", "sentence", "content".
    Raises error if no suitable text found.

    Parameters
    ----------
    item : Item
        Item to extract text from.

    Returns
    -------
    str
        Extracted text.

    Raises
    ------
    ValueError
        If no suitable text key found in rendered_elements.
    """
    for key in ["text", "sentence", "content"]:
        if key in item.rendered_elements:
            return item.rendered_elements[key]

    raise ValueError(
        f"Cannot extract text from item {item.id}. "
        f"Expected one of ['text', 'sentence', 'content'] in rendered_elements, "
        f"but found keys: {list(item.rendered_elements.keys())}. "
        f"Use the extract_text parameter to provide a custom extraction function."
    )
