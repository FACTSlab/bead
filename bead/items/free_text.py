"""Utilities for creating free text experimental items.

This module provides language-agnostic utilities for creating free text
items where participants provide open-ended text responses (e.g., paraphrasing,
question answering, cloze completion).

Integration Points
------------------
- Active Learning: bead/active_learning/models/free_text.py
- Simulation: bead/simulation/strategies/free_text.py
- Deployment: bead/deployment/jspsych/ (text input or textarea)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue


def create_free_text_item(
    text: str,
    prompt: str,
    max_length: int | None = None,
    validation_pattern: str | None = None,
    min_length: int | None = None,
    multiline: bool = False,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    *,
    text_element: str = "text",
    prompt_element: str = "prompt",
) -> Item:
    """Create a free text (open-ended) item.

    Parameters
    ----------
    text : str
        The stimulus text or context.
    prompt : str
        The question/instruction for what to enter (required).
    max_length : int | None
        Maximum character limit. None means unlimited.
    validation_pattern : str | None
        Optional regex pattern for validation (validated at deployment).
    min_length : int | None
        Minimum characters required. None means no minimum.
    multiline : bool
        True for textarea (multiline), False for single-line input (default).
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

    Returns
    -------
    Item
        Free text item with the stimulus and the prompt stored in
        rendered_elements under the declared element names.

    Raises
    ------
    ValueError
        If text or prompt is empty, if min_length > max_length, or if
        text_element and prompt_element are the same.

    Examples
    --------
    >>> item = create_free_text_item(
    ...     text="The dog chased the cat.",
    ...     prompt="Who chased whom?",
    ...     max_length=100
    ... )
    >>> item.rendered_elements["text"]
    'The dog chased the cat.'
    >>> item.rendered_elements["prompt"]
    'Who chased whom?'
    >>> item.item_metadata["max_length"]
    100

    >>> # Multiline paraphrase task
    >>> item = create_free_text_item(
    ...     text="The quick brown fox jumps over the lazy dog.",
    ...     prompt="Rewrite this sentence in your own words:",
    ...     multiline=True,
    ...     max_length=200
    ... )
    >>> item.item_metadata["multiline"]
    True

    >>> # Declared element names
    >>> item = create_free_text_item(
    ...     text="The dog chased the cat.",
    ...     prompt="Who chased whom?",
    ...     text_element="passage",
    ...     prompt_element="question",
    ... )
    >>> item.rendered_elements["question"]
    'Who chased whom?'
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    if not prompt or not prompt.strip():
        raise ValueError("prompt is required for free text items")

    if text_element == prompt_element:
        raise ValueError("text_element and prompt_element must name different elements")

    # Validate length constraints
    if min_length is not None and max_length is not None:
        if min_length > max_length:
            raise ValueError(
                f"min_length ({min_length}) cannot be greater than "
                f"max_length ({max_length})"
            )

    if item_template_id is None:
        item_template_id = uuid4()

    rendered_elements: dict[str, str] = {
        text_element: text,
        prompt_element: prompt,
    }

    # Build item metadata
    item_metadata: dict[str, MetadataValue] = {
        "multiline": multiline,
    }

    if max_length is not None:
        item_metadata["max_length"] = max_length

    if min_length is not None:
        item_metadata["min_length"] = min_length

    if validation_pattern is not None:
        item_metadata["validation_pattern"] = validation_pattern

    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        rendered_elements=rendered_elements,
        item_metadata=item_metadata,
    )


def create_free_text_items_from_texts(
    texts: list[str],
    prompt: str,
    max_length: int | None = None,
    validation_pattern: str | None = None,
    min_length: int | None = None,
    multiline: bool = False,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: Callable[[str], dict[str, MetadataValue]] | None = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
) -> list[Item]:
    """Create free text items from a list of texts with the same prompt.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    prompt : str
        The question/instruction for all items (required).
    max_length : int | None
        Maximum character limit for all items.
    validation_pattern : str | None
        Optional regex pattern for validation.
    min_length : int | None
        Minimum characters required.
    multiline : bool
        True for textarea, False for single-line input.
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

    Returns
    -------
    list[Item]
        Free text items for each text.

    Examples
    --------
    >>> texts = ["Sentence 1", "Sentence 2", "Sentence 3"]
    >>> items = create_free_text_items_from_texts(
    ...     texts,
    ...     prompt="Paraphrase this:",
    ...     multiline=True,
    ...     max_length=200,
    ...     metadata_fn=lambda t: {"original_length": len(t)}
    ... )
    >>> len(items)
    3
    >>> items[0].item_metadata["original_length"]
    10
    """
    free_text_items: list[Item] = []

    for text in texts:
        item_metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            item_metadata = metadata_fn(text)

        item = create_free_text_item(
            text=text,
            prompt=prompt,
            max_length=max_length,
            validation_pattern=validation_pattern,
            min_length=min_length,
            multiline=multiline,
            item_template_id=item_template_id,
            metadata=item_metadata,
            text_element=text_element,
            prompt_element=prompt_element,
        )
        free_text_items.append(item)

    return free_text_items


def create_free_text_items_with_context(
    contexts: list[str],
    prompts: list[str],
    max_length: int | None = None,
    validation_pattern: str | None = None,
    min_length: int | None = None,
    multiline: bool = False,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: (Callable[[str, str], dict[str, MetadataValue]] | None) = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
) -> list[Item]:
    """Create free text items with context + prompt pairs.

    Useful for reading comprehension, question answering where each context
    has a specific question.

    Parameters
    ----------
    contexts : list[str]
        Context texts (same length as prompts).
    prompts : list[str]
        Prompts/questions for each context.
    max_length : int | None
        Maximum character limit for all items.
    validation_pattern : str | None
        Optional regex pattern for validation.
    min_length : int | None
        Minimum characters required.
    multiline : bool
        True for textarea, False for single-line input.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.
    metadata_fn : Callable[[str, str], dict[str, MetadataValue]] | None
        Function to generate metadata from (context, prompt).
    text_element : str
        Name of the rendered element that holds the stimulus text
        (default: "text").
    prompt_element : str
        Name of the rendered element that holds the prompt
        (default: "prompt").

    Returns
    -------
    list[Item]
        Free text items with context + prompt structure.

    Raises
    ------
    ValueError
        If contexts and prompts have different lengths.

    Examples
    --------
    >>> contexts = ["The cat sat on the mat."]
    >>> prompts = ["What sat on the mat?"]
    >>> items = create_free_text_items_with_context(
    ...     contexts,
    ...     prompts,
    ...     max_length=50
    ... )
    >>> len(items)
    1
    >>> items[0].rendered_elements["text"]
    'The cat sat on the mat.'
    >>> items[0].rendered_elements["prompt"]
    'What sat on the mat?'
    """
    if len(contexts) != len(prompts):
        raise ValueError(
            f"contexts and prompts must have same length "
            f"(got {len(contexts)} and {len(prompts)})"
        )

    free_text_items: list[Item] = []

    for context, prompt in zip(contexts, prompts, strict=True):
        item_metadata: dict[str, MetadataValue] = {
            "context": context,
        }
        if metadata_fn:
            item_metadata.update(metadata_fn(context, prompt))

        item = create_free_text_item(
            text=context,
            prompt=prompt,
            max_length=max_length,
            validation_pattern=validation_pattern,
            min_length=min_length,
            multiline=multiline,
            item_template_id=item_template_id,
            metadata=item_metadata,
            text_element=text_element,
            prompt_element=prompt_element,
        )
        free_text_items.append(item)

    return free_text_items


def create_free_text_items_from_groups(
    items: list[Item],
    group_by: Callable[[Item], Any],
    prompt: str,
    max_length: int | None = None,
    validation_pattern: str | None = None,
    min_length: int | None = None,
    multiline: bool = False,
    *,
    extract_text: Callable[[Item], str] | None = None,
    include_group_metadata: bool = True,
    item_template_id: UUID | None = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
) -> list[Item]:
    """Create free text items from grouped source items.

    Groups items and creates one free text item per source item,
    preserving group information in metadata.

    Parameters
    ----------
    items : list[Item]
        Source items to process.
    group_by : Callable[[Item], Any]
        Function to extract grouping key from items.
    prompt : str
        The question/instruction for all items (required).
    max_length : int | None
        Maximum character limit.
    validation_pattern : str | None
        Optional regex pattern for validation.
    min_length : int | None
        Minimum characters required.
    multiline : bool
        True for textarea, False for single-line input.
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

    Returns
    -------
    list[Item]
        Free text items from source items.

    Examples
    --------
    >>> source_items = [
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "Sentence 1"},
    ...         item_metadata={"type": "simple"}
    ...     )
    ... ]
    >>> free_text_items = create_free_text_items_from_groups(
    ...     source_items,
    ...     group_by=lambda i: i.item_metadata["type"],
    ...     prompt="Paraphrase this:",
    ...     multiline=True
    ... )
    >>> len(free_text_items)
    1
    """
    # Group items
    groups: dict[Any, list[Item]] = defaultdict(list)
    for item in items:
        group_key = group_by(item)
        groups[group_key].append(item)

    free_text_items: list[Item] = []

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

            # Create free text item
            free_text_item = create_free_text_item(
                text=text,
                prompt=prompt,
                max_length=max_length,
                validation_pattern=validation_pattern,
                min_length=min_length,
                multiline=multiline,
                item_template_id=item_template_id,
                metadata=item_metadata,
                text_element=text_element,
                prompt_element=prompt_element,
            )
            free_text_items.append(free_text_item)

    return free_text_items


def create_free_text_items_cross_product(
    texts: list[str],
    prompts: list[str],
    max_length: int | None = None,
    validation_pattern: str | None = None,
    min_length: int | None = None,
    multiline: bool = False,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: (Callable[[str, str], dict[str, MetadataValue]] | None) = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
) -> list[Item]:
    """Create free text items from cross-product of texts and prompts.

    Useful when you want to apply multiple prompts to each text.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    prompts : list[str]
        List of prompts to apply.
    max_length : int | None
        Maximum character limit for all items.
    validation_pattern : str | None
        Optional regex pattern for validation.
    min_length : int | None
        Minimum characters required.
    multiline : bool
        True for textarea, False for single-line input.
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

    Returns
    -------
    list[Item]
        Free text items from cross-product.

    Examples
    --------
    >>> texts = ["Sentence 1", "Sentence 2"]
    >>> prompts = ["Paraphrase this:", "Summarize this:"]
    >>> items = create_free_text_items_cross_product(
    ...     texts, prompts, multiline=True, max_length=200
    ... )
    >>> len(items)
    4
    """
    free_text_items: list[Item] = []

    for text, prompt in product(texts, prompts):
        item_metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            item_metadata = metadata_fn(text, prompt)

        item = create_free_text_item(
            text=text,
            prompt=prompt,
            max_length=max_length,
            validation_pattern=validation_pattern,
            min_length=min_length,
            multiline=multiline,
            item_template_id=item_template_id,
            metadata=item_metadata,
            text_element=text_element,
            prompt_element=prompt_element,
        )
        free_text_items.append(item)

    return free_text_items


def create_filtered_free_text_items(
    items: list[Item],
    prompt: str,
    max_length: int | None = None,
    validation_pattern: str | None = None,
    min_length: int | None = None,
    multiline: bool = False,
    *,
    item_filter: Callable[[Item], bool] | None = None,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
    text_element: str = "text",
    prompt_element: str = "prompt",
) -> list[Item]:
    """Create free text items with filtering.

    Parameters
    ----------
    items : list[Item]
        Source items.
    prompt : str
        The question/instruction for all items (required).
    max_length : int | None
        Maximum character limit.
    validation_pattern : str | None
        Optional regex pattern for validation.
    min_length : int | None
        Minimum characters required.
    multiline : bool
        True for textarea, False for single-line input.
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

    Returns
    -------
    list[Item]
        Filtered free text items.

    Examples
    --------
    >>> free_text_items = create_filtered_free_text_items(
    ...     items,
    ...     prompt="Paraphrase this:",
    ...     multiline=True,
    ...     item_filter=lambda i: i.item_metadata.get("valid", True)
    ... )  # doctest: +SKIP
    """
    # Filter items
    filtered_items = items
    if item_filter:
        filtered_items = [item for item in items if item_filter(item)]

    free_text_items: list[Item] = []

    for item in filtered_items:
        # Extract text
        if extract_text:
            text: str = extract_text(item)
        else:
            text = _extract_text_from_item(item)

        # Create free text item
        item_metadata: dict[str, MetadataValue] = {
            "source_item_id": str(item.id),
        }

        free_text_item = create_free_text_item(
            text=text,
            prompt=prompt,
            max_length=max_length,
            validation_pattern=validation_pattern,
            min_length=min_length,
            multiline=multiline,
            item_template_id=item_template_id,
            metadata=item_metadata,
            text_element=text_element,
            prompt_element=prompt_element,
        )
        free_text_items.append(free_text_item)

    return free_text_items


def create_paraphrase_item(
    text: str,
    instruction: str = "Rewrite in your own words:",
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    *,
    text_element: str = "text",
    prompt_element: str = "prompt",
) -> Item:
    """Create a paraphrase generation item.

    Convenience function for paraphrase tasks with multiline input.

    Parameters
    ----------
    text : str
        The text to paraphrase.
    instruction : str
        The instruction for paraphrasing (default: "Rewrite in your own words:").
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

    Returns
    -------
    Item
        Paraphrase free text item.

    Examples
    --------
    >>> item = create_paraphrase_item(
    ...     "The quick brown fox jumps over the lazy dog."
    ... )
    >>> item.rendered_elements["prompt"]
    'Rewrite in your own words:'
    >>> item.item_metadata["multiline"]
    True
    """
    return create_free_text_item(
        text,
        prompt=instruction,
        multiline=True,
        max_length=500,
        item_template_id=item_template_id,
        metadata=metadata,
        text_element=text_element,
        prompt_element=prompt_element,
    )


def create_wh_question_item(
    text: str,
    question_word: str = "Who",
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    *,
    text_element: str = "text",
    prompt_element: str = "prompt",
) -> Item:
    """Create a WH-question answering item.

    Convenience function for WH-question answering with short text input.

    Parameters
    ----------
    text : str
        The context/passage for the question.
    question_word : str
        The question word to use (default: "Who").
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

    Returns
    -------
    Item
        WH-question free text item.

    Examples
    --------
    >>> item = create_wh_question_item(
    ...     "The dog chased the cat.",
    ...     question_word="What"
    ... )
    >>> "What" in item.rendered_elements["prompt"]
    True
    >>> item.item_metadata["max_length"]
    100
    """
    return create_free_text_item(
        text,
        prompt=f"{question_word} question answering:",
        multiline=False,
        max_length=100,
        item_template_id=item_template_id,
        metadata=metadata,
        text_element=text_element,
        prompt_element=prompt_element,
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
