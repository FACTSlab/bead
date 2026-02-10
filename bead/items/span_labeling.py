"""Utilities for creating span labeling experimental items.

This module provides language-agnostic utilities for creating items with
span annotations. Spans can be added to any existing item type (composability)
or used as standalone span labeling tasks.

Integration Points
------------------
- Active Learning: bead/active_learning/ (via alignment module)
- Deployment: bead/deployment/jspsych/ (span-label plugin)
- Tokenization: bead/tokenization/ (display-level tokens)
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue
from bead.items.spans import (
    LabelSourceType,
    Span,
    SpanSpec,
)
from bead.tokenization.config import TokenizerConfig
from bead.tokenization.tokenizers import TokenizedText, create_tokenizer

_SPAN_REF_PATTERN = re.compile(r"\[\[([^\]:]+?)(?::([^\]]+?))?\]\]")


def tokenize_item(
    item: Item,
    tokenizer_config: TokenizerConfig | None = None,
) -> Item:
    """Tokenize an item's rendered_elements.

    Populates ``tokenized_elements`` and ``token_space_after`` using the
    configured tokenizer. Returns a new ``Item`` (does not mutate).

    Parameters
    ----------
    item : Item
        Item to tokenize.
    tokenizer_config : TokenizerConfig | None
        Tokenizer configuration. If None, uses default (spaCy English).

    Returns
    -------
    Item
        New item with populated tokenized_elements and token_space_after.
    """
    if tokenizer_config is None:
        tokenizer_config = TokenizerConfig()

    tokenize = create_tokenizer(tokenizer_config)

    tokenized_elements: dict[str, list[str]] = {}
    token_space_after: dict[str, list[bool]] = {}

    for name, text in item.rendered_elements.items():
        result: TokenizedText = tokenize(text)
        tokenized_elements[name] = result.token_texts
        token_space_after[name] = result.space_after_flags

    # Create new item with tokenization data
    data = item.model_dump()
    data["tokenized_elements"] = tokenized_elements
    data["token_space_after"] = token_space_after
    return Item(**data)


def _validate_span_indices(
    spans: list[Span],
    tokenized_elements: dict[str, list[str]],
) -> None:
    """Validate span indices are within token bounds.

    Parameters
    ----------
    spans : list[Span]
        Spans to validate.
    tokenized_elements : dict[str, list[str]]
        Tokenized element data.

    Raises
    ------
    ValueError
        If any span index is out of bounds or references an unknown element.
    """
    for span in spans:
        for segment in span.segments:
            if segment.element_name not in tokenized_elements:
                raise ValueError(
                    f"Span '{span.span_id}' segment references element "
                    f"'{segment.element_name}' which is not in "
                    f"tokenized_elements. Available: "
                    f"{list(tokenized_elements.keys())}"
                )
            n_tokens = len(tokenized_elements[segment.element_name])
            for idx in segment.indices:
                if idx >= n_tokens:
                    raise ValueError(
                        f"Span '{span.span_id}' has index {idx} in element "
                        f"'{segment.element_name}' but element only has "
                        f"{n_tokens} tokens"
                    )


def create_span_item(
    text: str,
    spans: list[Span],
    prompt: str,
    tokenizer_config: TokenizerConfig | None = None,
    tokens: list[str] | None = None,
    labels: list[str] | None = None,
    span_spec: SpanSpec | None = None,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a standalone span labeling item.

    Tokenizes text using config, validates span indices against tokens.

    Parameters
    ----------
    text : str
        The stimulus text.
    spans : list[Span]
        Pre-defined span annotations.
    prompt : str
        Question or instruction for the participant.
    tokenizer_config : TokenizerConfig | None
        Tokenizer configuration. Ignored if ``tokens`` is provided.
    tokens : list[str] | None
        Pre-tokenized text (overrides tokenizer).
    labels : list[str] | None
        Fixed label set for span labeling.
    span_spec : SpanSpec | None
        Span specification. If None, creates a default static spec.
    item_template_id : UUID | None
        Template ID. If None, generates a new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional item metadata.

    Returns
    -------
    Item
        Span labeling item.

    Raises
    ------
    ValueError
        If text is empty or span indices are out of bounds.
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    if item_template_id is None:
        item_template_id = uuid4()

    if span_spec is None:
        span_spec = SpanSpec(
            interaction_mode="static",
            labels=labels,
        )

    # Store span_spec in item metadata for downstream access
    span_spec_data: dict[str, MetadataValue] = {}
    for k, v in span_spec.model_dump(mode="json").items():
        span_spec_data[k] = v

    # Tokenize
    if tokens is not None:
        tokenized_elements = {"text": tokens}
        # Infer space_after from text
        token_space_after = {"text": _infer_space_after(tokens, text)}
    else:
        if tokenizer_config is None:
            tokenizer_config = TokenizerConfig()
        tokenize = create_tokenizer(tokenizer_config)
        result = tokenize(text)
        tokenized_elements = {"text": result.token_texts}
        token_space_after = {"text": result.space_after_flags}

    # Validate spans
    _validate_span_indices(spans, tokenized_elements)

    item_metadata: dict[str, MetadataValue] = {"_span_spec": span_spec_data}
    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        rendered_elements={"text": text, "prompt": prompt},
        spans=spans,
        tokenized_elements=tokenized_elements,
        token_space_after=token_space_after,
        item_metadata=item_metadata,
    )


def create_interactive_span_item(
    text: str,
    prompt: str,
    tokenizer_config: TokenizerConfig | None = None,
    tokens: list[str] | None = None,
    label_set: list[str] | None = None,
    label_source: LabelSourceType = "fixed",
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create an item for interactive span selection by participants.

    Parameters
    ----------
    text : str
        The stimulus text.
    prompt : str
        Instruction for the participant.
    tokenizer_config : TokenizerConfig | None
        Tokenizer configuration.
    tokens : list[str] | None
        Pre-tokenized text (overrides tokenizer).
    label_set : list[str] | None
        Fixed label set (when label_source is "fixed").
    label_source : LabelSourceType
        Label source type ("fixed" or "wikidata").
    item_template_id : UUID | None
        Template ID. If None, generates a new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional item metadata.

    Returns
    -------
    Item
        Interactive span labeling item (no pre-defined spans).
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    if item_template_id is None:
        item_template_id = uuid4()

    # Build span spec from label parameters
    span_spec = SpanSpec(
        interaction_mode="interactive",
        label_source=label_source,
        labels=label_set,
    )
    span_spec_data: dict[str, MetadataValue] = {}
    for k, v in span_spec.model_dump(mode="json").items():
        span_spec_data[k] = v

    # Tokenize
    if tokens is not None:
        tokenized_elements = {"text": tokens}
        token_space_after = {"text": _infer_space_after(tokens, text)}
    else:
        if tokenizer_config is None:
            tokenizer_config = TokenizerConfig()
        tokenize = create_tokenizer(tokenizer_config)
        result = tokenize(text)
        tokenized_elements = {"text": result.token_texts}
        token_space_after = {"text": result.space_after_flags}

    item_metadata: dict[str, MetadataValue] = {"_span_spec": span_spec_data}
    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        rendered_elements={"text": text, "prompt": prompt},
        spans=[],
        tokenized_elements=tokenized_elements,
        token_space_after=token_space_after,
        item_metadata=item_metadata,
    )


def add_spans_to_item(
    item: Item,
    spans: list[Span],
    tokenizer_config: TokenizerConfig | None = None,
    span_spec: SpanSpec | None = None,
) -> Item:
    """Add span annotations to any existing item.

    This is the key composability function: any item (rating, forced choice,
    binary, etc.) can have spans added as an overlay. Tokenizes
    rendered_elements if not already tokenized. Returns a new Item.

    Parameters
    ----------
    item : Item
        Existing item to add spans to.
    spans : list[Span]
        Span annotations to add.
    tokenizer_config : TokenizerConfig | None
        Tokenizer configuration (used only if item lacks tokenization).
    span_spec : SpanSpec | None
        Span specification.

    Returns
    -------
    Item
        New item with spans added.

    Raises
    ------
    ValueError
        If span indices are out of bounds.
    """
    # Tokenize if needed
    if not item.tokenized_elements:
        item = tokenize_item(item, tokenizer_config)

    # Validate spans
    _validate_span_indices(spans, item.tokenized_elements)

    # Warn if prompt contains [[label]] references to nonexistent span labels
    prompt_text = item.rendered_elements.get("prompt", "")
    if prompt_text:
        all_spans = list(item.spans) + spans
        span_labels = {
            s.label.label for s in all_spans if s.label is not None
        }
        for match in _SPAN_REF_PATTERN.finditer(prompt_text):
            ref_label = match.group(1)
            if ref_label not in span_labels:
                warnings.warn(
                    f"Prompt contains [[{ref_label}]] but no span with "
                    f"label '{ref_label}' exists. Available labels: "
                    f"{sorted(span_labels)}",
                    UserWarning,
                    stacklevel=2,
                )

    # Build new item with spans
    data = item.model_dump()
    # Merge existing spans with new ones
    existing_spans = data.get("spans", [])
    data["spans"] = existing_spans + [s.model_dump() for s in spans]

    # Store span_spec in item metadata if provided
    if span_spec is not None:
        item_metadata = dict(data.get("item_metadata", {}))
        span_spec_data: dict[str, MetadataValue] = {}
        for k, v in span_spec.model_dump().items():
            span_spec_data[k] = v
        item_metadata["_span_spec"] = span_spec_data
        data["item_metadata"] = item_metadata

    return Item(**data)


def create_span_items_from_texts(
    texts: list[str],
    span_extractor: Callable[[str, list[str]], list[Span]],
    prompt: str,
    tokenizer_config: TokenizerConfig | None = None,
    labels: list[str] | None = None,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Batch create span items with automatic tokenization.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    span_extractor : Callable[[str, list[str]], list[Span]]
        Function that takes (text, tokens) and returns spans.
    prompt : str
        Question or instruction for the participant.
    tokenizer_config : TokenizerConfig | None
        Tokenizer configuration.
    labels : list[str] | None
        Fixed label set.
    item_template_id : UUID | None
        Shared template ID. If None, generates one per item.

    Returns
    -------
    list[Item]
        Span labeling items.
    """
    if tokenizer_config is None:
        tokenizer_config = TokenizerConfig()
    tokenize = create_tokenizer(tokenizer_config)

    items: list[Item] = []
    for text in texts:
        result = tokenize(text)
        tokens = result.token_texts
        spans = span_extractor(text, tokens)
        item = create_span_item(
            text=text,
            spans=spans,
            prompt=prompt,
            tokens=tokens,
            labels=labels,
            item_template_id=item_template_id,
        )
        items.append(item)

    return items


def _infer_space_after(tokens: list[str], text: str) -> list[bool]:
    """Infer space_after flags from pre-tokenized text.

    Attempts to locate each token in the original text and check if a
    space follows. Falls back to True for all tokens if alignment fails.

    Parameters
    ----------
    tokens : list[str]
        Token strings.
    text : str
        Original text.

    Returns
    -------
    list[bool]
        Per-token space_after flags.
    """
    flags: list[bool] = []
    offset = 0
    for token in tokens:
        idx = text.find(token, offset)
        if idx == -1:
            # Can't find token; assume space after
            flags.append(True)
        else:
            end = idx + len(token)
            space_after = end < len(text) and text[end] == " "
            flags.append(space_after)
            offset = end
    return flags
