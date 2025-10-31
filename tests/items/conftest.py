"""Shared fixtures for item model tests."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from bead.items.models import (
    Item,
    ItemCollection,
    ItemElement,
    ItemTemplate,
    ItemTemplateCollection,
    ModelOutput,
    PresentationSpec,
    TaskSpec,
)


@pytest.fixture
def sample_uuid() -> UUID:
    """Create a sample UUID for testing.

    Returns
    -------
    UUID
        Sample UUID.
    """
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def task_spec_ordinal() -> TaskSpec:
    """Create an ordinal scale task specification.

    Returns
    -------
    TaskSpec
        Ordinal scale task specification.
    """
    return TaskSpec(
        prompt="How natural does this sentence sound?",
        scale_bounds=(1, 7),
        scale_labels={1: "Very unnatural", 7: "Very natural"},
    )


@pytest.fixture
def task_spec_forced_choice() -> TaskSpec:
    """Create a forced choice task specification.

    Returns
    -------
    TaskSpec
        Forced choice task specification.
    """
    return TaskSpec(
        prompt="Which sentence sounds more natural?",
        options=["Sentence A", "Sentence B", "Both equally natural"],
    )


@pytest.fixture
def task_spec_multi_select() -> TaskSpec:
    """Create a multi-select task specification.

    Returns
    -------
    TaskSpec
        Multi-select task specification.
    """
    return TaskSpec(
        prompt="Select all acceptable sentences:",
        options=["Sentence A", "Sentence B", "Sentence C", "Sentence D"],
        min_selections=1,
        max_selections=4,
    )


@pytest.fixture
def task_spec_binary() -> TaskSpec:
    """Create a binary task specification.

    Returns
    -------
    TaskSpec
        Binary task specification.
    """
    return TaskSpec(
        prompt="Is this sentence acceptable?",
        options=["Yes", "No"],
    )


@pytest.fixture
def task_spec_free_text() -> TaskSpec:
    """Create a free text task specification.

    Returns
    -------
    TaskSpec
        Free text task specification.
    """
    return TaskSpec(
        prompt="Describe what this sentence means:",
        max_length=500,
    )


@pytest.fixture
def presentation_spec_static() -> PresentationSpec:
    """Create a static presentation specification.

    Returns
    -------
    PresentationSpec
        Static presentation specification.
    """
    return PresentationSpec(mode="static")


@pytest.fixture
def element_text() -> ItemElement:
    """Create a text element.

    Returns
    -------
    ItemElement
        Text element.
    """
    return ItemElement(
        element_type="text",
        element_name="context",
        content="Mary loves books.",
        order=1,
    )


@pytest.fixture
def element_template_ref(sample_uuid: UUID) -> ItemElement:
    """Create a template reference element.

    Parameters
    ----------
    sample_uuid : UUID
        UUID to use for the template reference.

    Returns
    -------
    ItemElement
        Template reference element.
    """
    return ItemElement(
        element_type="filled_template_ref",
        element_name="sentence",
        filled_template_ref_id=sample_uuid,
        order=2,
    )


@pytest.fixture
def item_template_simple(
    task_spec_ordinal: TaskSpec,
    presentation_spec_static: PresentationSpec,
    element_text: ItemElement,
) -> ItemTemplate:
    """Create a simple item template with text element.

    Parameters
    ----------
    task_spec_ordinal : TaskSpec
        Task specification for the template.
    presentation_spec_static : PresentationSpec
        Presentation specification for the template.
    element_text : ItemElement
        Text element to include.

    Returns
    -------
    ItemTemplate
        Simple item template (acceptability judgment, ordinal scale task).
    """
    return ItemTemplate(
        name="simple_rating",
        description="Simple acceptability rating task",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        elements=[element_text],
        task_spec=task_spec_ordinal,
        presentation_spec=presentation_spec_static,
    )


@pytest.fixture
def item_template_complex(
    task_spec_ordinal: TaskSpec,
    presentation_spec_static: PresentationSpec,
    element_text: ItemElement,
    element_template_ref: ItemElement,
) -> ItemTemplate:
    """Create a complex item template with multiple elements.

    Parameters
    ----------
    task_spec_ordinal : TaskSpec
        Task specification for the template.
    presentation_spec_static : PresentationSpec
        Presentation specification for the template.
    element_text : ItemElement
        Text element to include.
    element_template_ref : ItemElement
        Template reference element to include.

    Returns
    -------
    ItemTemplate
        Complex item template (acceptability judgment, ordinal scale task).
    """
    return ItemTemplate(
        name="context_target_rating",
        description="Rating with context and target",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        elements=[element_text, element_template_ref],
        task_spec=task_spec_ordinal,
        presentation_spec=presentation_spec_static,
        presentation_order=["context", "sentence"],
    )


@pytest.fixture
def item_template_collection(
    item_template_simple: ItemTemplate,
) -> ItemTemplateCollection:
    """Create an item template collection.

    Parameters
    ----------
    item_template_simple : ItemTemplate
        Template to include in collection.

    Returns
    -------
    ItemTemplateCollection
        Item template collection.
    """
    return ItemTemplateCollection(
        name="acceptability_templates",
        description="Templates for acceptability study",
        templates=[item_template_simple],
    )


@pytest.fixture
def item_template_nli_forced_choice(
    task_spec_forced_choice: TaskSpec,
    presentation_spec_static: PresentationSpec,
) -> ItemTemplate:
    """Create an NLI forced choice template (inference × forced_choice).

    Parameters
    ----------
    task_spec_forced_choice : TaskSpec
        Task specification for the template.
    presentation_spec_static : PresentationSpec
        Presentation specification for the template.

    Returns
    -------
    ItemTemplate
        NLI forced choice template.
    """
    premise_element = ItemElement(
        element_type="text",
        element_name="premise",
        content="Mary loves books.",
        order=1,
    )
    hypothesis_a = ItemElement(
        element_type="text",
        element_name="hypothesis_a",
        content="Mary enjoys reading.",
        order=2,
    )
    hypothesis_b = ItemElement(
        element_type="text",
        element_name="hypothesis_b",
        content="Mary hates books.",
        order=3,
    )
    return ItemTemplate(
        name="nli_forced_choice",
        description="Natural language inference forced choice",
        judgment_type="inference",
        task_type="forced_choice",
        elements=[premise_element, hypothesis_a, hypothesis_b],
        task_spec=task_spec_forced_choice,
        presentation_spec=presentation_spec_static,
        presentation_order=["premise", "hypothesis_a", "hypothesis_b"],
    )


@pytest.fixture
def item_template_plausibility_binary(
    task_spec_binary: TaskSpec,
    presentation_spec_static: PresentationSpec,
    element_text: ItemElement,
) -> ItemTemplate:
    """Create a plausibility binary judgment template (plausibility × binary).

    Parameters
    ----------
    task_spec_binary : TaskSpec
        Task specification for the template.
    presentation_spec_static : PresentationSpec
        Presentation specification for the template.
    element_text : ItemElement
        Text element to include.

    Returns
    -------
    ItemTemplate
        Plausibility binary judgment template.
    """
    return ItemTemplate(
        name="plausibility_binary",
        description="Plausibility binary judgment",
        judgment_type="plausibility",
        task_type="binary",
        elements=[element_text],
        task_spec=task_spec_binary,
        presentation_spec=presentation_spec_static,
    )


@pytest.fixture
def item_template_comprehension_free_text(
    task_spec_free_text: TaskSpec,
    presentation_spec_static: PresentationSpec,
    element_text: ItemElement,
) -> ItemTemplate:
    """Create a comprehension free text template (comprehension × free_text).

    Parameters
    ----------
    task_spec_free_text : TaskSpec
        Task specification for the template.
    presentation_spec_static : PresentationSpec
        Presentation specification for the template.
    element_text : ItemElement
        Text element to include.

    Returns
    -------
    ItemTemplate
        Comprehension free text template.
    """
    return ItemTemplate(
        name="comprehension_free_text",
        description="Comprehension free text response",
        judgment_type="comprehension",
        task_type="free_text",
        elements=[element_text],
        task_spec=task_spec_free_text,
        presentation_spec=presentation_spec_static,
    )


@pytest.fixture
def item_template_acceptability_multi_select(
    task_spec_multi_select: TaskSpec,
    presentation_spec_static: PresentationSpec,
) -> ItemTemplate:
    """Create an acceptability multi-select template (acceptability × multi_select).

    Parameters
    ----------
    task_spec_multi_select : TaskSpec
        Task specification for the template.
    presentation_spec_static : PresentationSpec
        Presentation specification for the template.

    Returns
    -------
    ItemTemplate
        Acceptability multi-select template.
    """
    sent_a = ItemElement(
        element_type="text",
        element_name="sentence_a",
        content="The cat broke the vase.",
        order=1,
    )
    sent_b = ItemElement(
        element_type="text",
        element_name="sentence_b",
        content="Cat the broke vase the.",
        order=2,
    )
    sent_c = ItemElement(
        element_type="text",
        element_name="sentence_c",
        content="The vase was broken by the cat.",
        order=3,
    )
    sent_d = ItemElement(
        element_type="text",
        element_name="sentence_d",
        content="Was broken the vase the cat.",
        order=4,
    )
    return ItemTemplate(
        name="acceptability_multi_select",
        description="Select all acceptable sentences",
        judgment_type="acceptability",
        task_type="multi_select",
        elements=[sent_a, sent_b, sent_c, sent_d],
        task_spec=task_spec_multi_select,
        presentation_spec=presentation_spec_static,
        presentation_order=["sentence_a", "sentence_b", "sentence_c", "sentence_d"],
    )


@pytest.fixture
def model_output_sample() -> ModelOutput:
    """Create a sample model output.

    Returns
    -------
    ModelOutput
        Sample model output.
    """
    return ModelOutput(
        model_name="gpt2",
        model_version="latest",
        operation="log_probability",
        inputs={"text": "The cat broke the vase"},
        output=-12.456,
        cache_key="abc123def456",
        computation_metadata={"device": "cpu", "timestamp": "2025-01-17T12:00:00Z"},
    )


@pytest.fixture
def item_simple(sample_uuid: UUID) -> Item:
    """Create a simple item.

    Parameters
    ----------
    sample_uuid : UUID
        UUID for the item template.

    Returns
    -------
    Item
        Simple item.
    """
    return Item(
        item_template_id=sample_uuid,
        filled_template_refs=[],
        rendered_elements={"sentence": "The cat broke the vase"},
    )


@pytest.fixture
def item_with_model_outputs(
    sample_uuid: UUID, model_output_sample: ModelOutput
) -> Item:
    """Create an item with model outputs.

    Parameters
    ----------
    sample_uuid : UUID
        UUID for the item template.
    model_output_sample : ModelOutput
        Model output to include.

    Returns
    -------
    Item
        Item with model outputs.
    """
    constraint_uuid = uuid4()
    return Item(
        item_template_id=sample_uuid,
        filled_template_refs=[sample_uuid],
        rendered_elements={"sentence": "The cat broke the vase"},
        model_outputs=[model_output_sample],
        constraint_satisfaction={constraint_uuid: True},
    )


@pytest.fixture
def item_collection(sample_uuid: UUID, item_simple: Item) -> ItemCollection:
    """Create an item collection.

    Parameters
    ----------
    sample_uuid : UUID
        UUID for source collections.
    item_simple : Item
        Item to include in collection.

    Returns
    -------
    ItemCollection
        Item collection.
    """
    return ItemCollection(
        name="test_items",
        source_template_collection_id=sample_uuid,
        source_filled_collection_id=sample_uuid,
        items=[item_simple],
        construction_stats={"total_constructed": 1, "total_filtered": 0},
    )
