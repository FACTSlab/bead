"""Pytest fixtures for resource tests."""

from __future__ import annotations

import pytest

from sash.resources import (
    DSLConstraint,
    ExtensionalConstraint,
    IntensionalConstraint,
    LexicalItem,
    Lexicon,
    RelationalConstraint,
    Slot,
    Template,
    TemplateCollection,
)


@pytest.fixture
def sample_lexical_item() -> LexicalItem:
    """Provide a sample lexical item."""
    return LexicalItem(
        lemma="walk",
        pos="VERB",
        features={"tense": "present", "transitive": True},
        attributes={"frequency": 1000},
        source="manual",
    )


@pytest.fixture
def sample_noun() -> LexicalItem:
    """Provide a sample noun."""
    return LexicalItem(
        lemma="dog",
        pos="NOUN",
        features={"number": "singular", "animacy": "animate"},
        attributes={"frequency": 500},
    )


@pytest.fixture
def sample_extensional_constraint(
    sample_lexical_item: LexicalItem,
) -> ExtensionalConstraint:
    """Provide a sample extensional constraint."""
    return ExtensionalConstraint(
        mode="allow",
        items=[sample_lexical_item.id],
    )


@pytest.fixture
def sample_intensional_constraint() -> IntensionalConstraint:
    """Provide a sample intensional constraint."""
    return IntensionalConstraint(
        property="pos",
        operator="==",
        value="VERB",
    )


@pytest.fixture
def sample_relational_constraint() -> RelationalConstraint:
    """Provide a sample relational constraint."""
    return RelationalConstraint(
        slot_a="subject",
        slot_b="object",
        relation="different",
        property="lemma",
    )


@pytest.fixture
def sample_dsl_constraint() -> DSLConstraint:
    """Provide a sample DSL constraint."""
    return DSLConstraint(
        expression="pos == 'VERB' and len(lemma) > 3",
    )


@pytest.fixture
def sample_slot(sample_intensional_constraint: IntensionalConstraint) -> Slot:
    """Provide a sample slot."""
    return Slot(
        name="subject",
        description="The subject of the sentence",
        constraints=[sample_intensional_constraint],
        required=True,
    )


@pytest.fixture
def sample_template(sample_slot: Slot) -> Template:
    """Provide a sample template."""
    verb_slot = Slot(name="verb", required=True)
    object_slot = Slot(name="object", required=True)
    return Template(
        name="simple_transitive",
        template_string="{subject} {verb} {object}.",
        slots={
            "subject": sample_slot,
            "verb": verb_slot,
            "object": object_slot,
        },
        tags=["transitive", "simple"],
    )


@pytest.fixture
def sample_lexicon() -> Lexicon:
    """Provide a sample lexicon with multiple items."""
    lexicon = Lexicon(name="test_lexicon", language_code="en")
    lexicon.add(LexicalItem(lemma="walk", pos="VERB", attributes={"frequency": 1000}))
    lexicon.add(LexicalItem(lemma="run", pos="VERB", attributes={"frequency": 800}))
    lexicon.add(LexicalItem(lemma="dog", pos="NOUN", attributes={"frequency": 500}))
    return lexicon


@pytest.fixture
def sample_template_collection(sample_template: Template) -> TemplateCollection:
    """Provide a sample template collection."""
    collection = TemplateCollection(name="test_collection")

    # Add the sample template
    collection.add(sample_template)

    # Add another simple template
    slot_x = Slot(name="x", required=True)
    template2 = Template(
        name="intransitive",
        template_string="{x} happened.",
        slots={"x": slot_x},
        tags=["intransitive", "simple"],
    )
    collection.add(template2)

    return collection


@pytest.fixture
def english_item() -> LexicalItem:
    """Provide an English lexical item."""
    return LexicalItem(lemma="walk", pos="VERB", language_code="en")


@pytest.fixture
def korean_item() -> LexicalItem:
    """Provide a Korean lexical item."""
    return LexicalItem(lemma="먹다", pos="VERB", language_code="ko")


@pytest.fixture
def english_template() -> Template:
    """Provide an English template."""
    slot = Slot(name="x", required=True)
    return Template(
        name="english_test",
        template_string="{x}.",
        slots={"x": slot},
        language_code="en",
    )


@pytest.fixture
def korean_template() -> Template:
    """Provide a Korean template."""
    slot = Slot(name="x", required=True)
    return Template(
        name="korean_test",
        template_string="{x}.",
        slots={"x": slot},
        language_code="ko",
    )
