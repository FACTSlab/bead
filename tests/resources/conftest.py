"""Pytest fixtures for resource tests."""

from __future__ import annotations

import pytest

from sash.resources import (
    DSLConstraint,
    ExtensionalConstraint,
    IntensionalConstraint,
    LexicalItem,
    RelationalConstraint,
    Slot,
    Template,
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
