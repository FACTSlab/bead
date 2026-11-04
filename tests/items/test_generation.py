"""Tests for cross-product item generation utilities."""

from __future__ import annotations

import pytest

from bead.items.generation import (
    create_cross_product_items,
    create_filtered_cross_product_items,
    create_stratified_cross_product_items,
)
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template


@pytest.fixture
def templates() -> list[Template]:
    """Provide two templates without declared slots."""
    return [
        Template(name="intransitive", template_string="{subject} {head}.", slots={}),
        Template(
            name="transitive", template_string="{subject} {head} {object}.", slots={}
        ),
    ]


@pytest.fixture
def lexicons() -> dict[str, Lexicon]:
    """Provide a lexicon keyed by a slot name other than "verb"."""
    head = Lexicon(
        name="heads",
        items=(
            LexicalItem(
                lemma="walk",
                language_code="eng",
                form="walks",
                features={"freq": "high"},
            ),
            LexicalItem(lemma="eat", language_code="eng", features={"freq": "high"}),
            LexicalItem(lemma="devour", language_code="eng", features={"freq": "low"}),
        ),
    )
    return {"head": head}


class TestCreateCrossProductItems:
    """Test create_cross_product_items() function."""

    def test_slot_must_be_named(
        self, templates: list[Template], lexicons: dict[str, Lexicon]
    ) -> None:
        """Test that the cross-product slot has no default."""
        with pytest.raises(TypeError, match="cross_product_slot"):
            create_cross_product_items(templates, lexicons)

    def test_keys_derive_from_declared_slot(
        self, templates: list[Template], lexicons: dict[str, Lexicon]
    ) -> None:
        """Test that rendered element keys are derived from the declared slot."""
        items = list(
            create_cross_product_items(templates, lexicons, cross_product_slot="head")
        )

        assert len(items) == 6
        first = items[0]
        assert first.rendered_elements["head_lemma"] == "walk"
        assert first.rendered_elements["head_form"] == "walks"
        assert "verb_lemma" not in first.rendered_elements
        assert items[1].rendered_elements["head_form"] == "eat"

    def test_missing_slot_lexicon_raises(
        self, templates: list[Template], lexicons: dict[str, Lexicon]
    ) -> None:
        """Test that naming a slot without a lexicon raises ValueError."""
        with pytest.raises(ValueError, match="Lexicon for slot 'verb' not found"):
            list(
                create_cross_product_items(
                    templates, lexicons, cross_product_slot="verb"
                )
            )


class TestCreateFilteredCrossProductItems:
    """Test create_filtered_cross_product_items() function."""

    def test_slot_must_be_named(
        self, templates: list[Template], lexicons: dict[str, Lexicon]
    ) -> None:
        """Test that the cross-product slot has no default."""
        with pytest.raises(TypeError, match="cross_product_slot"):
            create_filtered_cross_product_items(templates, lexicons)

    def test_keys_derive_from_declared_slot(
        self, templates: list[Template], lexicons: dict[str, Lexicon]
    ) -> None:
        """Test filtering with keys derived from the declared slot."""
        items = list(
            create_filtered_cross_product_items(
                templates,
                lexicons,
                cross_product_slot="head",
                template_filter=lambda t: t.name == "transitive",
                item_filter=lambda i: i.lemma != "eat",
            )
        )

        assert [i.rendered_elements["head_lemma"] for i in items] == [
            "walk",
            "devour",
        ]


class TestCreateStratifiedCrossProductItems:
    """Test create_stratified_cross_product_items() function."""

    def test_slot_must_be_named(
        self, templates: list[Template], lexicons: dict[str, Lexicon]
    ) -> None:
        """Test that the cross-product slot has no default."""
        with pytest.raises(TypeError, match="cross_product_slot"):
            create_stratified_cross_product_items(
                templates,
                lexicons,
                stratify_by=lambda i: str(i.features["freq"]),
                items_per_stratum=1,
            )

    def test_keys_derive_from_declared_slot(
        self, templates: list[Template], lexicons: dict[str, Lexicon]
    ) -> None:
        """Test stratified sampling with keys derived from the declared slot."""
        items = list(
            create_stratified_cross_product_items(
                templates,
                lexicons,
                cross_product_slot="head",
                stratify_by=lambda i: str(i.features["freq"]),
                items_per_stratum=1,
            )
        )

        assert sorted({i.rendered_elements["head_lemma"] for i in items}) == [
            "devour",
            "walk",
        ]
        assert len(items) == 4
