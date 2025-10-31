"""Tests for lexical item models."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from bead.resources import LexicalItem


class TestLexicalItemCreation:
    """Test lexical item creation."""

    def test_create_with_all_fields(self) -> None:
        """Test creating a lexical item with all fields."""
        item = LexicalItem(
            lemma="walk",
            pos="VERB",
            form="walked",
            features={"tense": "past", "transitive": True},
            attributes={"frequency": 1000, "rating": 4.5},
            source="manual",
        )
        assert item.lemma == "walk"
        assert item.pos == "VERB"
        assert item.form == "walked"
        assert item.features["tense"] == "past"
        assert item.attributes["frequency"] == 1000
        assert item.source == "manual"

    def test_create_with_minimal_fields(self) -> None:
        """Test creating a lexical item with minimal fields."""
        item = LexicalItem(lemma="run")
        assert item.lemma == "run"
        assert item.pos is None
        assert item.form is None
        assert item.language_code is None
        assert item.features == {}
        assert item.attributes == {}
        assert item.source is None

    def test_create_with_empty_features_dict(self) -> None:
        """Test creating a lexical item with empty features dict."""
        item = LexicalItem(lemma="jump", features={})
        assert item.features == {}

    def test_create_with_empty_attributes_dict(self) -> None:
        """Test creating a lexical item with empty attributes dict."""
        item = LexicalItem(lemma="swim", attributes={})
        assert item.attributes == {}

    def test_create_with_none_pos(self) -> None:
        """Test creating a lexical item with None pos."""
        item = LexicalItem(lemma="fly", pos=None)
        assert item.pos is None

    def test_create_with_form_different_from_lemma(self) -> None:
        """Test creating a lexical item with form different from lemma."""
        item = LexicalItem(lemma="go", form="went")
        assert item.lemma == "go"
        assert item.form == "went"

    def test_create_with_special_characters_in_lemma(self) -> None:
        """Test creating a lexical item with special characters in lemma."""
        item = LexicalItem(lemma="rock-and-roll")
        assert item.lemma == "rock-and-roll"

    def test_create_with_nested_features(self) -> None:
        """Test creating a lexical item with nested features."""
        item = LexicalItem(
            lemma="test",
            features={"morphology": {"prefix": "re", "suffix": "ed"}},
        )
        assert item.features["morphology"]["prefix"] == "re"


class TestLexicalItemValidation:
    """Test lexical item validation."""

    def test_empty_lemma_fails(self) -> None:
        """Test that empty lemma validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="")
        assert "lemma must be non-empty" in str(exc_info.value)

    def test_whitespace_only_lemma_fails(self) -> None:
        """Test that whitespace-only lemma validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="   ")
        assert "lemma must be non-empty" in str(exc_info.value)

    def test_lowercase_pos_fails(self) -> None:
        """Test that lowercase pos validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="test", pos="verb")
        assert "pos must be uppercase" in str(exc_info.value)

    def test_mixed_case_pos_fails(self) -> None:
        """Test that mixed case pos validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="test", pos="Verb")
        assert "pos must be uppercase" in str(exc_info.value)


class TestLexicalItemMutability:
    """Test lexical item mutability."""

    def test_features_are_mutable(self) -> None:
        """Test that features dict is mutable."""
        item = LexicalItem(lemma="test")
        item.features["new_feature"] = "value"
        assert item.features["new_feature"] == "value"

    def test_attributes_are_mutable(self) -> None:
        """Test that attributes dict is mutable."""
        item = LexicalItem(lemma="test")
        item.attributes["new_attr"] = 42
        assert item.attributes["new_attr"] == 42


class TestLexicalItemInheritance:
    """Test lexical item inheritance from BeadBaseModel."""

    def test_inherits_uuidv7_id(self) -> None:
        """Test that lexical item inherits UUID id from BeadBaseModel."""
        item = LexicalItem(lemma="test")
        assert isinstance(item.id, UUID)

    def test_inherits_timestamps(self) -> None:
        """Test that lexical item inherits timestamps from BeadBaseModel."""
        item = LexicalItem(lemma="test")
        assert hasattr(item, "created_at")
        assert hasattr(item, "modified_at")
        assert item.created_at is not None
        assert item.modified_at is not None

    def test_metadata_tracking(self) -> None:
        """Test that lexical item has metadata tracking."""
        item = LexicalItem(lemma="test")
        assert hasattr(item, "metadata")


class TestLexicalItemSerialization:
    """Test lexical item serialization."""

    def test_model_dump(self) -> None:
        """Test lexical item serialization with model_dump."""
        item = LexicalItem(
            lemma="walk",
            pos="VERB",
            features={"tense": "present"},
            attributes={"frequency": 1000},
        )
        data = item.model_dump()
        assert data["lemma"] == "walk"
        assert data["pos"] == "VERB"
        assert data["features"]["tense"] == "present"
        assert data["attributes"]["frequency"] == 1000

    def test_deserialization(self) -> None:
        """Test lexical item deserialization with model_validate."""
        data = {
            "lemma": "run",
            "pos": "VERB",
            "features": {"tense": "past"},
            "attributes": {"frequency": 500},
        }
        item = LexicalItem.model_validate(data)
        assert item.lemma == "run"
        assert item.pos == "VERB"
        assert item.features["tense"] == "past"
        assert item.attributes["frequency"] == 500

    def test_model_copy(self) -> None:
        """Test lexical item model_copy."""
        item = LexicalItem(
            lemma="walk",
            pos="VERB",
            features={"tense": "present"},
        )
        copy = item.model_copy()
        assert copy.lemma == item.lemma
        assert copy.pos == item.pos
        assert copy.id == item.id  # Copies preserve ID


class TestLexicalItemAttributeTypes:
    """Test lexical item with various attribute types."""

    def test_string_attribute(self) -> None:
        """Test lexical item with string attribute."""
        item = LexicalItem(lemma="test", attributes={"category": "motion"})
        assert item.attributes["category"] == "motion"

    def test_int_attribute(self) -> None:
        """Test lexical item with int attribute."""
        item = LexicalItem(lemma="test", attributes={"count": 42})
        assert item.attributes["count"] == 42

    def test_float_attribute(self) -> None:
        """Test lexical item with float attribute."""
        item = LexicalItem(lemma="test", attributes={"rating": 4.5})
        assert item.attributes["rating"] == 4.5

    def test_bool_attribute(self) -> None:
        """Test lexical item with bool attribute."""
        item = LexicalItem(lemma="test", attributes={"is_common": True})
        assert item.attributes["is_common"] is True

    def test_list_attribute(self) -> None:
        """Test lexical item with list attribute."""
        item = LexicalItem(lemma="test", attributes={"synonyms": ["run", "jog"]})
        assert item.attributes["synonyms"] == ["run", "jog"]


class TestLexicalItemLanguageCode:
    """Test lexical item language code functionality."""

    def test_create_with_language_code(self) -> None:
        """Test creating a lexical item with language code."""
        item = LexicalItem(lemma="walk", pos="VERB", language_code="en")
        assert item.language_code == "eng"  # Normalized to ISO 639-3

    def test_language_code_normalization(self) -> None:
        """Test that language codes are normalized to ISO 639-3."""
        # English: en → eng
        item1 = LexicalItem(lemma="test", language_code="en")
        assert item1.language_code == "eng"

        # Korean: ko → kor
        item2 = LexicalItem(lemma="테스트", language_code="ko")
        assert item2.language_code == "kor"

        # Already ISO 639-3 stays the same
        item3 = LexicalItem(lemma="test", language_code="eng")
        assert item3.language_code == "eng"

    def test_language_code_validation(self) -> None:
        """Test that invalid language codes are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="test", language_code="invalid")
        assert "Invalid language code" in str(exc_info.value)

    def test_language_code_iso639_1(self) -> None:
        """Test ISO 639-1 (2-letter) language codes."""
        item = LexicalItem(lemma="먹다", language_code="ko")
        assert item.language_code == "kor"  # Normalized to ISO 639-3

    def test_language_code_iso639_3(self) -> None:
        """Test ISO 639-3 (3-letter) language codes."""
        item = LexicalItem(lemma="test", language_code="eng")
        assert item.language_code == "eng"

    def test_language_code_none(self) -> None:
        """Test that None language code is valid (optional)."""
        item = LexicalItem(lemma="test", language_code=None)
        assert item.language_code is None
