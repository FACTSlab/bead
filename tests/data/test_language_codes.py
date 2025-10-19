"""Tests for ISO 639 language code validation and types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sash.data.language_codes import validate_iso639_code
from sash.resources.lexicon import Lexicon
from sash.resources.models import LexicalItem
from sash.resources.structures import Slot, Template


def test_validate_iso639_code_valid_2_letter() -> None:
    """Test validation of valid ISO 639-1 codes."""
    assert validate_iso639_code("en") == "en"
    assert validate_iso639_code("ko") == "ko"
    assert validate_iso639_code("zu") == "zu"


def test_validate_iso639_code_valid_3_letter() -> None:
    """Test validation of valid ISO 639-3 codes."""
    assert validate_iso639_code("eng") == "eng"
    assert validate_iso639_code("kor") == "kor"
    assert validate_iso639_code("zul") == "zul"


def test_validate_iso639_code_case_insensitive() -> None:
    """Test that codes are normalized to lowercase."""
    assert validate_iso639_code("EN") == "en"
    assert validate_iso639_code("ENG") == "eng"
    assert validate_iso639_code("Ko") == "ko"


def test_validate_iso639_code_none() -> None:
    """Test that None is allowed."""
    assert validate_iso639_code(None) is None


def test_validate_iso639_code_invalid_length() -> None:
    """Test that invalid length codes are rejected."""
    with pytest.raises(ValueError, match="Must be 2 letters.*or 3 letters"):
        validate_iso639_code("e")

    with pytest.raises(ValueError, match="Must be 2 letters.*or 3 letters"):
        validate_iso639_code("engl")


def test_validate_iso639_code_invalid_code() -> None:
    """Test that non-existent codes are rejected."""
    with pytest.raises(ValueError, match="Invalid ISO 639"):
        validate_iso639_code("xx")

    with pytest.raises(ValueError, match="Invalid ISO 639"):
        validate_iso639_code("zzz")


def test_validate_iso639_code_non_alpha() -> None:
    """Test that non-alphabetic codes are rejected."""
    with pytest.raises(ValueError, match="Invalid ISO 639"):
        validate_iso639_code("e1")

    with pytest.raises(ValueError, match="Invalid ISO 639"):
        validate_iso639_code("en-")


def test_lexical_item_valid_language_code() -> None:
    """Test LexicalItem accepts valid language codes."""
    item = LexicalItem(lemma="walk", language_code="en")
    assert item.language_code == "en"

    item = LexicalItem(lemma="먹다", language_code="ko")
    assert item.language_code == "ko"


def test_lexical_item_invalid_language_code() -> None:
    """Test LexicalItem rejects invalid language codes."""
    with pytest.raises(ValidationError, match="Invalid ISO 639"):
        LexicalItem(lemma="test", language_code="invalid")


def test_lexical_item_none_language_code() -> None:
    """Test LexicalItem accepts None for language_code."""
    item = LexicalItem(lemma="test", language_code=None)
    assert item.language_code is None


def test_lexicon_valid_language_code() -> None:
    """Test Lexicon accepts valid language codes."""
    lex = Lexicon(name="test", language_code="en")
    assert lex.language_code == "en"


def test_lexicon_case_normalization() -> None:
    """Test that language codes are normalized to lowercase."""
    lex = Lexicon(name="test", language_code="EN")
    assert lex.language_code == "en"

    item = LexicalItem(lemma="test", language_code="KO")
    assert item.language_code == "ko"


def test_template_valid_language_code() -> None:
    """Test Template accepts valid language codes."""
    slot = Slot(name="x")
    template = Template(
        name="test",
        template_string="{x}.",
        slots={"x": slot},
        language_code="en",
    )
    assert template.language_code == "en"


def test_template_invalid_language_code() -> None:
    """Test Template rejects invalid language codes."""
    slot = Slot(name="x")
    with pytest.raises(ValidationError, match="Invalid ISO 639"):
        Template(
            name="test",
            template_string="{x}.",
            slots={"x": slot},
            language_code="invalid",
        )
