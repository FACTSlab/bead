"""Test ISO 639 language code validation."""

import pytest

from bead.data.language_codes import validate_iso639_code


def test_validate_iso639_1_codes() -> None:
    """Test validation of ISO 639-1 codes (2-letter)."""
    assert validate_iso639_code("en") == "eng"
    assert validate_iso639_code("es") == "spa"
    assert validate_iso639_code("fr") == "fra"
    assert validate_iso639_code("de") == "deu"
    assert validate_iso639_code("ko") == "kor"
    assert validate_iso639_code("ja") == "jpn"
    assert validate_iso639_code("zh") == "zho"
    assert validate_iso639_code("ar") == "ara"


def test_validate_iso639_3_codes() -> None:
    """Test validation of ISO 639-3 codes (3-letter)."""
    assert validate_iso639_code("eng") == "eng"
    assert validate_iso639_code("spa") == "spa"
    assert validate_iso639_code("fra") == "fra"
    assert validate_iso639_code("deu") == "deu"
    assert validate_iso639_code("kor") == "kor"
    assert validate_iso639_code("jpn") == "jpn"
    assert validate_iso639_code("zho") == "zho"


def test_validate_none_is_valid() -> None:
    """Test that None is a valid language code (optional)."""
    assert validate_iso639_code(None) is None


def test_validate_invalid_code_raises() -> None:
    """Test that invalid codes raise ValueError."""
    with pytest.raises(ValueError, match="Invalid language code: 'invalid'"):
        validate_iso639_code("invalid")

    with pytest.raises(ValueError, match="Invalid language code: 'zzzz'"):
        validate_iso639_code("zzzz")

    with pytest.raises(ValueError, match="Invalid language code: 'aaa1'"):
        validate_iso639_code("aaa1")


def test_validate_empty_string_raises() -> None:
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid language code: ''"):
        validate_iso639_code("")


def test_validate_less_common_languages() -> None:
    """Test validation of less common language codes."""
    # Igbo
    assert validate_iso639_code("ig") == "ibo"
    assert validate_iso639_code("ibo") == "ibo"

    # Yoruba
    assert validate_iso639_code("yo") == "yor"
    assert validate_iso639_code("yor") == "yor"

    # Zulu
    assert validate_iso639_code("zu") == "zul"
    assert validate_iso639_code("zul") == "zul"

    # Marathi
    assert validate_iso639_code("mr") == "mar"
    assert validate_iso639_code("mar") == "mar"
