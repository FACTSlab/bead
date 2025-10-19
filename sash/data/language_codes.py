"""ISO 639 language code validation and type definitions.

This module provides type-safe language code validation using ISO 639-1
(2-letter) and ISO 639-3 (3-letter) standards. Validation is enforced at
runtime using Pydantic's validation system.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, Field


def validate_iso639_code(code: str | None) -> str | None:
    """Validate that code conforms to ISO 639-1 or ISO 639-3.

    Parameters
    ----------
    code : str | None
        Language code to validate (e.g., "en", "eng", "ko", "kor").

    Returns
    -------
    str | None
        The validated language code (normalized to lowercase).

    Raises
    ------
    ValueError
        If code is not a valid ISO 639-1 (2-letter) or ISO 639-3 (3-letter) code.

    Notes
    -----
    Uses the `langcodes` package for validation, which supports:
    - ISO 639-1 (2-letter): "en", "ko", "ig", "mr", "zu"
    - ISO 639-3 (3-letter): "eng", "kor", "ibo", "mar", "zul"

    Examples
    --------
    >>> validate_iso639_code("en")
    'en'
    >>> validate_iso639_code("eng")
    'eng'
    >>> validate_iso639_code("korean")
    ValueError: Invalid ISO 639 language code: 'korean'
    >>> validate_iso639_code(None)
    None
    """
    if code is None:
        return None

    code_lower = code.lower().strip()

    # Validate length
    if len(code_lower) not in (2, 3):
        raise ValueError(
            f"Invalid ISO 639 language code: '{code}'. "
            f"Must be 2 letters (ISO 639-1) or 3 letters (ISO 639-3)."
        )

    try:
        import langcodes  # noqa: PLC0415

        # langcodes.Language() will raise if invalid
        lang = langcodes.Language.get(code_lower)

        # Check if it's a real language by checking the display name
        # Invalid codes will have "Unknown language" in the display name
        display_name = lang.display_name("en")
        if display_name.startswith("Unknown language"):
            raise ValueError(
                f"Invalid ISO 639 language code: '{code}'. "
                f"Not recognized as a valid ISO 639-1 or ISO 639-3 code."
            )

        # Return the normalized form (lowercase)
        return code_lower
    except ImportError:
        # Fallback: basic validation if langcodes not available
        # Check if it's alpha-only
        if not code_lower.isalpha():
            raise ValueError(
                f"Invalid ISO 639 language code: '{code}'. Must contain only letters."
            ) from None
        return code_lower
    except Exception as e:
        # Catch any langcodes-related errors (e.g., LanguageTagError)
        if "langcodes" in str(type(e).__module__):
            raise ValueError(
                f"Invalid ISO 639 language code: '{code}'. "
                f"Not recognized as ISO 639-1 or ISO 639-3. Error: {e}"
            ) from e
        # Re-raise other exceptions
        raise


# Type alias with validation baked in
LanguageCode = Annotated[
    str | None,
    AfterValidator(validate_iso639_code),
    Field(
        description=(
            "ISO 639-1 (2-letter) or ISO 639-3 (3-letter) language code. "
            "Examples: 'en', 'eng', 'ko', 'kor', 'ig', 'ibo', 'mr', 'mar', 'zu', 'zul'"
        )
    ),
]
