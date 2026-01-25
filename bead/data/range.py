"""Generic numeric range model with validation.

Provides a reusable Range[T] model for representing validated numeric ranges
with bounds checking, containment testing, and value clamping.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

T = TypeVar("T", int, float)


class Range(BaseModel, Generic[T]):  # noqa: UP046 - Pydantic requires Generic[T]
    """A validated numeric range with inclusive bounds.

    Provides a generic container for numeric ranges with automatic validation
    that min < max. Supports containment testing and value clamping.

    Attributes
    ----------
    min
        Minimum value (inclusive).
    max
        Maximum value (inclusive).

    Examples
    --------
    >>> scale = Range[int](min=1, max=7)
    >>> scale.contains(4)
    True
    >>> scale.contains(0)
    False
    >>> scale.clamp(10)
    7

    >>> probability = Range[float](min=0.0, max=1.0)
    >>> probability.contains(0.5)
    True
    >>> probability.clamp(-0.1)
    0.0
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    min: T
    max: T

    @model_validator(mode="after")
    def validate_order(self) -> Range[T]:
        """Validate that min is strictly less than max.

        Returns
        -------
        Range[T]
            The validated range instance.

        Raises
        ------
        ValueError
            If min is greater than or equal to max.
        """
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be less than max ({self.max})")
        return self

    def contains(self, value: T) -> bool:
        """Check if a value is within the range (inclusive).

        Parameters
        ----------
        value
            The value to check.

        Returns
        -------
        bool
            True if min <= value <= max, False otherwise.

        Examples
        --------
        >>> r = Range[int](min=1, max=5)
        >>> r.contains(3)
        True
        >>> r.contains(1)
        True
        >>> r.contains(5)
        True
        >>> r.contains(6)
        False
        """
        return self.min <= value <= self.max

    def clamp(self, value: T) -> T:
        """Clamp a value to the range bounds.

        Parameters
        ----------
        value
            The value to clamp.

        Returns
        -------
        T
            The clamped value (min if value < min, max if value > max,
            otherwise the original value).

        Examples
        --------
        >>> r = Range[int](min=1, max=5)
        >>> r.clamp(3)
        3
        >>> r.clamp(0)
        1
        >>> r.clamp(10)
        5
        """
        return max(self.min, min(self.max, value))
