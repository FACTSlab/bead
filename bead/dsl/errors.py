"""DSL-specific exceptions."""

from __future__ import annotations


class DSLError(Exception):
    """Base exception for DSL errors."""

    pass


class ParseError(DSLError):
    """Exception raised when parsing fails.

    Parameters
    ----------
    message : str
        Error message.
    line : int | None
        Line number where error occurred.
    column : int | None
        Column number where error occurred.
    text : str | None
        Text that caused the error.
    """

    def __init__(
        self,
        message: str,
        line: int | None = None,
        column: int | None = None,
        text: str | None = None,
    ) -> None:
        self.line = line
        self.column = column
        self.text = text
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [super().__str__()]
        if self.line is not None:
            parts.append(f" at line {self.line}")
        if self.column is not None:
            parts.append(f", column {self.column}")
        if self.text is not None:
            parts.append(f"\n  {self.text}")
        return "".join(parts)


class EvaluationError(DSLError):
    """Exception raised when evaluation fails."""

    pass
