"""Tests for DSL exception classes."""

from __future__ import annotations

import pytest

from sash.dsl.errors import DSLError, EvaluationError, ParseError


def test_dsl_error_creation() -> None:
    """Test DSLError can be created and raised."""
    error = DSLError("test error")
    assert str(error) == "test error"
    assert isinstance(error, Exception)


def test_dsl_error_raise() -> None:
    """Test DSLError can be raised."""
    with pytest.raises(DSLError, match="test error"):
        raise DSLError("test error")


def test_parse_error_basic() -> None:
    """Test ParseError with just message."""
    error = ParseError("syntax error")
    assert str(error) == "syntax error"
    assert error.line is None
    assert error.column is None
    assert error.text is None


def test_parse_error_with_location() -> None:
    """Test ParseError with line and column info."""
    error = ParseError("syntax error", line=10, column=5)
    assert error.line == 10
    assert error.column == 5
    assert "at line 10" in str(error)
    assert "column 5" in str(error)


def test_parse_error_with_text() -> None:
    """Test ParseError with source text."""
    error = ParseError("syntax error", text="pos == 'VERB")
    assert error.text == "pos == 'VERB"
    assert "pos == 'VERB" in str(error)


def test_parse_error_full() -> None:
    """Test ParseError with all information."""
    error = ParseError(
        "unexpected token",
        line=2,
        column=10,
        text="lemma == 'walk' and",
    )
    error_str = str(error)
    assert "unexpected token" in error_str
    assert "at line 2" in error_str
    assert "column 10" in error_str
    assert "lemma == 'walk' and" in error_str


def test_evaluation_error_creation() -> None:
    """Test EvaluationError can be created."""
    error = EvaluationError("evaluation failed")
    assert str(error) == "evaluation failed"
    assert isinstance(error, DSLError)


def test_evaluation_error_raise() -> None:
    """Test EvaluationError can be raised."""
    with pytest.raises(EvaluationError, match="evaluation failed"):
        raise EvaluationError("evaluation failed")


def test_error_inheritance() -> None:
    """Test error inheritance chain."""
    # ParseError inherits from DSLError
    assert issubclass(ParseError, DSLError)
    assert issubclass(ParseError, Exception)

    # EvaluationError inherits from DSLError
    assert issubclass(EvaluationError, DSLError)
    assert issubclass(EvaluationError, Exception)

    # DSLError inherits from Exception
    assert issubclass(DSLError, Exception)


def test_parse_error_catch_as_dsl_error() -> None:
    """Test ParseError can be caught as DSLError."""
    with pytest.raises(DSLError):
        raise ParseError("test")


def test_evaluation_error_catch_as_dsl_error() -> None:
    """Test EvaluationError can be caught as DSLError."""
    with pytest.raises(DSLError):
        raise EvaluationError("test")
