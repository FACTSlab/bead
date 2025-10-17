"""Pytest fixtures for DSL module tests."""

from __future__ import annotations

import pytest

from sash.dsl import ast


@pytest.fixture
def sample_literal_string() -> ast.Literal:
    """Provide sample string literal node."""
    return ast.Literal(value="hello")


@pytest.fixture
def sample_literal_int() -> ast.Literal:
    """Provide sample integer literal node."""
    return ast.Literal(value=42)


@pytest.fixture
def sample_variable() -> ast.Variable:
    """Provide sample variable node."""
    return ast.Variable(name="lemma")


@pytest.fixture
def sample_binary_op(
    sample_variable: ast.Variable, sample_literal_string: ast.Literal
) -> ast.BinaryOp:
    """Provide sample binary operation node."""
    return ast.BinaryOp(
        operator="==",
        left=sample_variable,
        right=sample_literal_string,
    )
