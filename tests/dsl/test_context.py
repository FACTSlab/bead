"""Tests for EvaluationContext."""

from __future__ import annotations

import pytest

from bead.dsl import EvaluationContext
from bead.dsl.errors import EvaluationError


def test_context_creation() -> None:
    """Test context can be created."""
    ctx = EvaluationContext()
    assert ctx is not None


def test_context_with_parent() -> None:
    """Test context can be created with parent."""
    parent = EvaluationContext()
    child = EvaluationContext(parent=parent)
    assert child is not None


def test_set_and_get_variable() -> None:
    """Test setting and getting a variable."""
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    assert ctx.get_variable("x") == 42


def test_get_undefined_variable_raises_error() -> None:
    """Test getting undefined variable raises error."""
    ctx = EvaluationContext()
    with pytest.raises(EvaluationError, match="Undefined variable: x"):
        ctx.get_variable("x")


def test_variable_in_parent_context() -> None:
    """Test getting variable from parent context."""
    parent = EvaluationContext()
    parent.set_variable("x", 42)
    child = EvaluationContext(parent=parent)
    assert child.get_variable("x") == 42


def test_variable_shadowing_in_child_context() -> None:
    """Test variable shadowing in child context."""
    parent = EvaluationContext()
    parent.set_variable("x", 42)
    child = EvaluationContext(parent=parent)
    child.set_variable("x", 100)
    assert child.get_variable("x") == 100
    assert parent.get_variable("x") == 42


def test_has_variable_true() -> None:
    """Test has_variable returns True for defined variable."""
    ctx = EvaluationContext()
    ctx.set_variable("x", 42)
    assert ctx.has_variable("x") is True


def test_has_variable_false() -> None:
    """Test has_variable returns False for undefined variable."""
    ctx = EvaluationContext()
    assert ctx.has_variable("x") is False


def test_has_variable_in_parent() -> None:
    """Test has_variable checks parent context."""
    parent = EvaluationContext()
    parent.set_variable("x", 42)
    child = EvaluationContext(parent=parent)
    assert child.has_variable("x") is True


def test_set_and_call_function() -> None:
    """Test setting and calling a function."""
    ctx = EvaluationContext()
    ctx.set_function("double", lambda x: x * 2)
    result = ctx.call_function("double", [5])
    assert result == 10


def test_call_undefined_function_raises_error() -> None:
    """Test calling undefined function raises error."""
    ctx = EvaluationContext()
    with pytest.raises(EvaluationError, match="Undefined function: foo"):
        ctx.call_function("foo", [])


def test_function_in_parent_context() -> None:
    """Test calling function from parent context."""
    parent = EvaluationContext()
    parent.set_function("double", lambda x: x * 2)
    child = EvaluationContext(parent=parent)
    result = child.call_function("double", [5])
    assert result == 10


def test_function_shadowing_in_child_context() -> None:
    """Test function shadowing in child context."""
    parent = EvaluationContext()
    parent.set_function("double", lambda x: x * 2)
    child = EvaluationContext(parent=parent)
    child.set_function("double", lambda x: x * 3)
    assert child.call_function("double", [5]) == 15
    assert parent.call_function("double", [5]) == 10


def test_has_function_true() -> None:
    """Test has_function returns True for defined function."""
    ctx = EvaluationContext()
    ctx.set_function("double", lambda x: x * 2)
    assert ctx.has_function("double") is True


def test_has_function_false() -> None:
    """Test has_function returns False for undefined function."""
    ctx = EvaluationContext()
    assert ctx.has_function("foo") is False


def test_has_function_in_parent() -> None:
    """Test has_function checks parent context."""
    parent = EvaluationContext()
    parent.set_function("double", lambda x: x * 2)
    child = EvaluationContext(parent=parent)
    assert child.has_function("double") is True


def test_create_child_context() -> None:
    """Test creating child context."""
    parent = EvaluationContext()
    parent.set_variable("x", 42)
    child = parent.create_child()
    assert child.get_variable("x") == 42
    child.set_variable("y", 20)
    assert child.get_variable("y") == 20


def test_parent_chain_lookup() -> None:
    """Test lookup through parent chain."""
    grandparent = EvaluationContext()
    grandparent.set_variable("x", 1)
    parent = EvaluationContext(parent=grandparent)
    parent.set_variable("y", 2)
    child = EvaluationContext(parent=parent)
    child.set_variable("z", 3)
    assert child.get_variable("x") == 1
    assert child.get_variable("y") == 2
    assert child.get_variable("z") == 3


def test_multiple_context_levels() -> None:
    """Test multiple context levels with shadowing."""
    level1 = EvaluationContext()
    level1.set_variable("x", 1)
    level2 = EvaluationContext(parent=level1)
    level2.set_variable("x", 2)
    level3 = EvaluationContext(parent=level2)
    level3.set_variable("x", 3)
    assert level3.get_variable("x") == 3
    assert level2.get_variable("x") == 2
    assert level1.get_variable("x") == 1


def test_context_isolation() -> None:
    """Test contexts are isolated from each other."""
    ctx1 = EvaluationContext()
    ctx1.set_variable("x", 1)
    ctx2 = EvaluationContext()
    ctx2.set_variable("x", 2)
    assert ctx1.get_variable("x") == 1
    assert ctx2.get_variable("x") == 2


def test_function_call_with_multiple_args() -> None:
    """Test calling function with multiple arguments."""
    ctx = EvaluationContext()
    ctx.set_function("add", lambda x, y: x + y)
    result = ctx.call_function("add", [3, 4])
    assert result == 7


def test_function_call_wrong_arg_count_raises_error() -> None:
    """Test calling function with wrong number of arguments raises error."""
    ctx = EvaluationContext()
    ctx.set_function("double", lambda x: x * 2)
    with pytest.raises(EvaluationError, match="Function call failed"):
        ctx.call_function("double", [1, 2])


def test_set_variable_with_none_value() -> None:
    """Test setting variable with None value."""
    ctx = EvaluationContext()
    ctx.set_variable("x", None)
    assert ctx.get_variable("x") is None


def test_set_variable_with_complex_types() -> None:
    """Test setting variable with complex types."""
    ctx = EvaluationContext()
    ctx.set_variable("list", [1, 2, 3])
    ctx.set_variable("dict", {"a": 1, "b": 2})
    ctx.set_variable("tuple", (1, 2, 3))
    assert ctx.get_variable("list") == [1, 2, 3]
    assert ctx.get_variable("dict") == {"a": 1, "b": 2}
    assert ctx.get_variable("tuple") == (1, 2, 3)
