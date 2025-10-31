"""Tests for standard library functions."""

from __future__ import annotations

from bead.dsl import STDLIB_FUNCTIONS, EvaluationContext, register_stdlib, stdlib


# String functions tests
def test_len_string() -> None:
    """Test len function with string."""
    assert stdlib.len_("hello") == 5


def test_len_list() -> None:
    """Test len function with list."""
    assert stdlib.len_([1, 2, 3]) == 3


def test_len_empty() -> None:
    """Test len function with empty string."""
    assert stdlib.len_("") == 0


def test_lower() -> None:
    """Test lower function."""
    assert stdlib.lower("HELLO") == "hello"
    assert stdlib.lower("Hello World") == "hello world"


def test_upper() -> None:
    """Test upper function."""
    assert stdlib.upper("hello") == "HELLO"
    assert stdlib.upper("Hello World") == "HELLO WORLD"


def test_startswith_true() -> None:
    """Test startswith function returns True."""
    assert stdlib.startswith("hello", "hel") is True


def test_startswith_false() -> None:
    """Test startswith function returns False."""
    assert stdlib.startswith("hello", "bye") is False


def test_endswith_true() -> None:
    """Test endswith function returns True."""
    assert stdlib.endswith("hello", "lo") is True


def test_endswith_false() -> None:
    """Test endswith function returns False."""
    assert stdlib.endswith("hello", "hi") is False


def test_contains_true() -> None:
    """Test contains function returns True."""
    assert stdlib.contains("hello", "ell") is True


def test_contains_false() -> None:
    """Test contains function returns False."""
    assert stdlib.contains("hello", "bye") is False


def test_replace() -> None:
    """Test replace function."""
    assert stdlib.replace("hello world", "world", "there") == "hello there"


def test_split() -> None:
    """Test split function."""
    assert stdlib.split("a,b,c", ",") == ["a", "b", "c"]
    assert stdlib.split("hello world") == ["hello", "world"]


# Collection functions tests
def test_count() -> None:
    """Test count function."""
    assert stdlib.count([1, 2, 2, 3], 2) == 2
    assert stdlib.count("hello", "l") == 2


def test_sum() -> None:
    """Test sum function."""
    assert stdlib.sum_([1, 2, 3]) == 6
    assert stdlib.sum_([1.5, 2.5]) == 4.0


def test_sum_empty() -> None:
    """Test sum function with empty list."""
    assert stdlib.sum_([]) == 0


def test_min() -> None:
    """Test min function."""
    assert stdlib.min_([3, 1, 2]) == 1


def test_max() -> None:
    """Test max function."""
    assert stdlib.max_([3, 1, 2]) == 3


def test_any_true() -> None:
    """Test any function returns True."""
    assert stdlib.any_([False, True, False]) is True


def test_any_false() -> None:
    """Test any function returns False."""
    assert stdlib.any_([False, False]) is False


def test_all_true() -> None:
    """Test all function returns True."""
    assert stdlib.all_([True, True, True]) is True


def test_all_false() -> None:
    """Test all function returns False."""
    assert stdlib.all_([True, False, True]) is False


# Type checking functions tests
def test_is_str_true() -> None:
    """Test is_str returns True for string."""
    assert stdlib.is_str("hello") is True


def test_is_str_false() -> None:
    """Test is_str returns False for non-string."""
    assert stdlib.is_str(42) is False


def test_is_int_true() -> None:
    """Test is_int returns True for integer."""
    assert stdlib.is_int(42) is True


def test_is_int_false_for_float() -> None:
    """Test is_int returns False for float."""
    assert stdlib.is_int(42.0) is False


def test_is_int_false_for_bool() -> None:
    """Test is_int returns False for boolean."""
    assert stdlib.is_int(True) is False


def test_is_float_true() -> None:
    """Test is_float returns True for float."""
    assert stdlib.is_float(42.0) is True


def test_is_float_false() -> None:
    """Test is_float returns False for non-float."""
    assert stdlib.is_float(42) is False


def test_is_bool_true() -> None:
    """Test is_bool returns True for boolean."""
    assert stdlib.is_bool(True) is True
    assert stdlib.is_bool(False) is True


def test_is_bool_false() -> None:
    """Test is_bool returns False for non-boolean."""
    assert stdlib.is_bool(1) is False


def test_is_list_true() -> None:
    """Test is_list returns True for list."""
    assert stdlib.is_list([1, 2, 3]) is True


def test_is_list_false() -> None:
    """Test is_list returns False for non-list."""
    assert stdlib.is_list((1, 2, 3)) is False


# Math functions tests
def test_abs_positive() -> None:
    """Test abs function with positive number."""
    assert stdlib.abs_(5) == 5


def test_abs_negative() -> None:
    """Test abs function with negative number."""
    assert stdlib.abs_(-5) == 5


def test_round() -> None:
    """Test round function."""
    assert stdlib.round_(3.14159, 2) == 3.14
    assert stdlib.round_(3.5) == 4.0


def test_floor() -> None:
    """Test floor function."""
    assert stdlib.floor(3.7) == 3
    assert stdlib.floor(-3.7) == -4


def test_ceil() -> None:
    """Test ceil function."""
    assert stdlib.ceil(3.2) == 4
    assert stdlib.ceil(-3.2) == -3


# Logic functions tests
def test_not_true() -> None:
    """Test not function with True."""
    assert stdlib.not_(True) is False


def test_not_false() -> None:
    """Test not function with False."""
    assert stdlib.not_(False) is True


def test_not_truthy() -> None:
    """Test not function with truthy value."""
    assert stdlib.not_(1) is False
    assert stdlib.not_("hello") is False


def test_not_falsy() -> None:
    """Test not function with falsy value."""
    assert stdlib.not_(0) is True
    assert stdlib.not_("") is True


# Registry tests
def test_stdlib_functions_registry() -> None:
    """Test STDLIB_FUNCTIONS registry has all functions."""
    expected_functions = [
        "len",
        "lower",
        "upper",
        "startswith",
        "endswith",
        "contains",
        "replace",
        "split",
        "count",
        "sum",
        "min",
        "max",
        "any",
        "all",
        "is_str",
        "is_int",
        "is_float",
        "is_bool",
        "is_list",
        "abs",
        "round",
        "floor",
        "ceil",
        "not",
    ]
    for func_name in expected_functions:
        assert func_name in STDLIB_FUNCTIONS


def test_register_stdlib() -> None:
    """Test register_stdlib registers all functions."""
    ctx = EvaluationContext()
    register_stdlib(ctx)
    assert ctx.has_function("len")
    assert ctx.has_function("lower")
    assert ctx.has_function("sum")
    assert ctx.has_function("is_str")
    assert ctx.has_function("abs")
    assert ctx.has_function("not")


def test_register_stdlib_function_callable() -> None:
    """Test registered functions are callable."""
    ctx = EvaluationContext()
    register_stdlib(ctx)
    result = ctx.call_function("len", ["hello"])
    assert result == 5


# Edge cases tests
def test_unicode_strings() -> None:
    """Test functions work with unicode strings."""
    assert stdlib.len_("héllo") == 5
    assert stdlib.upper("héllo") == "HÉLLO"
    assert stdlib.lower("HÉLLO") == "héllo"


def test_empty_list_operations() -> None:
    """Test operations on empty lists."""
    assert stdlib.len_([]) == 0
    assert stdlib.sum_([]) == 0
    assert stdlib.any_([]) is False
    assert stdlib.all_([]) is True


def test_split_with_default_separator() -> None:
    """Test split with default separator."""
    assert stdlib.split("hello world") == ["hello", "world"]


def test_count_no_occurrences() -> None:
    """Test count with no occurrences."""
    assert stdlib.count([1, 2, 3], 4) == 0
    assert stdlib.count("hello", "x") == 0
