"""Tests for DSL parser."""

from __future__ import annotations

import pytest

from bead.dsl import ast, parse
from bead.dsl.errors import ParseError


# Literals
def test_parse_string_literal_double_quotes() -> None:
    """Parse string literal with double quotes."""
    node = parse('"hello"')
    assert isinstance(node, ast.Literal)
    assert node.value == "hello"


def test_parse_string_literal_single_quotes() -> None:
    """Parse string literal with single quotes."""
    node = parse("'world'")
    assert isinstance(node, ast.Literal)
    assert node.value == "world"


def test_parse_integer_literal() -> None:
    """Parse integer literal."""
    node = parse("42")
    assert isinstance(node, ast.Literal)
    assert node.value == 42
    assert isinstance(node.value, int)


def test_parse_negative_integer() -> None:
    """Parse negative integer literal."""
    node = parse("-42")
    assert isinstance(node, ast.UnaryOp)
    assert node.operator == "-"
    assert isinstance(node.operand, ast.Literal)
    assert node.operand.value == 42


def test_parse_float_literal() -> None:
    """Parse float literal."""
    node = parse("3.14")
    assert isinstance(node, ast.Literal)
    assert node.value == 3.14
    assert isinstance(node.value, float)


def test_parse_boolean_true() -> None:
    """Parse boolean true."""
    node = parse("true")
    assert isinstance(node, ast.Literal)
    assert node.value is True


def test_parse_boolean_false() -> None:
    """Parse boolean false."""
    node = parse("false")
    assert isinstance(node, ast.Literal)
    assert node.value is False


# Variables
def test_parse_simple_variable() -> None:
    """Parse simple variable."""
    node = parse("lemma")
    assert isinstance(node, ast.Variable)
    assert node.name == "lemma"


def test_parse_variable_with_underscore() -> None:
    """Parse variable with underscores."""
    node = parse("is_transitive")
    assert isinstance(node, ast.Variable)
    assert node.name == "is_transitive"


def test_parse_variable_with_numbers() -> None:
    """Parse variable with numbers."""
    node = parse("var123")
    assert isinstance(node, ast.Variable)
    assert node.name == "var123"


# Operators
def test_parse_equality_operator() -> None:
    """Parse equality operator."""
    node = parse("pos == 'VERB'")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "=="
    assert isinstance(node.left, ast.Variable)
    assert node.left.name == "pos"
    assert isinstance(node.right, ast.Literal)
    assert node.right.value == "VERB"


def test_parse_inequality_operator() -> None:
    """Parse inequality operator."""
    node = parse("x != 5")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "!="


def test_parse_less_than() -> None:
    """Parse less than operator."""
    node = parse("x < 10")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "<"


def test_parse_greater_than() -> None:
    """Parse greater than operator."""
    node = parse("x > 10")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == ">"


def test_parse_less_than_or_equal() -> None:
    """Parse less than or equal operator."""
    node = parse("x <= 10")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "<="


def test_parse_greater_than_or_equal() -> None:
    """Parse greater than or equal operator."""
    node = parse("x >= 10")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == ">="


def test_parse_logical_and() -> None:
    """Parse logical and operator."""
    node = parse("x and y")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "and"
    assert isinstance(node.left, ast.Variable)
    assert isinstance(node.right, ast.Variable)


def test_parse_logical_or() -> None:
    """Parse logical or operator."""
    node = parse("x or y")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "or"


def test_parse_logical_not() -> None:
    """Parse logical not operator."""
    node = parse("not x")
    assert isinstance(node, ast.UnaryOp)
    assert node.operator == "not"
    assert isinstance(node.operand, ast.Variable)


def test_parse_in_operator() -> None:
    """Parse in operator."""
    node = parse("x in [1, 2, 3]")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "in"


def test_parse_not_in_operator() -> None:
    """Parse not in operator."""
    node = parse("x not in [1, 2, 3]")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "not in"
    assert isinstance(node.left, ast.Variable)
    assert isinstance(node.right, ast.ListLiteral)


# Arithmetic
def test_parse_addition() -> None:
    """Parse addition."""
    node = parse("x + 5")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "+"


def test_parse_subtraction() -> None:
    """Parse subtraction."""
    node = parse("x - 5")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "-"


def test_parse_multiplication() -> None:
    """Parse multiplication."""
    node = parse("x * 5")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "*"


def test_parse_division() -> None:
    """Parse division."""
    node = parse("x / 5")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "/"


def test_parse_modulo() -> None:
    """Parse modulo."""
    node = parse("x % 5")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "%"


def test_parse_unary_minus() -> None:
    """Parse unary minus."""
    node = parse("-x")
    assert isinstance(node, ast.UnaryOp)
    assert node.operator == "-"


def test_parse_unary_plus() -> None:
    """Parse unary plus."""
    node = parse("+x")
    assert isinstance(node, ast.UnaryOp)
    assert node.operator == "+"


def test_parse_arithmetic_precedence() -> None:
    """Parse arithmetic with correct precedence."""
    # 2 + 3 * 4 should parse as 2 + (3 * 4)
    node = parse("2 + 3 * 4")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "+"
    assert isinstance(node.left, ast.Literal)
    assert node.left.value == 2
    assert isinstance(node.right, ast.BinaryOp)
    assert node.right.operator == "*"


# Complex expressions
def test_parse_nested_boolean() -> None:
    """Parse nested boolean expressions."""
    node = parse("(x and y) or z")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "or"
    assert isinstance(node.left, ast.BinaryOp)
    assert node.left.operator == "and"


def test_parse_function_call_no_args() -> None:
    """Parse function call with no arguments."""
    node = parse("now()")
    assert isinstance(node, ast.FunctionCall)
    assert node.function.name == "now"
    assert len(node.arguments) == 0


def test_parse_function_call_one_arg() -> None:
    """Parse function call with one argument."""
    node = parse("len(lemma)")
    assert isinstance(node, ast.FunctionCall)
    assert node.function.name == "len"
    assert len(node.arguments) == 1
    assert isinstance(node.arguments[0], ast.Variable)


def test_parse_function_call_multiple_args() -> None:
    """Parse function call with multiple arguments."""
    node = parse("substr(text, 0, 5)")
    assert isinstance(node, ast.FunctionCall)
    assert node.function.name == "substr"
    assert len(node.arguments) == 3


def test_parse_attribute_access() -> None:
    """Parse attribute access."""
    node = parse("item.lemma")
    assert isinstance(node, ast.AttributeAccess)
    assert isinstance(node.object, ast.Variable)
    assert node.object.name == "item"
    assert node.attribute == "lemma"


def test_parse_nested_attribute_access() -> None:
    """Parse nested attribute access."""
    node = parse("obj.attr1.attr2")
    assert isinstance(node, ast.AttributeAccess)
    assert node.attribute == "attr2"
    assert isinstance(node.object, ast.AttributeAccess)
    assert node.object.attribute == "attr1"


def test_parse_list_literal_empty() -> None:
    """Parse empty list literal."""
    node = parse("[]")
    assert isinstance(node, ast.ListLiteral)
    assert len(node.elements) == 0


def test_parse_list_literal_with_elements() -> None:
    """Parse list literal with elements."""
    node = parse('["a", "b", "c"]')
    assert isinstance(node, ast.ListLiteral)
    assert len(node.elements) == 3
    assert all(isinstance(e, ast.Literal) for e in node.elements)


def test_parse_complex_nested_expression() -> None:
    """Parse complex nested expression."""
    # (pos == "VERB" and len(lemma) > 3) or transitive == true
    expr = '(pos == "VERB" and len(lemma) > 3) or transitive == true'
    node = parse(expr)
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "or"
    assert isinstance(node.left, ast.BinaryOp)
    assert node.left.operator == "and"
    assert isinstance(node.right, ast.BinaryOp)


def test_parse_expression_with_comment() -> None:
    """Parse expression with comment."""
    node = parse("x == 5  # this is a comment")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "=="


def test_parse_expression_with_whitespace() -> None:
    """Parse expression with extra whitespace."""
    node = parse("  x   ==   5  ")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "=="


def test_parse_parentheses() -> None:
    """Parse expression with parentheses."""
    node = parse("(x + y) * z")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "*"
    assert isinstance(node.left, ast.BinaryOp)
    assert node.left.operator == "+"


def test_parse_operator_precedence_bool() -> None:
    """Parse boolean operators with correct precedence."""
    # x or y and z should parse as x or (y and z)
    node = parse("x or y and z")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "or"
    assert isinstance(node.right, ast.BinaryOp)
    assert node.right.operator == "and"


def test_parse_comparison_chain() -> None:
    """Parse comparison chains."""
    # a < b < c parses as (a < b) < c
    node = parse("a < b < c")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "<"
    assert isinstance(node.left, ast.BinaryOp)
    assert node.left.operator == "<"


# Error handling
def test_parse_error_invalid_syntax() -> None:
    """Parse error for invalid syntax."""
    with pytest.raises(ParseError):
        parse("==")


def test_parse_error_unmatched_parentheses() -> None:
    """Parse error for unmatched parentheses."""
    with pytest.raises(ParseError):
        parse("(x + y")


def test_parse_error_invalid_operator() -> None:
    """Parse error for invalid operator."""
    with pytest.raises(ParseError):
        parse("x @ y")


def test_parse_error_includes_location() -> None:
    """Parse error includes line and column info."""
    try:
        parse("x ==")
        pytest.fail("Should have raised ParseError")
    except ParseError as e:
        # Error should have location info
        assert e.line is not None or e.text is not None


def test_parse_empty_string() -> None:
    """Parse error for empty string."""
    with pytest.raises(ParseError):
        parse("")


def test_parse_only_whitespace() -> None:
    """Parse error for only whitespace."""
    with pytest.raises(ParseError):
        parse("   ")


def test_parse_unclosed_string() -> None:
    """Parse error for unclosed string."""
    with pytest.raises(ParseError):
        parse('"hello')


def test_parse_function_in_expression() -> None:
    """Parse function call in binary expression."""
    node = parse("len(lemma) > 5")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == ">"
    assert isinstance(node.left, ast.FunctionCall)
    assert node.left.function.name == "len"


def test_parse_attribute_in_comparison() -> None:
    """Parse attribute access in comparison."""
    node = parse("item.pos == 'VERB'")
    assert isinstance(node, ast.BinaryOp)
    assert node.operator == "=="
    assert isinstance(node.left, ast.AttributeAccess)
