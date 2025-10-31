"""Tests for AST node classes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bead.dsl import ast


def test_literal_string() -> None:
    """Test Literal with string value."""
    node = ast.Literal(value="hello")
    assert node.value == "hello"
    assert isinstance(node, ast.ASTNode)


def test_literal_int() -> None:
    """Test Literal with integer value."""
    node = ast.Literal(value=42)
    assert node.value == 42
    assert isinstance(node.value, int)


def test_literal_float() -> None:
    """Test Literal with float value."""
    node = ast.Literal(value=3.14)
    assert node.value == 3.14
    assert isinstance(node.value, float)


def test_literal_bool_true() -> None:
    """Test Literal with boolean true value."""
    node = ast.Literal(value=True)
    assert node.value is True
    assert isinstance(node.value, bool)


def test_literal_bool_false() -> None:
    """Test Literal with boolean false value."""
    node = ast.Literal(value=False)
    assert node.value is False
    assert isinstance(node.value, bool)


def test_variable_creation() -> None:
    """Test Variable creation."""
    node = ast.Variable(name="lemma")
    assert node.name == "lemma"
    assert isinstance(node, ast.ASTNode)


def test_variable_with_underscore() -> None:
    """Test Variable with underscore in name."""
    node = ast.Variable(name="is_transitive")
    assert node.name == "is_transitive"


def test_binary_op_creation() -> None:
    """Test BinaryOp creation."""
    left = ast.Variable(name="pos")
    right = ast.Literal(value="VERB")
    node = ast.BinaryOp(operator="==", left=left, right=right)
    assert node.operator == "=="
    assert node.left == left
    assert node.right == right


def test_binary_op_nested() -> None:
    """Test BinaryOp with nested operands."""
    # (a == b) and (c == d)
    left_op = ast.BinaryOp(
        operator="==",
        left=ast.Variable(name="a"),
        right=ast.Variable(name="b"),
    )
    right_op = ast.BinaryOp(
        operator="==",
        left=ast.Variable(name="c"),
        right=ast.Variable(name="d"),
    )
    node = ast.BinaryOp(operator="and", left=left_op, right=right_op)
    assert node.operator == "and"
    assert isinstance(node.left, ast.BinaryOp)
    assert isinstance(node.right, ast.BinaryOp)


def test_unary_op_creation() -> None:
    """Test UnaryOp creation."""
    operand = ast.Variable(name="x")
    node = ast.UnaryOp(operator="not", operand=operand)
    assert node.operator == "not"
    assert node.operand == operand


def test_unary_op_minus() -> None:
    """Test UnaryOp with minus operator."""
    operand = ast.Literal(value=42)
    node = ast.UnaryOp(operator="-", operand=operand)
    assert node.operator == "-"
    assert node.operand.value == 42


def test_function_call_no_args() -> None:
    """Test FunctionCall with no arguments."""
    func = ast.Variable(name="now")
    node = ast.FunctionCall(function=func, arguments=[])
    assert node.function.name == "now"
    assert len(node.arguments) == 0


def test_function_call_one_arg() -> None:
    """Test FunctionCall with one argument."""
    func = ast.Variable(name="len")
    arg = ast.Variable(name="lemma")
    node = ast.FunctionCall(function=func, arguments=[arg])
    assert node.function.name == "len"
    assert len(node.arguments) == 1
    assert node.arguments[0] == arg


def test_function_call_multiple_args() -> None:
    """Test FunctionCall with multiple arguments."""
    func = ast.Variable(name="substring")
    args = [
        ast.Variable(name="text"),
        ast.Literal(value=0),
        ast.Literal(value=5),
    ]
    node = ast.FunctionCall(function=func, arguments=args)
    assert node.function.name == "substring"
    assert len(node.arguments) == 3


def test_list_literal_empty() -> None:
    """Test ListLiteral with no elements."""
    node = ast.ListLiteral(elements=[])
    assert len(node.elements) == 0


def test_list_literal_with_elements() -> None:
    """Test ListLiteral with elements."""
    elements = [
        ast.Literal(value="a"),
        ast.Literal(value="b"),
        ast.Literal(value="c"),
    ]
    node = ast.ListLiteral(elements=elements)
    assert len(node.elements) == 3
    assert node.elements[0].value == "a"
    assert node.elements[1].value == "b"
    assert node.elements[2].value == "c"


def test_attribute_access_creation() -> None:
    """Test AttributeAccess creation."""
    obj = ast.Variable(name="item")
    node = ast.AttributeAccess(object=obj, attribute="lemma")
    assert node.object == obj
    assert node.attribute == "lemma"


def test_attribute_access_nested() -> None:
    """Test AttributeAccess with nested object."""
    # obj.attr1.attr2
    inner = ast.AttributeAccess(
        object=ast.Variable(name="obj"),
        attribute="attr1",
    )
    outer = ast.AttributeAccess(object=inner, attribute="attr2")
    assert outer.attribute == "attr2"
    assert isinstance(outer.object, ast.AttributeAccess)


def test_ast_node_serialization() -> None:
    """Test AST node serialization to dict."""
    node = ast.Literal(value=42)
    data = node.model_dump()
    assert isinstance(data, dict)
    assert data["value"] == 42


def test_ast_node_deserialization() -> None:
    """Test AST node deserialization from dict."""
    data = {"value": "hello"}
    node = ast.Literal(**data)
    assert node.value == "hello"


def test_ast_node_validation_error() -> None:
    """Test AST node validation with invalid types."""
    with pytest.raises(ValidationError):
        # Variable requires 'name' field
        ast.Variable()  # type: ignore[call-arg]


def test_nested_ast_structure() -> None:
    """Test complex nested AST structure."""
    # (pos == "VERB" and len(lemma) > 3) or transitive == true
    left_left = ast.BinaryOp(
        operator="==",
        left=ast.Variable(name="pos"),
        right=ast.Literal(value="VERB"),
    )
    left_right = ast.BinaryOp(
        operator=">",
        left=ast.FunctionCall(
            function=ast.Variable(name="len"),
            arguments=[ast.Variable(name="lemma")],
        ),
        right=ast.Literal(value=3),
    )
    left = ast.BinaryOp(operator="and", left=left_left, right=left_right)

    right = ast.BinaryOp(
        operator="==",
        left=ast.Variable(name="transitive"),
        right=ast.Literal(value=True),
    )

    root = ast.BinaryOp(operator="or", left=left, right=right)

    assert root.operator == "or"
    assert isinstance(root.left, ast.BinaryOp)
    assert isinstance(root.right, ast.BinaryOp)


def test_ast_node_equality() -> None:
    """Test AST node equality."""
    node1 = ast.Literal(value=42)
    node2 = ast.Literal(value=42)
    # Pydantic models compare by value
    assert node1.value == node2.value


def test_ast_node_model_dump() -> None:
    """Test AST node model_dump method."""
    node = ast.BinaryOp(
        operator="==",
        left=ast.Variable(name="x"),
        right=ast.Literal(value=10),
    )
    data = node.model_dump()
    assert data["operator"] == "=="
    # Check that left and right are present and are dicts
    assert "left" in data
    assert "right" in data
    assert isinstance(data["left"], dict)
    assert isinstance(data["right"], dict)
    # Verify the nested nodes have base fields
    assert "id" in data["left"]
    assert "id" in data["right"]


def test_ast_node_model_copy() -> None:
    """Test AST node model_copy method."""
    original = ast.Variable(name="test")
    copy = original.model_copy()
    assert copy.name == original.name
    # They should be different objects but equal values
    assert copy is not original
    assert copy.name == original.name


def test_binary_op_all_operators() -> None:
    """Test BinaryOp with various operators."""
    operators = [
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "and",
        "or",
        "in",
        "not in",
        "+",
        "-",
        "*",
        "/",
        "%",
    ]
    left = ast.Variable(name="x")
    right = ast.Literal(value=5)

    for op in operators:
        node = ast.BinaryOp(operator=op, left=left, right=right)
        assert node.operator == op


def test_unary_op_all_operators() -> None:
    """Test UnaryOp with various operators."""
    operators = ["not", "-", "+"]
    operand = ast.Variable(name="x")

    for op in operators:
        node = ast.UnaryOp(operator=op, operand=operand)
        assert node.operator == op
