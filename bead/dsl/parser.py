"""Constraint DSL parser.

This module provides the parser for constraint expressions using the Lark
parsing library. The parser converts constraint strings into AST nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer
from lark.exceptions import LarkError, UnexpectedCharacters, UnexpectedInput

from bead.dsl import ast
from bead.dsl.errors import ParseError

# Load grammar from file
_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
_GRAMMAR = _GRAMMAR_PATH.read_text()

# Create Lark parser
_PARSER = Lark(
    _GRAMMAR,
    start="start",
    parser="lalr",  # Fast LALR parser
    propagate_positions=True,  # Track line/column info
    keep_all_tokens=True,  # Keep operator tokens
)


class ASTBuilder(Transformer):  # type: ignore[type-arg]
    """Transformer that converts Lark parse tree to AST nodes."""

    def string_literal(self, items: list[Token]) -> ast.Literal:
        """Transform string literal."""
        token = items[0]
        # Remove quotes
        value = str(token.value)[1:-1]
        return ast.Literal(value=value)

    def number_literal(self, items: list[Token]) -> ast.Literal:
        """Transform number literal."""
        token = items[0]
        value_str = str(token.value)
        # Parse as int or float
        value: int | float = (
            int(value_str) if "." not in value_str else float(value_str)
        )
        return ast.Literal(value=value)

    def true_literal(self, items: list[Token]) -> ast.Literal:
        """Transform true literal."""
        return ast.Literal(value=True)

    def false_literal(self, items: list[Token]) -> ast.Literal:
        """Transform false literal."""
        return ast.Literal(value=False)

    def variable(self, items: list[Token]) -> ast.Variable:
        """Transform variable reference."""
        name = str(items[0].value)
        return ast.Variable(name=name)

    def binary_op(self, items: list[Any]) -> ast.BinaryOp:
        """Transform binary operation."""
        # Items: [left, operator_token, right]
        left = items[0]
        # Get operator from token
        operator_token = items[1]
        if isinstance(operator_token, Token):
            operator = str(operator_token.value)
        else:
            operator = str(operator_token)
        right = items[2]
        return ast.BinaryOp(operator=operator, left=left, right=right)

    def binary_op_not_in(self, items: list[Any]) -> ast.BinaryOp:
        """Transform 'not in' binary operation."""
        # Items: [left, Token('not'), Token('in'), right]
        # Filter to get only AST nodes
        nodes = [item for item in items if not isinstance(item, Token)]
        left = nodes[0]
        right = nodes[1]
        return ast.BinaryOp(operator="not in", left=left, right=right)

    def unary_op(self, items: list[Any]) -> ast.UnaryOp:
        """Transform unary operation."""
        # Items: [operator_token, operand]
        operator_token = items[0]
        if isinstance(operator_token, Token):
            operator = str(operator_token.value)
        else:
            operator = str(operator_token)
        operand = items[1]
        return ast.UnaryOp(operator=operator, operand=operand)

    def attribute_access(self, items: list[Any]) -> ast.AttributeAccess:
        """Transform attribute access."""
        # Items: [object, dot_token, name_token]
        obj = items[0]
        # Last token is the attribute name
        attribute = str(items[-1].value)
        return ast.AttributeAccess(object=obj, attribute=attribute)

    def subscript(self, items: list[Any]) -> ast.Subscript:
        """Transform subscript access."""
        # Items: [object, lbracket_token, index_expr, rbracket_token]
        obj = items[0]
        # The index expression is items[1] (non-Token items)
        index_expr = [item for item in items[1:] if not isinstance(item, Token)][0]
        return ast.Subscript(object=obj, index=index_expr)

    def function_call(self, items: list[Any]) -> ast.FunctionCall:
        """Transform function call."""
        # Items: [atom, lparen, arguments_list/None, rparen]
        # The first item is the function expression (can be Variable or AttributeAccess)
        function = items[0]

        # Arguments are in the list returned by the arguments rule (if present)
        # Filter for non-Token, non-None items and flatten lists
        arguments = []
        for item in items[1:]:
            if isinstance(item, list):
                # This is the arguments list
                arguments.extend(item)  # type: ignore[arg-type]
            elif not isinstance(item, Token) and item is not None:
                arguments.append(item)  # type: ignore[arg-type]
        return ast.FunctionCall(function=function, arguments=arguments)  # type: ignore[arg-type]

    def list_literal(self, items: list[Any]) -> ast.ListLiteral:
        """Transform list literal."""
        # Filter out bracket tokens and None, flatten lists
        elements = []
        for item in items:
            if isinstance(item, list):
                # This is the list_elements list
                elements.extend(item)  # type: ignore[arg-type]
            elif not isinstance(item, Token) and item is not None:
                elements.append(item)  # type: ignore[arg-type]
        return ast.ListLiteral(elements=elements)  # type: ignore[arg-type]

    def arguments(self, items: list[Any]) -> list[Any]:
        """Transform function arguments, returning flat list."""
        # Filter out comma tokens
        return [item for item in items if not isinstance(item, Token)]

    def list_elements(self, items: list[Any]) -> list[Any]:
        """Transform list elements, returning flat list."""
        # Filter out comma tokens
        return [item for item in items if not isinstance(item, Token)]

    def atom(self, items: list[Any]) -> ast.ASTNode:
        """Transform atom (handles parenthesized expressions)."""
        # If we have parentheses, items = [lparen_token, expr, rparen_token]
        # Return just the expression (filter out tokens)
        nodes = [item for item in items if not isinstance(item, Token)]
        return nodes[0] if nodes else items[0]


def parse(expression: str) -> ast.ASTNode:
    """Parse constraint expression into AST.

    Parameters
    ----------
    expression : str
        Constraint expression to parse.

    Returns
    -------
    ast.ASTNode
        Root node of the abstract syntax tree.

    Raises
    ------
    ParseError
        If the expression cannot be parsed.

    Examples
    --------
    >>> node = parse("pos == 'VERB'")
    >>> isinstance(node, ast.BinaryOp)
    True
    >>> node.operator
    '=='
    """
    try:
        # Parse with Lark
        tree = _PARSER.parse(expression)  # type: ignore[no-untyped-call]

        # Transform to AST
        transformer = ASTBuilder()
        result: ast.Expression = transformer.transform(tree)  # type: ignore[no-untyped-call]

        return result  # type: ignore[return-value]

    except UnexpectedCharacters as e:
        # Handle invalid characters (must come before UnexpectedInput)
        raise ParseError(
            f"Unexpected character at position {e.column}",
            line=e.line,
            column=e.column,
            text=expression,
        ) from e
    except UnexpectedInput as e:
        # Handle UnexpectedToken (has .token) and UnexpectedEOF
        token_str = getattr(e, "token", "unexpected input")
        raise ParseError(
            f"Unexpected input: {token_str}",
            line=e.line,
            column=e.column,
            text=expression,
        ) from e
    except LarkError as e:
        raise ParseError(f"Parse error: {e}", text=expression) from e
