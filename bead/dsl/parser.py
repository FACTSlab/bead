"""Constraint DSL parser.

This module provides the parser for constraint expressions using the Lark
parsing library. The parser converts constraint strings into AST nodes.
"""

from __future__ import annotations

from pathlib import Path

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

    def binary_op(self, items: list[Token | ast.ASTNode]) -> ast.BinaryOp:
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

    def binary_op_not_in(self, items: list[Token | ast.ASTNode]) -> ast.BinaryOp:
        """Transform 'not in' binary operation."""
        # Items: [left, Token('not'), Token('in'), right]
        # Filter to get only AST nodes
        nodes = [item for item in items if not isinstance(item, Token)]
        left = nodes[0]
        right = nodes[1]
        return ast.BinaryOp(operator="not in", left=left, right=right)

    def unary_op(self, items: list[Token | ast.ASTNode]) -> ast.UnaryOp:
        """Transform unary operation."""
        # Items: [operator_token, operand]
        operator_token = items[0]
        if isinstance(operator_token, Token):
            operator = str(operator_token.value)
        else:
            operator = str(operator_token)
        operand = items[1]
        return ast.UnaryOp(operator=operator, operand=operand)

    def attribute_access(self, items: list[Token | ast.ASTNode]) -> ast.AttributeAccess:
        """Transform attribute access."""
        # Items: [object, dot_token, name_token]
        obj = items[0]
        # Last token is the attribute name
        attribute = str(items[-1].value)
        return ast.AttributeAccess(object=obj, attribute=attribute)

    def subscript(self, items: list[Token | ast.ASTNode]) -> ast.Subscript:
        """Transform subscript access."""
        # Items: [object, lbracket_token, index_expr, rbracket_token]
        obj = items[0]
        # The index expression is items[1] (non-Token items)
        index_expr = [item for item in items[1:] if not isinstance(item, Token)][0]
        return ast.Subscript(object=obj, index=index_expr)

    def function_call(
        self, items: list[Token | ast.ASTNode | list[ast.ASTNode] | None]
    ) -> ast.FunctionCall:
        """Transform function call."""
        # Items: [atom, lparen, arguments_list/None, rparen]
        # The first item is the function expression (can be Variable or AttributeAccess)
        if not items or not isinstance(items[0], ast.ASTNode):
            raise ValueError("Function call must have a function expression")
        function: ast.ASTNode = items[0]

        # Arguments are in the list returned by the arguments rule (if present)
        # Filter for non-Token, non-None items and flatten lists
        arguments: list[ast.ASTNode] = []
        for item in items[1:]:
            if isinstance(item, list):
                # This is the arguments list - arg is already ast.ASTNode
                for arg in item:
                    arguments.append(arg)
            elif isinstance(item, ast.ASTNode):
                arguments.append(item)
        return ast.FunctionCall(function=function, arguments=arguments)

    def list_literal(
        self, items: list[Token | ast.ASTNode | list[ast.ASTNode] | None]
    ) -> ast.ListLiteral:
        """Transform list literal."""
        # Filter out bracket tokens and None, flatten lists
        elements: list[ast.ASTNode] = []
        for item in items:
            if isinstance(item, list):
                # This is the list_elements list - elem is already ast.ASTNode
                for elem in item:
                    elements.append(elem)
            elif isinstance(item, ast.ASTNode):
                elements.append(item)
        return ast.ListLiteral(elements=elements)

    def arguments(self, items: list[Token | ast.ASTNode]) -> list[ast.ASTNode]:
        """Transform function arguments, returning flat list."""
        # Filter out comma tokens
        return [item for item in items if isinstance(item, ast.ASTNode)]

    def list_elements(self, items: list[Token | ast.ASTNode]) -> list[ast.ASTNode]:
        """Transform list elements, returning flat list."""
        # Filter out comma tokens
        return [item for item in items if isinstance(item, ast.ASTNode)]

    def atom(self, items: list[Token | ast.ASTNode]) -> ast.ASTNode:
        """Transform atom (handles parenthesized expressions)."""
        # If we have parentheses, items = [lparen_token, expr, rparen_token]
        # Return just the expression (filter out tokens)
        nodes = [item for item in items if isinstance(item, ast.ASTNode)]
        if nodes:
            return nodes[0]
        # Fallback: if no AST nodes, try to get first item (shouldn't happen)
        if items and isinstance(items[0], ast.ASTNode):
            return items[0]
        raise ValueError("Atom must contain an AST node")


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
        result = transformer.transform(tree)

        if not isinstance(result, ast.ASTNode):
            raise ParseError(
                f"Parser returned unexpected type: {type(result)}", text=expression
            )
        return result

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
