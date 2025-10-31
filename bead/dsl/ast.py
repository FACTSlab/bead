"""Abstract Syntax Tree node definitions for constraint DSL.

This module defines the AST nodes that represent parsed constraint expressions.
Each node type corresponds to a construct in the constraint DSL grammar.
"""

from __future__ import annotations

from bead.data.base import BeadBaseModel


class ASTNode(BeadBaseModel):
    """Base class for all AST nodes.

    All AST nodes inherit from BeadBaseModel to get:
    - Automatic validation
    - Serialization support
    - Metadata tracking
    """

    pass


class Literal(ASTNode):
    """Literal value node (string, number, boolean).

    Examples
    --------
    >>> node = Literal(value="hello")
    >>> node.value
    'hello'
    >>> node = Literal(value=42)
    >>> node.value
    42
    """

    value: str | int | float | bool


class Variable(ASTNode):
    """Variable reference node.

    References a variable in the evaluation context (e.g., item attributes).

    Examples
    --------
    >>> node = Variable(name="lemma")
    >>> node.name
    'lemma'
    """

    name: str


class BinaryOp(ASTNode):
    """Binary operation node.

    Represents operations like: a == b, x > y, p and q

    Attributes
    ----------
    operator : str
        The operator (==, !=, <, >, <=, >=, and, or, in, etc.)
    left : ASTNode
        Left operand
    right : ASTNode
        Right operand

    Examples
    --------
    >>> left = Variable(name="pos")
    >>> right = Literal(value="VERB")
    >>> node = BinaryOp(operator="==", left=left, right=right)
    >>> node.operator
    '=='
    """

    operator: str
    left: ASTNode
    right: ASTNode


class UnaryOp(ASTNode):
    """Unary operation node.

    Represents operations like: not x, -y

    Examples
    --------
    >>> operand = Variable(name="is_transitive")
    >>> node = UnaryOp(operator="not", operand=operand)
    >>> node.operator
    'not'
    """

    operator: str
    operand: ASTNode


class FunctionCall(ASTNode):
    """Function call node.

    Represents function calls and method calls like:
    - len(x), startswith("pre")
    - obj.method(arg)

    Examples
    --------
    >>> func = Variable(name="len")
    >>> arg = Variable(name="lemma")
    >>> node = FunctionCall(function=func, arguments=[arg])
    >>> node.function.name
    'len'
    """

    function: ASTNode  # Variable for functions, AttributeAccess for methods
    arguments: list[ASTNode]


class ListLiteral(ASTNode):
    """List literal node.

    Represents list literals like: ["a", "b", "c"]

    Examples
    --------
    >>> elements = [Literal(value="a"), Literal(value="b")]
    >>> node = ListLiteral(elements=elements)
    >>> len(node.elements)
    2
    """

    elements: list[ASTNode]


class AttributeAccess(ASTNode):
    """Attribute access node.

    Represents attribute access like: item.lemma, obj.property

    Examples
    --------
    >>> obj = Variable(name="item")
    >>> node = AttributeAccess(object=obj, attribute="lemma")
    >>> node.attribute
    'lemma'
    """

    object: ASTNode
    attribute: str


class Subscript(ASTNode):
    """Subscript access node.

    Represents subscript access like: item['key'], obj[0]

    Examples
    --------
    >>> obj = Variable(name="item")
    >>> key = Literal(value="key")
    >>> node = Subscript(object=obj, index=key)
    >>> node.index.value
    'key'
    """

    object: ASTNode
    index: ASTNode
