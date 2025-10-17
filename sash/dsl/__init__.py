"""Constraint Domain-Specific Language (DSL).

This module provides a DSL for expressing constraints on lexical items
in templates. The DSL includes:

- Boolean operators: and, or, not
- Comparison operators: ==, !=, <, >, <=, >=
- Membership operators: in, not in
- Arithmetic operators: +, -, *, /, %
- Function calls
- Attribute access
- List literals

Examples
--------
>>> from sash.dsl import parse
>>> node = parse("lemma == 'walk' and pos == 'VERB'")
>>> node.operator
'and'
"""

from sash.dsl.ast import (
    ASTNode,
    AttributeAccess,
    BinaryOp,
    FunctionCall,
    ListLiteral,
    Literal,
    UnaryOp,
    Variable,
)
from sash.dsl.errors import DSLError, EvaluationError, ParseError
from sash.dsl.parser import parse

__all__ = [
    # AST nodes
    "ASTNode",
    "Literal",
    "Variable",
    "BinaryOp",
    "UnaryOp",
    "FunctionCall",
    "ListLiteral",
    "AttributeAccess",
    # Errors
    "DSLError",
    "ParseError",
    "EvaluationError",
    # Parser
    "parse",
]
