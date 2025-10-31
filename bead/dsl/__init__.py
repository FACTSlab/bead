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
- Standard library functions

Examples
--------
>>> from bead.dsl import parse, evaluate
>>> node = parse("lemma == 'walk' and pos == 'VERB'")
>>> from bead.dsl import EvaluationContext
>>> ctx = EvaluationContext()
>>> ctx.set_variable("lemma", "walk")
>>> ctx.set_variable("pos", "VERB")
>>> evaluate(node, ctx)
True
"""

from typing import Any

from bead.dsl.ast import (
    ASTNode,
    AttributeAccess,
    BinaryOp,
    FunctionCall,
    ListLiteral,
    Literal,
    UnaryOp,
    Variable,
)
from bead.dsl.context import EvaluationContext
from bead.dsl.errors import DSLError, EvaluationError, ParseError
from bead.dsl.evaluator import Evaluator
from bead.dsl.parser import parse
from bead.dsl.stdlib import SIMULATION_FUNCTIONS, STDLIB_FUNCTIONS, register_stdlib

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
    # Evaluation
    "Evaluator",
    "EvaluationContext",
    "evaluate",
    # Standard library
    "STDLIB_FUNCTIONS",
    "SIMULATION_FUNCTIONS",
    "register_stdlib",
]


# Convenience function
def evaluate(node: ASTNode, context: EvaluationContext, use_cache: bool = True) -> Any:
    """Evaluate a constraint expression.

    Parameters
    ----------
    node : ASTNode
        Parsed AST node to evaluate.
    context : EvaluationContext
        Evaluation context with variables and functions.
    use_cache : bool
        Whether to use caching.

    Returns
    -------
    Any
        Result of evaluation.

    Examples
    --------
    >>> from bead.dsl import parse, EvaluationContext
    >>> ctx = EvaluationContext()
    >>> ctx.set_variable("x", 10)
    >>> node = parse("x > 5")
    >>> evaluate(node, ctx)
    True
    """
    evaluator = Evaluator(use_cache=use_cache)
    return evaluator.evaluate(node, context)
