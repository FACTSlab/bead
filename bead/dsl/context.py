"""Evaluation context for constraint DSL.

This module provides the EvaluationContext class that manages variable
bindings and function lookups during constraint evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bead.dsl.errors import EvaluationError


class EvaluationContext:
    """Evaluation context for constraint expressions.

    The context provides:
    - Variable bindings (e.g., item attributes)
    - Function registry (built-in and custom functions)
    - Parent context chain for scoping

    Parameters
    ----------
    parent : EvaluationContext | None
        Parent context for variable/function lookup chain.

    Examples
    --------
    >>> ctx = EvaluationContext()
    >>> ctx.set_variable("x", 42)
    >>> ctx.get_variable("x")
    42
    >>> ctx.set_function("double", lambda x: x * 2)
    >>> ctx.call_function("double", [5])
    10
    """

    def __init__(self, parent: EvaluationContext | None = None) -> None:
        self._variables: dict[str, Any] = {}
        self._functions: dict[str, Callable[..., Any]] = {}
        self._parent = parent

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the context.

        Parameters
        ----------
        name : str
            Variable name.
        value : Any
            Variable value.
        """
        self._variables[name] = value

    def get_variable(self, name: str) -> Any:
        """Get a variable from the context (searches parent chain).

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        Any
            Variable value.

        Raises
        ------
        EvaluationError
            If variable is not defined in context or parent chain.
        """
        if name in self._variables:
            return self._variables[name]
        if self._parent is not None:
            return self._parent.get_variable(name)
        raise EvaluationError(f"Undefined variable: {name}")

    def has_variable(self, name: str) -> bool:
        """Check if variable exists in context or parent chain.

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        bool
            True if variable exists, False otherwise.
        """
        if name in self._variables:
            return True
        if self._parent is not None:
            return self._parent.has_variable(name)
        return False

    def set_function(self, name: str, func: Callable[..., Any]) -> None:
        """Register a function in the context.

        Parameters
        ----------
        name : str
            Function name.
        func : Callable[..., Any]
            Function to register.
        """
        self._functions[name] = func

    def call_function(self, name: str, args: list[Any]) -> Any:
        """Call a function with arguments.

        Parameters
        ----------
        name : str
            Function name.
        args : list[Any]
            Function arguments.

        Returns
        -------
        Any
            Function return value.

        Raises
        ------
        EvaluationError
            If function is not defined or call fails.
        """
        if name in self._functions:
            try:
                return self._functions[name](*args)
            except TypeError as e:
                raise EvaluationError(f"Function call failed for '{name}': {e}") from e
        if self._parent is not None:
            return self._parent.call_function(name, args)
        raise EvaluationError(f"Undefined function: {name}")

    def has_function(self, name: str) -> bool:
        """Check if function exists in context or parent chain.

        Parameters
        ----------
        name : str
            Function name.

        Returns
        -------
        bool
            True if function exists, False otherwise.
        """
        if name in self._functions:
            return True
        if self._parent is not None:
            return self._parent.has_function(name)
        return False

    def create_child(self) -> EvaluationContext:
        """Create a child context with this context as parent.

        Returns
        -------
        EvaluationContext
            New child context.

        Examples
        --------
        >>> parent = EvaluationContext()
        >>> parent.set_variable("x", 10)
        >>> child = parent.create_child()
        >>> child.get_variable("x")
        10
        >>> child.set_variable("y", 20)
        >>> child.get_variable("y")
        20
        """
        return EvaluationContext(parent=self)
