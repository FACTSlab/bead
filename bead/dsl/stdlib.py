"""Standard library functions for constraint DSL.

This module provides built-in functions that can be used in constraint
expressions. Functions are organized by category and registered with
the evaluation context.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bead.dsl.context import EvaluationContext


# String functions
def len_(s: Any) -> int:
    """Return length of string or collection.

    Parameters
    ----------
    s : Any
        String or collection to measure.

    Returns
    -------
    int
        Length of string or collection.

    Examples
    --------
    >>> len_("hello")
    5
    >>> len_([1, 2, 3])
    3
    """
    return len(s)


def lower(s: str) -> str:
    """Convert string to lowercase.

    Parameters
    ----------
    s : str
        String to convert.

    Returns
    -------
    str
        Lowercase string.

    Examples
    --------
    >>> lower("HELLO")
    'hello'
    """
    return s.lower()


def upper(s: str) -> str:
    """Convert string to uppercase.

    Parameters
    ----------
    s : str
        String to convert.

    Returns
    -------
    str
        Uppercase string.

    Examples
    --------
    >>> upper("hello")
    'HELLO'
    """
    return s.upper()


def startswith(s: str, prefix: str) -> bool:
    """Check if string starts with prefix.

    Parameters
    ----------
    s : str
        String to check.
    prefix : str
        Prefix to look for.

    Returns
    -------
    bool
        True if string starts with prefix.

    Examples
    --------
    >>> startswith("hello", "hel")
    True
    >>> startswith("hello", "bye")
    False
    """
    return s.startswith(prefix)


def endswith(s: str, suffix: str) -> bool:
    """Check if string ends with suffix.

    Parameters
    ----------
    s : str
        String to check.
    suffix : str
        Suffix to look for.

    Returns
    -------
    bool
        True if string ends with suffix.

    Examples
    --------
    >>> endswith("hello", "lo")
    True
    >>> endswith("hello", "hi")
    False
    """
    return s.endswith(suffix)


def contains(s: str, substring: str) -> bool:
    """Check if string contains substring.

    Parameters
    ----------
    s : str
        String to check.
    substring : str
        Substring to look for.

    Returns
    -------
    bool
        True if string contains substring.

    Examples
    --------
    >>> contains("hello", "ell")
    True
    >>> contains("hello", "bye")
    False
    """
    return substring in s


def replace(s: str, old: str, new: str) -> str:
    """Replace occurrences of substring.

    Parameters
    ----------
    s : str
        String to modify.
    old : str
        Substring to replace.
    new : str
        Replacement substring.

    Returns
    -------
    str
        String with replacements.

    Examples
    --------
    >>> replace("hello world", "world", "there")
    'hello there'
    """
    return s.replace(old, new)


def split(s: str, sep: str = " ") -> list[str]:
    """Split string by separator.

    Parameters
    ----------
    s : str
        String to split.
    sep : str
        Separator string. Defaults to space.

    Returns
    -------
    list[str]
        List of substrings.

    Examples
    --------
    >>> split("a,b,c", ",")
    ['a', 'b', 'c']
    """
    return s.split(sep)


# Collection functions
def count(collection: Any, item: Any) -> int:
    """Count occurrences of item in collection.

    Parameters
    ----------
    collection : Any
        Collection to search.
    item : Any
        Item to count.

    Returns
    -------
    int
        Number of occurrences.

    Examples
    --------
    >>> count([1, 2, 2, 3], 2)
    2
    >>> count("hello", "l")
    2
    """
    return collection.count(item)


def sum_(collection: list[int | float]) -> int | float:
    """Sum numeric collection.

    Parameters
    ----------
    collection : list[int | float]
        Collection of numbers.

    Returns
    -------
    int | float
        Sum of all numbers.

    Examples
    --------
    >>> sum_([1, 2, 3])
    6
    >>> sum_([1.5, 2.5])
    4.0
    """
    return sum(collection)


def min_(collection: list[Any]) -> Any:
    """Return minimum value from collection.

    Parameters
    ----------
    collection : list[Any]
        Collection to search.

    Returns
    -------
    Any
        Minimum value.

    Examples
    --------
    >>> min_([3, 1, 2])
    1
    """
    return min(collection)


def max_(collection: list[Any]) -> Any:
    """Return maximum value from collection.

    Parameters
    ----------
    collection : list[Any]
        Collection to search.

    Returns
    -------
    Any
        Maximum value.

    Examples
    --------
    >>> max_([3, 1, 2])
    3
    """
    return max(collection)


def any_(collection: list[Any]) -> bool:
    """Check if any element is truthy.

    Parameters
    ----------
    collection : list[Any]
        Collection to check.

    Returns
    -------
    bool
        True if any element is truthy.

    Examples
    --------
    >>> any_([False, True, False])
    True
    >>> any_([False, False])
    False
    """
    return any(collection)


def all_(collection: list[Any]) -> bool:
    """Check if all elements are truthy.

    Parameters
    ----------
    collection : list[Any]
        Collection to check.

    Returns
    -------
    bool
        True if all elements are truthy.

    Examples
    --------
    >>> all_([True, True, True])
    True
    >>> all_([True, False, True])
    False
    """
    return all(collection)


# Type checking functions
def is_str(value: Any) -> bool:
    """Check if value is a string.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a string.

    Examples
    --------
    >>> is_str("hello")
    True
    >>> is_str(42)
    False
    """
    return isinstance(value, str)


def is_int(value: Any) -> bool:
    """Check if value is an integer.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is an integer.

    Examples
    --------
    >>> is_int(42)
    True
    >>> is_int(42.0)
    False
    """
    return isinstance(value, int) and not isinstance(value, bool)


def is_float(value: Any) -> bool:
    """Check if value is a float.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a float.

    Examples
    --------
    >>> is_float(42.0)
    True
    >>> is_float(42)
    False
    """
    return isinstance(value, float)


def is_bool(value: Any) -> bool:
    """Check if value is a boolean.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a boolean.

    Examples
    --------
    >>> is_bool(True)
    True
    >>> is_bool(1)
    False
    """
    return isinstance(value, bool)


def is_list(value: Any) -> bool:
    """Check if value is a list.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a list.

    Examples
    --------
    >>> is_list([1, 2, 3])
    True
    >>> is_list((1, 2, 3))
    False
    """
    return isinstance(value, list)


# Conversion functions
def str_(value: Any) -> str:
    """Convert value to string.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    str
        String representation of value.

    Examples
    --------
    >>> str_(42)
    '42'
    >>> str_(True)
    'True'
    """
    return str(value)


# Math functions
def abs_(value: int | float) -> int | float:
    """Return absolute value.

    Parameters
    ----------
    value : int | float
        Numeric value.

    Returns
    -------
    int | float
        Absolute value.

    Examples
    --------
    >>> abs_(-5)
    5
    >>> abs_(5)
    5
    """
    return abs(value)


def round_(value: float, ndigits: int = 0) -> float:
    """Round numeric value.

    Parameters
    ----------
    value : float
        Value to round.
    ndigits : int
        Number of decimal places.

    Returns
    -------
    float
        Rounded value.

    Examples
    --------
    >>> round_(3.14159, 2)
    3.14
    """
    return round(value, ndigits)


def floor(value: float) -> int:
    """Return floor of value.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    int
        Floor value.

    Examples
    --------
    >>> floor(3.7)
    3
    >>> floor(-3.7)
    -4
    """
    return math.floor(value)


def ceil(value: float) -> int:
    """Return ceiling of value.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    int
        Ceiling value.

    Examples
    --------
    >>> ceil(3.2)
    4
    >>> ceil(-3.2)
    -3
    """
    return math.ceil(value)


# Logic functions
def not_(value: Any) -> bool:
    """Return logical negation of value.

    Parameters
    ----------
    value : Any
        Value to negate.

    Returns
    -------
    bool
        Logical negation.

    Examples
    --------
    >>> not_(True)
    False
    >>> not_(False)
    True
    >>> not_(0)
    True
    """
    return not value


# ============================================================================
# Simulation Functions
# ============================================================================


def sigmoid(x: float) -> float:
    """Sigmoid activation function.

    Converts unbounded value to probability in (0, 1).

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Sigmoid output in (0, 1).

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> round(sigmoid(5.0), 3)
    0.993
    >>> round(sigmoid(-5.0), 3)
    0.007
    """
    return 1.0 / (1.0 + math.exp(-x))


def softmax(values: list[float]) -> list[float]:
    """Softmax function over list of values.

    Converts list of scores to probability distribution.

    Parameters
    ----------
    values : list[float]
        Input scores.

    Returns
    -------
    list[float]
        Probability distribution (sums to 1.0).

    Examples
    --------
    >>> probs = softmax([1.0, 2.0, 3.0])
    >>> [round(p, 2) for p in probs]
    [0.09, 0.24, 0.67]
    """
    if not values:
        return []
    exp_values = [math.exp(v) for v in values]
    total = sum(exp_values)
    return [e / total for e in exp_values]


def sample_categorical(probs: list[float], seed: int | None = None) -> int:
    """Sample from categorical distribution.

    Parameters
    ----------
    probs : list[float]
        Probability distribution.
    seed : int | None
        Random seed.

    Returns
    -------
    int
        Sampled index (0-based).

    Examples
    --------
    >>> sample_categorical([0.2, 0.5, 0.3], seed=42)
    1
    """
    if seed is not None:
        random.seed(seed)
    return random.choices(range(len(probs)), weights=probs)[0]


def add_noise(
    value: float, noise_type: str, strength: float, seed: int | None = None
) -> float:
    """Add noise to a value.

    Parameters
    ----------
    value : float
        Original value.
    noise_type : str
        Type of noise ("gaussian", "uniform").
    strength : float
        Noise strength (stddev for gaussian, range for uniform).
    seed : int | None
        Random seed.

    Returns
    -------
    float
        Value with noise added.

    Examples
    --------
    >>> result = add_noise(5.0, "gaussian", 0.1, seed=42)
    >>> isinstance(result, float)
    True
    >>> result = add_noise(5.0, "uniform", 0.1, seed=42)
    >>> isinstance(result, float)
    True
    """
    if seed is not None:
        random.seed(seed)

    if noise_type == "gaussian":
        return value + random.gauss(0, strength)
    elif noise_type == "uniform":
        return value + random.uniform(-strength, strength)
    else:
        return value


def model_output(item: Any, key: str, default: Any = None) -> Any:
    """Extract model output from item.

    Parameters
    ----------
    item : Item
        Item with model outputs.
    key : str
        Key to extract (e.g., "lm_score", "embedding").
    default : Any
        Default value if key not found.

    Returns
    -------
    Any
        Extracted value or default.

    Examples
    --------
    >>> # Would work with actual Item object
    >>> # model_output(item, "lm_score", default=0.0)
    >>> # -12.4
    """
    if not hasattr(item, "model_outputs"):
        return default

    for output in item.model_outputs:
        if output.operation == key or key in output.computation_metadata:
            return output.output

    # Try item_metadata as fallback
    if hasattr(item, "item_metadata") and key in item.item_metadata:
        return item.item_metadata[key]

    return default


def distance(emb1: list[float], emb2: list[float], metric: str = "cosine") -> float:
    """Compute distance between embeddings.

    Parameters
    ----------
    emb1 : list[float]
        First embedding.
    emb2 : list[float]
        Second embedding.
    metric : str
        Distance metric ("cosine", "euclidean", "manhattan").

    Returns
    -------
    float
        Distance value.

    Examples
    --------
    >>> distance([1.0, 0.0], [0.0, 1.0], "cosine")
    1.0
    >>> round(distance([1.0, 0.0], [0.0, 1.0], "euclidean"), 3)
    1.414
    >>> distance([1.0, 0.0], [0.0, 1.0], "manhattan")
    2.0
    """
    if metric == "cosine":
        dot = sum(a * b for a, b in zip(emb1, emb2, strict=True))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1.0 - (dot / (norm1 * norm2))

    elif metric == "euclidean":
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(emb1, emb2, strict=True)))

    elif metric == "manhattan":
        return sum(abs(a - b) for a, b in zip(emb1, emb2, strict=True))

    else:
        msg = f"Unknown metric: {metric}"
        raise ValueError(msg)


def preference_prob(score1: float, score2: float, temperature: float = 1.0) -> float:
    """Compute preference probability using sigmoid.

    P(choose option 1) = sigmoid((score1 - score2) / temperature)

    Parameters
    ----------
    score1 : float
        Score for option 1.
    score2 : float
        Score for option 2.
    temperature : float
        Temperature for scaling.

    Returns
    -------
    float
        Probability of choosing option 1.

    Examples
    --------
    >>> round(preference_prob(10.0, 5.0, temperature=1.0), 3)
    0.993
    >>> round(preference_prob(10.0, 5.0, temperature=5.0), 2)
    0.73
    """
    return sigmoid((score1 - score2) / temperature)


# Register simulation functions
SIMULATION_FUNCTIONS: dict[str, Any] = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "sample_categorical": sample_categorical,
    "add_noise": add_noise,
    "model_output": model_output,
    "distance": distance,
    "preference_prob": preference_prob,
}


# Registry
STDLIB_FUNCTIONS: dict[str, Any] = {
    # String functions
    "len": len_,
    "lower": lower,
    "upper": upper,
    "startswith": startswith,
    "endswith": endswith,
    "contains": contains,
    "replace": replace,
    "split": split,
    # Collection functions
    "count": count,
    "sum": sum_,
    "min": min_,
    "max": max_,
    "any": any_,
    "all": all_,
    # Type checking
    "is_str": is_str,
    "is_int": is_int,
    "is_float": is_float,
    "is_bool": is_bool,
    "is_list": is_list,
    # Conversion functions
    "str": str_,
    # Math functions
    "abs": abs_,
    "round": round_,
    "floor": floor,
    "ceil": ceil,
    # Logic functions
    "not": not_,
}

# Update STDLIB_FUNCTIONS with simulation functions
STDLIB_FUNCTIONS.update(SIMULATION_FUNCTIONS)


def register_stdlib(context: EvaluationContext) -> None:
    """Register all standard library functions in context.

    Parameters
    ----------
    context : EvaluationContext
        Context to register functions in.

    Examples
    --------
    >>> from bead.dsl.context import EvaluationContext
    >>> ctx = EvaluationContext()
    >>> register_stdlib(ctx)
    >>> ctx.call_function("len", ["hello"])
    5
    """
    for name, func in STDLIB_FUNCTIONS.items():
        context.set_function(name, func)
