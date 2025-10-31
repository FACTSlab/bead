"""Utilities for API-based model adapters.

This module provides shared utilities for API-based model adapters,
including retry logic with exponential backoff and rate limiting.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate function with retry logic and exponential backoff.

    Retries a function call on specified exceptions with exponential backoff
    between attempts. The delay between retries grows exponentially:
    delay = initial_delay * (backoff_factor ** attempt).

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (default: 3).
    initial_delay : float
        Initial delay in seconds before first retry (default: 1.0).
    backoff_factor : float
        Multiplicative factor for delay between retries (default: 2.0).
    exceptions : tuple[type[Exception], ...]
        Tuple of exception types to catch and retry on (default: (Exception,)).

    Returns
    -------
    Callable
        Decorated function with retry logic.

    Examples
    --------
    >>> @retry_with_backoff(max_retries=3, initial_delay=1.0)
    ... def call_api():
    ...     # May raise transient errors
    ...     return api.get_data()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = initial_delay * (backoff_factor**attempt)
                        time.sleep(delay)
                    else:
                        # Last attempt failed, re-raise
                        raise

            # Should never reach here, but for type checker
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected state in retry_with_backoff")

        return wrapper

    return decorator


class RateLimiter:
    """Rate limiter for API calls.

    Tracks call timestamps and enforces a maximum rate of calls per minute.
    Uses a sliding window algorithm to ensure the rate limit is respected.

    Parameters
    ----------
    calls_per_minute : int
        Maximum number of calls allowed per minute (default: 60).

    Attributes
    ----------
    calls_per_minute : int
        Maximum number of calls allowed per minute.
    call_times : list[float]
        Timestamps of recent API calls.
    """

    def __init__(self, calls_per_minute: int = 60) -> None:
        self.calls_per_minute = calls_per_minute
        self.call_times: list[float] = []

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded.

        Checks if making a call now would exceed the rate limit.
        If so, sleeps until enough time has passed.
        """
        now = time.time()

        # Remove calls older than 1 minute
        cutoff_time = now - 60.0
        self.call_times = [t for t in self.call_times if t > cutoff_time]

        # If at rate limit, wait until oldest call expires
        if len(self.call_times) >= self.calls_per_minute:
            oldest_call = self.call_times[0]
            wait_time = 60.0 - (now - oldest_call)
            if wait_time > 0:
                time.sleep(wait_time)
            # Clean up again after waiting
            now = time.time()
            cutoff_time = now - 60.0
            self.call_times = [t for t in self.call_times if t > cutoff_time]

        # Record this call
        self.call_times.append(time.time())


def rate_limit(
    calls_per_minute: int = 60,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate function with rate limiting for API calls.

    Enforces a maximum rate of API calls per minute using a shared
    RateLimiter instance. Calls that would exceed the rate limit
    will block until the limit resets.

    Parameters
    ----------
    calls_per_minute : int
        Maximum number of calls allowed per minute (default: 60).

    Returns
    -------
    Callable
        Decorated function with rate limiting.

    Examples
    --------
    >>> @rate_limit(calls_per_minute=30)
    ... def call_api():
    ...     return api.get_data()
    """
    limiter = RateLimiter(calls_per_minute=calls_per_minute)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            limiter.wait_if_needed()
            return func(*args, **kwargs)

        return wrapper

    return decorator
