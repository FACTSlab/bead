"""Abstract base class for external resource adapters.

This module defines the interface that all resource adapters must implement
to fetch lexical items from external linguistic databases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sash.data.language_codes import LanguageCode
from sash.resources.models import LexicalItem


class ResourceAdapter(ABC):
    """Abstract base class for external resource adapters.

    Resource adapters fetch lexical items from external linguistic databases
    and convert them to the sash LexicalItem format. All adapters must
    implement language_code filtering to support multi-language workflows.

    Subclasses must implement:
    - fetch_items(): Retrieve items from the external resource
    - is_available(): Check if the external resource is accessible

    Examples
    --------
    >>> class MyAdapter(ResourceAdapter):
    ...     def fetch_items(self, query=None, language_code=None, **kwargs):
    ...         # Fetch from external resource
    ...         return [LexicalItem(lemma="walk", pos="VERB", language_code="en")]
    ...     def is_available(self):
    ...         return True
    >>> adapter = MyAdapter()
    >>> items = adapter.fetch_items(query="walk", language_code="en")
    >>> len(items) > 0
    True
    """

    @abstractmethod
    def fetch_items(
        self,
        query: str | None = None,
        language_code: LanguageCode = None,
        **kwargs: Any,
    ) -> list[LexicalItem]:
        """Fetch lexical items from external resource.

        Parameters
        ----------
        query : str | None
            Query string in adapter-specific format (e.g., lemma, predicate name,
            class identifier). If None, behavior is adapter-specific (may return
            all items, raise error, or use default query).
        language_code : LanguageCode
            ISO 639-1 (2-letter) or ISO 639-3 (3-letter) language code to filter
            results. Examples: "en", "eng", "ko", "kor". If None, returns items
            for all available languages.
        **kwargs : Any
            Additional adapter-specific parameters (e.g., pos="VERB",
            resource="verbnet", include_features=True).

        Returns
        -------
        list[LexicalItem]
            Lexical items fetched from the external resource. Each item should
            have language_code set if known.

        Raises
        ------
        ValueError
            If query is invalid or required parameters are missing.
        RuntimeError
            If the external resource is unavailable or the request fails.

        Examples
        --------
        >>> adapter = MyAdapter()
        >>> items = adapter.fetch_items(query="break", language_code="en")
        >>> all(item.language_code == "en" for item in items)
        True
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the external resource is available.

        This method should verify that the external resource can be accessed,
        whether via installed packages, accessible data files, or network APIs.

        Returns
        -------
        bool
            True if the resource can be accessed, False otherwise.

        Examples
        --------
        >>> adapter = MyAdapter()
        >>> adapter.is_available()
        True
        """
        ...
