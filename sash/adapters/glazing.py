"""Adapter for glazing package (VerbNet, PropBank, FrameNet).

This module provides an adapter to fetch lexical items from VerbNet, PropBank,
and FrameNet via the glazing package using the proper loader classes.
"""

from __future__ import annotations

from typing import Any, Literal

import glazing
import glazing.framenet.loader as fn_loader
import glazing.propbank.loader as pb_loader
import glazing.verbnet.loader as vn_loader

from sash.adapters.base import ResourceAdapter
from sash.adapters.cache import AdapterCache
from sash.data.language_codes import LanguageCode
from sash.resources.models import LexicalItem


class GlazingAdapter(ResourceAdapter):
    """Adapter for glazing package (VerbNet, PropBank, FrameNet).

    This adapter fetches verb frame information from VerbNet, PropBank, or
    FrameNet and converts it to LexicalItem format. Frame information is
    stored in the attributes field.

    Parameters
    ----------
    resource : Literal["verbnet", "propbank", "framenet"]
        Which glazing resource to use.
    cache : AdapterCache | None
        Optional cache instance. If None, no caching is performed.

    Examples
    --------
    >>> adapter = GlazingAdapter(resource="verbnet")
    >>> items = adapter.fetch_items(query="break", language_code="en")
    >>> all(item.language_code == "en" for item in items)
    True
    """

    def __init__(
        self,
        resource: Literal["verbnet", "propbank", "framenet"] = "verbnet",
        cache: AdapterCache | None = None,
    ) -> None:
        """Initialize glazing adapter.

        Parameters
        ----------
        resource : Literal["verbnet", "propbank", "framenet"]
            Which glazing resource to use. Default: "verbnet".
        cache : AdapterCache | None
            Optional cache instance.
        """
        self.resource = resource
        self.cache = cache
        self._loader: (
            vn_loader.VerbNetLoader
            | pb_loader.PropBankLoader
            | fn_loader.FrameNetLoader
            | None
        ) = None

    def _get_loader(
        self,
    ) -> vn_loader.VerbNetLoader | pb_loader.PropBankLoader | fn_loader.FrameNetLoader:
        """Get or create the appropriate loader for the resource.

        Returns
        -------
        VerbNetLoader | PropBankLoader | FrameNetLoader
            The loader instance for the configured resource.
        """
        if self._loader is None:
            if self.resource == "verbnet":
                self._loader = vn_loader.VerbNetLoader()
            elif self.resource == "propbank":
                self._loader = pb_loader.PropBankLoader()
            else:  # framenet
                self._loader = fn_loader.FrameNetLoader()
        return self._loader

    def fetch_items(
        self,
        query: str | None = None,
        language_code: LanguageCode = None,
        **kwargs: Any,
    ) -> list[LexicalItem]:
        """Fetch items from glazing resource.

        Parameters
        ----------
        query : str | None
            Lemma or predicate to query (e.g., "break", "run").
        language_code : LanguageCode
            Language code filter. Glazing resources are primarily English,
            so language_code="en" is typical. Other languages may not be
            supported.
        **kwargs : Any
            Additional parameters (resource-specific).

        Returns
        -------
        list[LexicalItem]
            Lexical items with frame information in attributes.

        Raises
        ------
        ValueError
            If query is None (glazing requires a query).
        RuntimeError
            If glazing resource access fails.

        Examples
        --------
        >>> adapter = GlazingAdapter(resource="verbnet")
        >>> items = adapter.fetch_items(query="break", language_code="en")
        >>> len(items) > 0
        True
        """
        if query is None:
            raise ValueError("GlazingAdapter requires a query string")

        # Check cache
        cache_key = None
        if self.cache:
            cache_key = self.cache.make_key(
                f"glazing_{self.resource}",
                query=query,
                language_code=language_code,
                **kwargs,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Fetch from glazing
        try:
            items = self._fetch_from_resource(query, language_code)

            # Cache result
            if self.cache and cache_key:
                self.cache.set(cache_key, items)

            return items

        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch from glazing {self.resource}: {e}"
            ) from e

    def _fetch_from_resource(
        self, query: str, language_code: LanguageCode
    ) -> list[LexicalItem]:
        """Fetch from specific glazing resource.

        Parameters
        ----------
        query : str
            Lemma or predicate to query.
        language_code : LanguageCode
            Language code filter.

        Returns
        -------
        list[LexicalItem]
            Lexical items from the resource.
        """
        if self.resource == "verbnet":
            return self._fetch_verbnet(query, language_code)
        elif self.resource == "propbank":
            return self._fetch_propbank(query, language_code)
        else:  # framenet
            return self._fetch_framenet(query, language_code)

    def _fetch_verbnet(
        self, query: str, language_code: LanguageCode
    ) -> list[LexicalItem]:
        """Fetch from VerbNet using VerbNetLoader.

        Parameters
        ----------
        query : str
            Verb lemma to search for.
        language_code : LanguageCode
            Language code filter.

        Returns
        -------
        list[LexicalItem]
            LexicalItem objects for matching verb classes.
        """
        loader = self._get_loader()
        assert isinstance(loader, vn_loader.VerbNetLoader)

        items: list[LexicalItem] = []

        # Search through all verb classes for members matching the query
        for verb_class in loader.classes.values():
            if verb_class.members:
                for member in verb_class.members:
                    if member.name == query:
                        # Create LexicalItem for this verb class
                        item = LexicalItem(
                            lemma=query,
                            pos="VERB",
                            language_code=language_code or "en",
                            attributes={
                                "verbnet_class": verb_class.id,
                                "themroles": [r.type for r in verb_class.themroles]
                                if verb_class.themroles
                                else [],
                                "frame_count": len(verb_class.frames)
                                if verb_class.frames
                                else 0,
                            },
                        )
                        items.append(item)

        return items

    def _fetch_propbank(
        self, query: str, language_code: LanguageCode
    ) -> list[LexicalItem]:
        """Fetch from PropBank using PropBankLoader.

        Parameters
        ----------
        query : str
            Predicate lemma to search for.
        language_code : LanguageCode
            Language code filter.

        Returns
        -------
        list[LexicalItem]
            LexicalItem objects for matching predicates.
        """
        loader = self._get_loader()
        assert isinstance(loader, pb_loader.PropBankLoader)

        items: list[LexicalItem] = []

        # Get frameset for the predicate
        frameset = loader.get_frameset(query)
        if frameset and frameset.rolesets:
            for roleset in frameset.rolesets:
                # Create LexicalItem for each roleset
                item = LexicalItem(
                    lemma=query,
                    pos="VERB",
                    language_code=language_code or "en",
                    attributes={
                        "propbank_roleset_id": roleset.id,
                        "roleset_name": roleset.name if roleset.name else "",
                        "roles": [
                            {
                                "arg": role.n,
                                "function": role.f,
                                "description": role.descr,
                            }
                            for role in roleset.roles
                        ]
                        if roleset.roles
                        else [],
                    },
                )
                items.append(item)

        return items

    def _fetch_framenet(
        self, query: str, language_code: LanguageCode
    ) -> list[LexicalItem]:
        """Fetch from FrameNet using FrameNetLoader.

        Parameters
        ----------
        query : str
            Lexical unit name to search for.
        language_code : LanguageCode
            Language code filter.

        Returns
        -------
        list[LexicalItem]
            LexicalItem objects for matching frames.
        """
        loader = self._get_loader()
        assert isinstance(loader, fn_loader.FrameNetLoader)

        items: list[LexicalItem] = []

        # Search through all frames for lexical units matching the query
        for frame in loader.frames:
            if frame.lexical_units:
                for lu in frame.lexical_units:
                    if query.lower() in lu.name.lower():
                        # Create LexicalItem for each matching lexical unit
                        item = LexicalItem(
                            lemma=query,
                            pos=str(lu.pos) if lu.pos else "UNKNOWN",
                            language_code=language_code or "en",
                            attributes={
                                "framenet_frame": frame.name,
                                "frame_id": frame.id,
                                "lexical_unit_id": lu.id,
                                "lexical_unit_name": lu.name,
                                "frame_elements": [
                                    fe.name for fe in frame.frame_elements
                                ]
                                if frame.frame_elements
                                else [],
                            },
                        )
                        items.append(item)

        return items

    def is_available(self) -> bool:
        """Check if glazing package is available.

        Returns
        -------
        bool
            True if glazing can be imported and data is initialized, False
            otherwise.

        Examples
        --------
        >>> adapter = GlazingAdapter()
        >>> adapter.is_available()
        True
        """
        try:
            # Check if glazing is initialized
            glazing.check_initialization()
            # Try to create a loader to verify data is accessible
            self._get_loader()
            return True
        except Exception:
            return False
