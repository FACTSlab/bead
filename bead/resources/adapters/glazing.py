"""Adapter for glazing package (VerbNet, PropBank, FrameNet).

This module provides an adapter to fetch lexical items from VerbNet, PropBank,
and FrameNet via the glazing package using the proper loader classes.
"""

from __future__ import annotations

from typing import Any, Literal

import glazing
from glazing.framenet.loader import FrameNetLoader
from glazing.propbank.loader import PropBankLoader
from glazing.verbnet.loader import VerbNetLoader

from bead.data.language_codes import LanguageCode
from bead.resources.adapters.base import ResourceAdapter
from bead.resources.adapters.cache import AdapterCache
from bead.resources.lexical_item import LexicalItem


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
        self._loader: VerbNetLoader | PropBankLoader | FrameNetLoader | None = None

    def _get_loader(self) -> VerbNetLoader | PropBankLoader | FrameNetLoader:
        """Get or create the appropriate loader for the resource.

        Returns
        -------
        VerbNetLoader | PropBankLoader | FrameNetLoader
            The loader instance for the configured resource.
        """
        if self._loader is None:
            if self.resource == "verbnet":
                self._loader = VerbNetLoader()
            elif self.resource == "propbank":
                self._loader = PropBankLoader()
            else:  # framenet
                self._loader = FrameNetLoader()
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
            If None, fetches ALL items from the resource.
        language_code : LanguageCode
            Language code filter. Glazing resources are primarily English,
            so language_code="en" is typical. Other languages may not be
            supported.
        **kwargs : Any
            Additional parameters:
            - include_frames (bool): Include detailed frame information
              (syntax, examples, descriptions). Default: False.

        Returns
        -------
        list[LexicalItem]
            Lexical items with frame information in attributes.

        Raises
        ------
        RuntimeError
            If glazing resource access fails.

        Examples
        --------
        >>> # Query specific verb
        >>> adapter = GlazingAdapter(resource="verbnet")
        >>> items = adapter.fetch_items(query="break", language_code="en")
        >>> len(items) > 0
        True
        >>> # Fetch all items from resource
        >>> all_items = adapter.fetch_items(query=None, language_code="en")
        >>> len(all_items) > 100
        True
        >>> # Include detailed frame information
        >>> items = adapter.fetch_items(  # doctest: +SKIP
        ...     query="break", language_code="en", include_frames=True
        ... )
        >>> "frames" in items[0].attributes  # doctest: +SKIP
        True
        """
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
            items = self._fetch_from_resource(query, language_code, **kwargs)

            # Cache result
            if self.cache and cache_key:
                self.cache.set(cache_key, items)

            return items

        except NotImplementedError:
            # Re-raise NotImplementedError without wrapping
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch from glazing {self.resource}: {e}"
            ) from e

    def _fetch_from_resource(
        self, query: str | None, language_code: LanguageCode, **kwargs: Any
    ) -> list[LexicalItem]:
        """Fetch from specific glazing resource.

        Parameters
        ----------
        query : str | None
            Lemma or predicate to query. If None, fetch all items.
        language_code : LanguageCode
            Language code filter.
        **kwargs : Any
            Additional parameters (e.g., include_frames).

        Returns
        -------
        list[LexicalItem]
            Lexical items from the resource.
        """
        if self.resource == "verbnet":
            return self._fetch_verbnet(query, language_code, **kwargs)
        elif self.resource == "propbank":
            return self._fetch_propbank(query, language_code, **kwargs)
        else:  # framenet
            return self._fetch_framenet(query, language_code, **kwargs)

    def _fetch_verbnet(
        self, query: str | None, language_code: LanguageCode, **kwargs: Any
    ) -> list[LexicalItem]:
        """Fetch from VerbNet using VerbNetLoader.

        Parameters
        ----------
        query : str | None
            Verb lemma to search for. If None, fetch ALL verbs.
        language_code : LanguageCode
            Language code filter.
        **kwargs : Any
            Additional parameters:
            - include_frames (bool): Include detailed frame information.

        Returns
        -------
        list[LexicalItem]
            LexicalItem objects for matching verb classes.
        """
        loader = self._get_loader()
        assert isinstance(loader, VerbNetLoader)

        include_frames = kwargs.get("include_frames", False)
        items: list[LexicalItem] = []

        # Search through all verb classes
        for verb_class in loader.classes.values():
            if not verb_class.members:
                continue

            for member in verb_class.members:
                # Filter by query if provided
                if query is not None and member.name != query:
                    continue

                # Build attributes
                attributes: dict[str, Any] = {
                    "verbnet_class": verb_class.id,
                    "themroles": [r.type for r in verb_class.themroles]
                    if verb_class.themroles
                    else [],
                    "frame_count": len(verb_class.frames) if verb_class.frames else 0,
                }

                # Add detailed frame information if requested
                if include_frames and verb_class.frames:
                    frames_data = []
                    for frame in verb_class.frames:
                        frame_dict: dict[str, Any] = {
                            "primary": frame.description.primary,
                            "secondary": frame.description.secondary,
                        }

                        # Extract syntax elements
                        if frame.syntax and hasattr(frame.syntax, "elements"):
                            syntax_elements = []
                            for element in frame.syntax.elements:
                                pos = element.pos
                                value = (
                                    element.value if hasattr(element, "value") else None
                                )
                                syntax_elements.append((pos, value))
                            frame_dict["syntax"] = syntax_elements
                        else:
                            frame_dict["syntax"] = []

                        # Extract examples
                        if frame.examples:
                            frame_dict["examples"] = [ex.text for ex in frame.examples]
                        else:
                            frame_dict["examples"] = []

                        frames_data.append(frame_dict)

                    attributes["frames"] = frames_data

                # Create LexicalItem for this verb class
                features = {"pos": "VERB", **attributes}
                item = LexicalItem(
                    lemma=member.name,
                    language_code=language_code or "en",
                    features=features,
                    source="VerbNet",
                )
                items.append(item)

        return items

    def _fetch_propbank(
        self, query: str | None, language_code: LanguageCode, **kwargs: Any
    ) -> list[LexicalItem]:
        """Fetch from PropBank using PropBankLoader.

        Parameters
        ----------
        query : str | None
            Predicate lemma to search for. If None, fetch ALL predicates.
        language_code : LanguageCode
            Language code filter.
        **kwargs : Any
            Additional parameters:
            - include_frames (bool): Include detailed frame/roleset information.

        Returns
        -------
        list[LexicalItem]
            LexicalItem objects for matching predicates.
        """
        loader = self._get_loader()
        assert isinstance(loader, PropBankLoader)

        include_frames = kwargs.get("include_frames", False)
        items: list[LexicalItem] = []

        # If query is None, iterate through all framesets
        if query is None:
            # Get all framesets from PropBank
            for frameset in loader.framesets.values():
                items.extend(
                    self._create_propbank_items(frameset, language_code, include_frames)
                )
        else:
            # Get specific frameset for the predicate
            frameset = loader.get_frameset(query)
            if frameset:
                items.extend(
                    self._create_propbank_items(frameset, language_code, include_frames)
                )

        return items

    def _create_propbank_items(
        self, frameset: Any, language_code: LanguageCode, include_frames: bool
    ) -> list[LexicalItem]:
        """Create LexicalItem objects from a PropBank frameset.

        Parameters
        ----------
        frameset : Any
            PropBank frameset object.
        language_code : LanguageCode
            Language code filter.
        include_frames : bool
            Whether to include detailed roleset information.

        Returns
        -------
        list[LexicalItem]
            LexicalItem objects for the frameset's rolesets.
        """
        items: list[LexicalItem] = []

        if not frameset.rolesets:
            return items

        for roleset in frameset.rolesets:
            attributes: dict[str, Any] = {
                "propbank_roleset_id": roleset.id,
                "roleset_name": roleset.name if roleset.name else "",
            }

            # Add detailed role information if requested
            if include_frames and roleset.roles:
                attributes["roles"] = [
                    {
                        "arg": role.n,
                        "function": role.f,
                        "description": role.descr,
                    }
                    for role in roleset.roles
                ]

                # Add examples if available
                if hasattr(roleset, "examples") and roleset.examples:
                    attributes["examples"] = [
                        ex.text for ex in roleset.examples if hasattr(ex, "text")
                    ]

            # Create LexicalItem for each roleset
            # Use predicate_lemma attribute from PropBank frameset
            lemma = (
                frameset.predicate_lemma
                if hasattr(frameset, "predicate_lemma")
                else str(frameset)
            )
            features = {"pos": "VERB", **attributes}
            item = LexicalItem(
                lemma=lemma,
                language_code=language_code or "en",
                features=features,
                source="PropBank",
            )
            items.append(item)

        return items

    def _fetch_framenet(
        self, query: str | None, language_code: LanguageCode, **kwargs: Any
    ) -> list[LexicalItem]:
        """Fetch from FrameNet using FrameNetLoader.

        Parameters
        ----------
        query : str | None
            Lemma to search for. If None, fetch ALL lexical units.
        language_code : LanguageCode
            Language code filter.
        **kwargs : Any
            Additional parameters:
            - include_frames (bool): Include detailed frame information.

        Returns
        -------
        list[LexicalItem]
            LexicalItem objects for matching lexical units.
        """
        loader = self._get_loader()
        assert isinstance(loader, FrameNetLoader)

        include_frames = kwargs.get("include_frames", False)
        items: list[LexicalItem] = []

        # Iterate through all frames and their lexical units
        for frame in loader.frames:
            if not frame.lexical_units:
                continue

            for lu in frame.lexical_units:
                # Extract lemma from lexical unit name (format: "lemma.pos")
                lemma = lu.name.split(".")[0] if "." in lu.name else lu.name

                # Filter by query if provided
                if query is not None and lemma != query:
                    continue

                # Create LexicalItem for this lexical unit
                item = self._create_framenet_item(
                    lu, frame, language_code, include_frames
                )
                items.append(item)

        return items

    def _create_framenet_item(
        self, lu: Any, frame: Any, language_code: LanguageCode, include_frames: bool
    ) -> LexicalItem:
        """Create a LexicalItem from a FrameNet lexical unit.

        Parameters
        ----------
        lu : Any
            FrameNet LexicalUnit object.
        frame : Any
            FrameNet Frame object containing the lexical unit.
        language_code : LanguageCode
            Language code filter.
        include_frames : bool
            Whether to include detailed frame information.

        Returns
        -------
        LexicalItem
            LexicalItem object for the lexical unit.
        """
        # Extract lemma from lexical unit name (format: "lemma.pos")
        lemma = lu.name.split(".")[0] if "." in lu.name else lu.name

        # Map FrameNet POS to standard POS tags
        pos_map = {"V": "VERB", "N": "NOUN", "A": "ADJ", "ADV": "ADV", "PREP": "ADP"}
        pos = pos_map.get(lu.pos, "VERB")

        # Build attributes
        attributes: dict[str, Any] = {
            "framenet_frame": frame.name,
            "framenet_frame_id": frame.id,
            "lexical_unit_name": lu.name,
            "lexical_unit_id": lu.id,
        }

        # Add definition if available
        if hasattr(lu, "definition") and lu.definition:
            attributes["definition"] = lu.definition

        # Add detailed frame information if requested
        if include_frames:
            attributes["frame_definition"] = frame.definition

            # Add frame elements (semantic roles)
            if frame.frame_elements:
                attributes["frame_elements"] = [
                    {
                        "name": fe.name,
                        "core_type": fe.core_type,
                        "definition": fe.definition
                        if hasattr(fe, "definition")
                        else None,
                    }
                    for fe in frame.frame_elements
                ]

        features = {"pos": pos, **attributes}
        return LexicalItem(
            lemma=lemma,
            language_code=language_code or "en",
            features=features,
            source="FrameNet",
        )

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
