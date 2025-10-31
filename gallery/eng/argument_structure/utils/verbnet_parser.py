"""VerbNet data extraction using bead adapters.

This module wraps GlazingAdapter to extract all VerbNet verbs and frames
for the argument structure project. It uses the bead adapter infrastructure
to provide project-specific extraction methods while maintaining consistency
with the overall bead architecture.
"""

from __future__ import annotations

from bead.resources.adapters.cache import AdapterCache
from bead.resources.adapters.glazing import GlazingAdapter
from bead.resources.models import LexicalItem


class VerbNetExtractor:
    """Extract VerbNet data using bead adapters.

    This is a convenience wrapper around GlazingAdapter that provides
    project-specific extraction methods while using the standard sash
    adapter infrastructure.

    Parameters
    ----------
    cache : AdapterCache | None
        Optional cache instance for adapter results.

    Examples
    --------
    >>> extractor = VerbNetExtractor()
    >>> verbs = extractor.extract_all_verbs()
    >>> len(verbs) > 3000
    True
    >>> frames = extractor.extract_all_verbs_with_frames()
    >>> "frames" in frames[0].attributes
    True
    """

    def __init__(self, cache: AdapterCache | None = None) -> None:
        """Initialize extractor with optional cache.

        Parameters
        ----------
        cache : AdapterCache | None
            Optional cache for adapter results.
        """
        self.adapter = GlazingAdapter(resource="verbnet", cache=cache)

    def extract_all_verbs(self) -> list[LexicalItem]:
        """Extract all VerbNet verbs as LexicalItem objects.

        Returns LexicalItem objects with basic metadata in attributes:
        - verbnet_class: Class ID
        - themroles: List of thematic roles
        - frame_count: Number of frames

        Returns
        -------
        list[LexicalItem]
            All VerbNet verbs with basic metadata.

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> verbs = extractor.extract_all_verbs()
        >>> "break" in [v.lemma for v in verbs]
        True
        """
        return self.adapter.fetch_items(query=None, language_code="en")

    def extract_all_verbs_with_frames(self) -> list[LexicalItem]:
        """Extract all VerbNet verbs with detailed frame information.

        Returns LexicalItem objects with detailed frame data in attributes:
        - verbnet_class: Class ID
        - themroles: List of thematic roles
        - frame_count: Number of frames
        - frames: List of frame dictionaries with:
            - primary: Primary frame description
            - secondary: Secondary frame description
            - syntax: List of (pos, value) tuples
            - examples: List of example sentences

        Returns
        -------
        list[LexicalItem]
            All VerbNet verbs with detailed frame information.

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> verbs = extractor.extract_all_verbs_with_frames()
        >>> "frames" in verbs[0].attributes
        True
        >>> len(verbs[0].attributes["frames"]) > 0
        True
        """
        return self.adapter.fetch_items(
            query=None, language_code="en", include_frames=True
        )

    def get_verb_classes(self, lemma: str) -> list[str]:
        """Get all VerbNet classes for a verb.

        Parameters
        ----------
        lemma : str
            Verb lemma (e.g., "break", "run").

        Returns
        -------
        list[str]
            List of VerbNet class IDs for this verb.

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> classes = extractor.get_verb_classes("break")
        >>> len(classes) > 0
        True
        """
        items = self.adapter.fetch_items(query=lemma, language_code="en")
        return [item.attributes["verbnet_class"] for item in items]

    def get_verb_with_frames(self, lemma: str) -> list[LexicalItem]:
        """Get verb with detailed frame information.

        Parameters
        ----------
        lemma : str
            Verb lemma (e.g., "break", "run").

        Returns
        -------
        list[LexicalItem]
            LexicalItem objects for each verb-class pair with frames.

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> items = extractor.get_verb_with_frames("break")
        >>> "frames" in items[0].attributes
        True
        """
        return self.adapter.fetch_items(
            query=lemma, language_code="en", include_frames=True
        )

    def is_particle_verb(self, lemma: str) -> bool:
        """Check if verb has particle (space or hyphen).

        Parameters
        ----------
        lemma : str
            Verb lemma.

        Returns
        -------
        bool
            True if particle verb (contains space or hyphen).

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> extractor.is_particle_verb("turn off")
        True
        >>> extractor.is_particle_verb("break")
        False
        """
        return " " in lemma or "-" in lemma

    def get_particle_verbs(self) -> list[LexicalItem]:
        """Get all particle verbs.

        Returns
        -------
        list[LexicalItem]
            All verbs containing spaces or hyphens.

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> particle_verbs = extractor.get_particle_verbs()
        >>> all(extractor.is_particle_verb(v.lemma) for v in particle_verbs)
        True
        """
        all_verbs = self.extract_all_verbs()
        return [v for v in all_verbs if self.is_particle_verb(v.lemma)]

    def get_clausal_frames(self) -> list[dict]:
        """Get all frames with clausal complements.

        Returns frames containing clausal patterns (S, that, whether, if, to, etc.).

        Returns
        -------
        list[dict]
            Dictionaries with:
            - lemma: Verb lemma
            - verbnet_class: Class ID
            - frame: Frame dictionary

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> clausal = extractor.get_clausal_frames()
        >>> len(clausal) > 0
        True
        """
        all_verbs = self.extract_all_verbs_with_frames()
        clausal_keywords = [
            "S",
            "that",
            "whether",
            "if",
            "to",
            "SCOMP",
            "VCOMP",
            "wh-",
        ]

        clausal_frames = []
        for verb in all_verbs:
            if "frames" not in verb.attributes:
                continue

            for frame in verb.attributes["frames"]:
                if any(kw in frame["primary"] for kw in clausal_keywords):
                    clausal_frames.append(
                        {
                            "lemma": verb.lemma,
                            "verbnet_class": verb.attributes["verbnet_class"],
                            "frame": frame,
                        }
                    )

        return clausal_frames

    def get_frames_with_pp(self) -> list[dict]:
        """Get all frames with prepositional phrases.

        Returns frames containing PP patterns.

        Returns
        -------
        list[dict]
            Dictionaries with:
            - lemma: Verb lemma
            - verbnet_class: Class ID
            - frame: Frame dictionary

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> pp_frames = extractor.get_frames_with_pp()
        >>> len(pp_frames) > 0
        True
        """
        all_verbs = self.extract_all_verbs_with_frames()

        pp_frames = []
        for verb in all_verbs:
            if "frames" not in verb.attributes:
                continue

            for frame in verb.attributes["frames"]:
                if "PP" in frame["primary"]:
                    pp_frames.append(
                        {
                            "lemma": verb.lemma,
                            "verbnet_class": verb.attributes["verbnet_class"],
                            "frame": frame,
                        }
                    )

        return pp_frames

    def export_to_lexicon_format(self, verbs: list[LexicalItem]) -> list[dict]:
        """Export verbs to format suitable for Lexicon.from_dicts().

        Parameters
        ----------
        verbs : list[LexicalItem]
            Verbs to export.

        Returns
        -------
        list[dict]
            Dictionaries suitable for Lexicon construction.

        Examples
        --------
        >>> extractor = VerbNetExtractor()
        >>> verbs = extractor.extract_all_verbs()[:5]
        >>> dicts = extractor.export_to_lexicon_format(verbs)
        >>> all("lemma" in d and "pos" in d for d in dicts)
        True
        """
        return [
            {
                "lemma": v.lemma,
                "pos": v.pos,
                "language_code": v.language_code,
                "features": v.features,
                "attributes": v.attributes,
            }
            for v in verbs
        ]
