"""Morphological paradigm extraction using bead adapters.

This module wraps UniMorphAdapter to extract verb forms for the
argument structure project. It uses the bead adapter infrastructure
to provide project-specific convenience methods for retrieving
required verb inflections.
"""

from __future__ import annotations

from bead.resources.adapters.cache import AdapterCache
from bead.resources.adapters.unimorph import UniMorphAdapter
from bead.resources.lexical_item import LexicalItem


class MorphologyExtractor:
    """Extract verb forms using UniMorphAdapter.

    This wrapper provides project-specific convenience methods while
    using the standard bead adapter infrastructure for morphological
    paradigm extraction.

    Parameters
    ----------
    cache : AdapterCache | None
        Optional cache for adapter results.

    Examples
    --------
    >>> extractor = MorphologyExtractor()
    >>> forms = extractor.get_verb_forms("walk")
    >>> "3sg_present" in forms
    True
    """

    def __init__(self, cache: AdapterCache | None = None) -> None:
        """Initialize extractor with optional cache.

        Parameters
        ----------
        cache : AdapterCache | None
            Optional cache for adapter results.
        """
        self.adapter = UniMorphAdapter(cache=cache)

    def get_verb_forms(self, lemma: str) -> dict[str, LexicalItem | None]:
        """Get required verb forms as LexicalItem objects.

        Extracts the four core verb forms needed for template filling:
        - 3sg_present: 3rd person singular present (e.g., "walks")
        - past: Simple past (e.g., "walked")
        - present_participle: Present participle (e.g., "walking")
        - past_participle: Past participle (e.g., "walked")

        Parameters
        ----------
        lemma : str
            Verb lemma (e.g., "walk", "run").

        Returns
        -------
        dict[str, LexicalItem | None]
            Dictionary mapping form names to LexicalItem objects.
            Returns None for forms not found in UniMorph.

        Examples
        --------
        >>> extractor = MorphologyExtractor()
        >>> forms = extractor.get_verb_forms("walk")
        >>> forms["3sg_present"].form
        'walks'
        """
        # Fetch all forms for this lemma
        items = self.adapter.fetch_items(query=lemma, language_code="en")

        # Map to required forms
        forms: dict[str, LexicalItem | None] = {
            "3sg_present": None,
            "past": None,
            "present_participle": None,
            "past_participle": None,
        }

        for item in items:
            # Check features dict for matching forms
            # UniMorphAdapter parses features into dimensions
            if item.features.get("pos") != "V":
                continue

            tense = item.features.get("tense")
            person = item.features.get("person")
            number = item.features.get("number")
            verb_form = item.features.get("verb_form")

            # 3sg present: V;PRS;3;SG
            if tense == "PRS" and person == "3" and number == "SG":
                forms["3sg_present"] = item

            # Past: V;PST
            elif tense == "PST" and not verb_form:
                forms["past"] = item

            # Present participle: V;V.PTCP;PRS
            elif verb_form == "V.PTCP" and tense == "PRS":
                forms["present_participle"] = item

            # Past participle: V;V.PTCP;PST
            elif verb_form == "V.PTCP" and tense == "PST":
                forms["past_participle"] = item

        return forms

    def handle_particle_verb(self, lemma: str) -> dict[str, LexicalItem | None]:
        """Handle multi-word predicates like 'turn off' or 'cross-examine'.

        Strategy: Extract forms for the main verb component and construct
        particle verb forms by combining the inflected main verb with
        the particle.

        Parameters
        ----------
        lemma : str
            Multi-word verb (e.g., "turn off", "cross-examine").

        Returns
        -------
        dict[str, LexicalItem | None]
            Verb forms with particle (may need manual construction).

        Examples
        --------
        >>> extractor = MorphologyExtractor()
        >>> forms = extractor.handle_particle_verb("turn off")
        >>> forms["3sg_present"].lemma
        'turn off'
        """
        # Split on space or hyphen
        parts = lemma.replace("-", " ").split()

        if len(parts) < 2:
            # Not actually multi-word
            return self.get_verb_forms(lemma)

        # Get forms for the main verb (first element for most cases)
        main_verb = parts[0]
        base_forms = self.get_verb_forms(main_verb)

        # Construct particle verb forms
        particle_forms: dict[str, LexicalItem | None] = {}
        for key, item in base_forms.items():
            if item is None:
                particle_forms[key] = None
            else:
                # Create new item with particle verb lemma
                # Keep the inflected form from base verb
                particle_forms[key] = LexicalItem(
                    lemma=lemma,  # Full particle verb
                    form=item.form,  # Inflected form from base
                    language_code=item.language_code,
                    features={
                        **item.features,
                        "base_verb": main_verb,
                        "is_particle_verb": True,
                    },
                )

        return particle_forms

    def create_progressive_forms(
        self, present_participle: LexicalItem
    ) -> dict[str, LexicalItem]:
        """Create progressive forms with auxiliaries.

        Constructs present and past progressive forms by combining
        the appropriate auxiliary (is/was) with the present participle.

        Parameters
        ----------
        present_participle : LexicalItem
            Present participle form (e.g., "walking").

        Returns
        -------
        dict[str, LexicalItem]
            Keys: "present_progressive_3sg", "past_progressive_3sg"

        Examples
        --------
        >>> extractor = MorphologyExtractor()
        >>> forms = extractor.get_verb_forms("walk")
        >>> prog = extractor.create_progressive_forms(forms["present_participle"])
        >>> prog["present_progressive_3sg"].form
        'is walking'
        """
        base_lemma = present_participle.lemma
        participle_form = present_participle.form or base_lemma

        # Create progressive forms with auxiliaries
        present_prog = LexicalItem(
            lemma=base_lemma,
            form=f"is {participle_form}",
            language_code="eng",
            features={
                "pos": "VERB",
                "tense": "PRS",
                "aspect": "PROG",
                "person": "3",
                "number": "SG",
                "construction": "progressive",
                "auxiliary": "is",
                "participle": participle_form,
            },
            source="UniMorph",
        )

        past_prog = LexicalItem(
            lemma=base_lemma,
            form=f"was {participle_form}",
            language_code="eng",
            features={
                "pos": "VERB",
                "tense": "PST",
                "aspect": "PROG",
                "person": "3",
                "number": "SG",
                "construction": "progressive",
                "auxiliary": "was",
                "participle": participle_form,
            },
        )

        return {
            "present_progressive_3sg": present_prog,
            "past_progressive_3sg": past_prog,
        }

    def get_all_required_forms(self, lemma: str) -> list[LexicalItem]:
        """Get all required forms for templates.

        Extracts all verb forms needed for template filling, including
        both simple forms and progressive constructions.

        Parameters
        ----------
        lemma : str
            Verb lemma.

        Returns
        -------
        list[LexicalItem]
            All required forms (simple + progressive).

        Examples
        --------
        >>> extractor = MorphologyExtractor()
        >>> all_forms = extractor.get_all_required_forms("walk")
        >>> len(all_forms) >= 4
        True
        """
        # Handle particle verbs
        if " " in lemma or "-" in lemma:
            forms = self.handle_particle_verb(lemma)
        else:
            forms = self.get_verb_forms(lemma)

        # Collect non-None forms
        result = [item for item in forms.values() if item is not None]

        # Add progressive forms if we have present participle
        if forms["present_participle"] is not None:
            prog_forms = self.create_progressive_forms(forms["present_participle"])
            result.extend(prog_forms.values())

        return result
