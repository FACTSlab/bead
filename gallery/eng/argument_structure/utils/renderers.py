"""Custom template renderers for English argument structure experiments.

This module provides English-specific rendering logic for handling repeated
nouns with "another"/"the other" patterns in argument structure templates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bead.templates.renderers import TemplateRenderer

if TYPE_CHECKING:
    from bead.resources.lexical_item import LexicalItem
    from bead.resources.template import Slot


def _count_noun_occurrences(
    slot_fillers: Mapping[str, LexicalItem],
    template_slots: Mapping[str, Slot],
) -> dict[str, int]:
    """Count how many times each noun lemma appears in the template.

    Parameters
    ----------
    slot_fillers : Mapping[str, LexicalItem]
        Mapping from slot names to lexical items.
    template_slots : Mapping[str, Slot]
        Mapping from slot names to slot definitions.

    Returns
    -------
    dict[str, int]
        Mapping from noun lemmas to their occurrence counts.
    """
    noun_counts: dict[str, int] = {}

    for slot_name in template_slots:
        if slot_name.startswith("noun_") and slot_name in slot_fillers:
            noun_lemma = slot_fillers[slot_name].lemma
            noun_counts[noun_lemma] = noun_counts.get(noun_lemma, 0) + 1

    return noun_counts


def _ordinal_word(n: int) -> str:
    """Convert number to ordinal word (e.g., 1 -> 'first', 2 -> 'second').

    Parameters
    ----------
    n : int
        Number to convert (1-10).

    Returns
    -------
    str
        Ordinal word representation.
    """
    ordinals = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }
    return ordinals.get(n, f"{n}th")


class OtherNounRenderer(TemplateRenderer):
    """Renderer with special handling for repeated nouns in English.

    This renderer handles repeated noun slots in English argument structure
    templates by using "another"/"the other" for second occurrences and
    ordinals ("a second", "a third", etc.) for subsequent occurrences.

    The rendering rules are:
    1. First occurrence: Use original determiner + noun (e.g., "a cat")
    2. Second occurrence when total=2: Use "another"/"the other" based on determiner
    3. Third+ occurrence: Use ordinals (e.g., "a second cat", "a third cat")

    This renderer specifically targets determiner-noun pairs where:
    - Noun slot names start with "noun_" (e.g., "noun_subj", "noun_dobj")
    - Corresponding determiner slots are "det_" + suffix (e.g., "det_subj")

    Examples
    --------
    >>> from bead.resources.lexical_item import LexicalItem
    >>> from bead.resources.template import Slot
    >>> renderer = OtherNounRenderer()
    >>>
    >>> # Template with repeated noun
    >>> template_string = "{det_1} {noun_1} and {det_2} {noun_2}"
    >>> slot_fillers = {
    ...     "det_1": LexicalItem(lemma="a", language_code="eng"),
    ...     "noun_1": LexicalItem(lemma="cat", language_code="eng"),
    ...     "det_2": LexicalItem(lemma="a", language_code="eng"),
    ...     "noun_2": LexicalItem(lemma="cat", language_code="eng"),  # Same noun!
    ... }
    >>> slots = {name: Slot(name=name) for name in slot_fillers.keys()}
    >>> renderer.render(template_string, slot_fillers, slots)
    'a cat and another cat'
    """

    def render(
        self,
        template_string: str,
        slot_fillers: Mapping[str, LexicalItem],
        template_slots: Mapping[str, Slot],
    ) -> str:
        """Render template with "other" handling for repeated nouns.

        Parameters
        ----------
        template_string : str
            Template string with {slot_name} placeholders.
        slot_fillers : Mapping[str, LexicalItem]
            Mapping from slot names to lexical items.
        template_slots : Mapping[str, Slot]
            Mapping from slot names to slot definitions.

        Returns
        -------
        str
            Rendered text with proper "another"/"the other" handling.
        """
        # Count total occurrences of each noun
        noun_total_counts = _count_noun_occurrences(slot_fillers, template_slots)

        # Identify det+noun pairs in template order
        # Find position of each slot in template string
        slot_positions: dict[str, int] = {}
        for slot_name in template_slots.keys():
            placeholder = f"{{{slot_name}}}"
            pos = template_string.find(placeholder)
            if pos >= 0:
                slot_positions[slot_name] = pos

        # Sort noun slots by their position in template
        noun_slots_in_order = [
            name
            for name in sorted(slot_positions.keys(), key=lambda x: slot_positions[x])
            if name.startswith("noun_")
        ]

        # Build det+noun pairs in template order
        det_noun_pairs: list[tuple[str, str]] = []
        for noun_slot in noun_slots_in_order:
            suffix = noun_slot[5:]  # Remove "noun_" prefix
            det_slot = f"det_{suffix}"
            if det_slot in slot_fillers and noun_slot in slot_fillers:
                det_noun_pairs.append((det_slot, noun_slot))

        # Track current occurrence of each noun
        noun_usage: dict[str, int] = {}

        # Build result by replacing det+noun pairs
        result = template_string

        for det_slot, noun_slot in det_noun_pairs:
            det_item = slot_fillers[det_slot]
            noun_item = slot_fillers[noun_slot]

            det_surface = det_item.form if det_item.form else det_item.lemma
            noun_surface = noun_item.form if noun_item.form else noun_item.lemma
            noun_lemma = noun_item.lemma

            occurrence = noun_usage.get(noun_lemma, 0)
            total_occurrences = noun_total_counts[noun_lemma]

            # Determine rendering based on occurrence
            if occurrence == 0:
                # First occurrence: use original determiner
                rendered = f"{det_surface} {noun_surface}"
            elif total_occurrences == 2:
                # Exactly 2 occurrences: use "another" or "the other"
                if det_surface.lower() == "a":
                    rendered = f"another {noun_surface}"
                elif det_surface.lower() == "the":
                    rendered = f"the other {noun_surface}"
                else:
                    rendered = f"another {noun_surface}"
            else:
                # 3+ occurrences: use ordinals
                ordinal = _ordinal_word(occurrence + 1)
                rendered = f"a {ordinal} {noun_surface}"

            # Replace det+noun pair
            pattern = f"{{{det_slot}}} {{{noun_slot}}}"
            result = result.replace(pattern, rendered, 1)

            noun_usage[noun_lemma] = occurrence + 1

        # Handle remaining slots (verbs, preps, etc.) with simple substitution
        for slot_name, item in slot_fillers.items():
            placeholder = f"{{{slot_name}}}"
            if placeholder in result:
                surface = item.form if item.form else item.lemma
                result = result.replace(placeholder, surface)

        return result
