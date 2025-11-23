"""Template rendering system with plugin support.

This module provides the base template rendering interface and a simple
default implementation. Language-specific or custom rendering logic should
be implemented as external plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bead.resources.lexical_item import LexicalItem
    from bead.resources.template import Slot


class TemplateRenderer(ABC):
    """Base class for template renderers.

    Custom renderers should subclass this and implement the render() method.
    This allows for language-specific or experiment-specific rendering logic
    to be implemented as external plugins.

    Examples
    --------
    >>> from bead.templates.renderers import TemplateRenderer
    >>> class CustomRenderer(TemplateRenderer):
    ...     def render(self, template_string, slot_fillers, template_slots):
    ...         # Custom rendering logic here
    ...         return rendered_text
    """

    @abstractmethod
    def render(
        self,
        template_string: str,
        slot_fillers: Mapping[str, LexicalItem],
        template_slots: Mapping[str, Slot],
    ) -> str:
        """Render template with slot fillers.

        Parameters
        ----------
        template_string : str
            Template string with {slot_name} placeholders.
        slot_fillers : Mapping[str, LexicalItem]
            Mapping from slot names to lexical items that fill them.
        template_slots : Mapping[str, Slot]
            Mapping from slot names to slot definitions. Provides access
            to slot constraints and metadata for context-aware rendering.

        Returns
        -------
        str
            Rendered text with placeholders replaced.

        Examples
        --------
        >>> renderer = DefaultRenderer()
        >>> from bead.resources.lexical_item import LexicalItem
        >>> from bead.resources.template import Slot
        >>> fillers = {
        ...     "subj": LexicalItem(lemma="cat", language_code="eng"),
        ...     "verb": LexicalItem(lemma="run", form="runs", language_code="eng")
        ... }
        >>> slots = {
        ...     "subj": Slot(name="subj"),
        ...     "verb": Slot(name="verb")
        ... }
        >>> renderer.render("{subj} {verb}", fillers, slots)
        'cat runs'
        """
        ...


class DefaultRenderer(TemplateRenderer):
    """Default renderer using simple slot substitution.

    Uses item.form if available, otherwise item.lemma.
    This is a language-agnostic implementation suitable for most use cases.

    Examples
    --------
    >>> from bead.templates.renderers import DefaultRenderer
    >>> from bead.resources.lexical_item import LexicalItem
    >>> from bead.resources.template import Slot
    >>> renderer = DefaultRenderer()
    >>>
    >>> # Example 1: Basic rendering with lemmas
    >>> fillers = {
    ...     "det": LexicalItem(lemma="the", language_code="eng"),
    ...     "noun": LexicalItem(lemma="cat", language_code="eng"),
    ...     "verb": LexicalItem(lemma="run", language_code="eng")
    ... }
    >>> slots = {name: Slot(name=name) for name in fillers.keys()}
    >>> renderer.render("{det} {noun} {verb}", fillers, slots)
    'the cat run'
    >>>
    >>> # Example 2: Rendering with forms (inflected)
    >>> fillers_with_forms = {
    ...     "det": LexicalItem(lemma="the", language_code="eng"),
    ...     "noun": LexicalItem(lemma="cat", form="cats", language_code="eng"),
    ...     "verb": LexicalItem(lemma="run", form="runs", language_code="eng")
    ... }
    >>> renderer.render("{det} {noun} {verb}", fillers_with_forms, slots)
    'the cats runs'
    """

    def render(
        self,
        template_string: str,
        slot_fillers: Mapping[str, LexicalItem],
        template_slots: Mapping[str, Slot],
    ) -> str:
        """Render template with simple slot substitution.

        Uses item.form if available, otherwise item.lemma. This provides
        a straightforward rendering suitable for most templates without
        special language-specific handling.

        Parameters
        ----------
        template_string : str
            Template string with {slot_name} placeholders.
        slot_fillers : Mapping[str, LexicalItem]
            Mapping from slot names to lexical items.
        template_slots : Mapping[str, Slot]
            Mapping from slot names to slot definitions (unused in default
            implementation, but provided for subclasses).

        Returns
        -------
        str
            Rendered text with placeholders replaced by item forms or lemmas.

        Notes
        -----
        This renderer prioritizes item.form over item.lemma to support
        morphological inflection. If item.form is None, it falls back to
        item.lemma.

        The template_slots parameter is not used in the default implementation
        but is available for subclasses that need slot metadata for rendering.

        Examples
        --------
        >>> renderer = DefaultRenderer()
        >>> from bead.resources.lexical_item import LexicalItem
        >>> from bead.resources.template import Slot
        >>>
        >>> # Form takes precedence over lemma
        >>> fillers = {
        ...     "verb": LexicalItem(lemma="walk", form="walked", language_code="eng")
        ... }
        >>> slots = {"verb": Slot(name="verb")}
        >>> renderer.render("{verb}", fillers, slots)
        'walked'
        >>>
        >>> # Falls back to lemma when form is None
        >>> fillers_no_form = {
        ...     "verb": LexicalItem(lemma="walk", form=None, language_code="eng")
        ... }
        >>> renderer.render("{verb}", fillers_no_form, slots)
        'walk'
        """
        result = template_string

        for slot_name, item in slot_fillers.items():
            placeholder = f"{{{slot_name}}}"
            if placeholder in result:
                # Use form if available, otherwise lemma
                surface = item.form if item.form is not None else item.lemma
                result = result.replace(placeholder, surface)

        return result
