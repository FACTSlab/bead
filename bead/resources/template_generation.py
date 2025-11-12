"""Abstract base class for mapping external frame inventories to Templates.

This module provides language-agnostic base classes for generating Template
objects from external linguistic frame inventories (e.g., VerbNet, FrameNet,
PropBank, valency lexicons).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bead.resources.constraints import Constraint
from bead.resources.template import Slot, Template


class FrameToTemplateMapper(ABC):
    """Abstract base class for mapping frame inventories to Templates.

    This class provides a framework for generating Template objects from
    external linguistic frame data. Subclasses implement language- and
    resource-specific mapping logic.

    Examples
    --------
    Implementing a VerbNet mapper:
    >>> class VerbNetMapper(FrameToTemplateMapper):
    ...     def generate_from_frame(self, verb_lemma, frame_data):
    ...         slots = self.map_frame_to_slots(frame_data)
    ...         constraints = self.generate_constraints(frame_data, slots)
    ...         return Template(
    ...             name=f"{verb_lemma}_{frame_data['id']}",
    ...             template_string=frame_data['template_string'],
    ...             slots=slots,
    ...             constraints=constraints
    ...         )
    ...
    ...     def map_frame_to_slots(self, frame_data):
    ...         # Extract slots from VerbNet syntax
    ...         return {}
    ...
    ...     def generate_constraints(self, frame_data, slots):
    ...         # Generate constraints from VerbNet restrictions
    ...         return []
    """

    @abstractmethod
    def generate_from_frame(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Template | list[Template]:
        """Generate Template(s) from a frame specification.

        This is the main entry point for template generation. Subclasses
        implement the specific logic for their frame inventory.

        Parameters
        ----------
        *args : Any
            Positional arguments (frame data, identifiers, etc.).
        **kwargs : Any
            Keyword arguments (configuration options, etc.).

        Returns
        -------
        Template | list[Template]
            Generated template(s). May return multiple templates if the
            frame has multiple realizations (e.g., different complementizer
            types, alternations).

        Examples
        --------
        VerbNet implementation:
        >>> mapper.generate_from_frame(
        ...     verb_lemma="think",
        ...     verbnet_class="29.9",
        ...     frame_data={"primary": "NP V that S"}
        ... )  # doctest: +SKIP
        """
        ...

    @abstractmethod
    def map_frame_to_slots(
        self,
        frame_data: Any,
    ) -> dict[str, Slot]:
        """Map frame elements to Template slots.

        Converts frame-specific element descriptions into Slot objects
        with appropriate constraints.

        Parameters
        ----------
        frame_data : Any
            Frame specification from the external inventory.
            Type depends on the specific resource (dict, object, etc.).

        Returns
        -------
        dict[str, Slot]
            Slots keyed by slot name.

        Examples
        --------
        Mapping VerbNet syntax to slots:
        >>> slots = mapper.map_frame_to_slots({
        ...     "syntax": [
        ...         ("NP", "Agent"),
        ...         ("V", None),
        ...         ("NP", "Theme")
        ...     ]
        ... })  # doctest: +SKIP
        >>> "subject" in slots
        True
        """
        ...

    @abstractmethod
    def generate_constraints(
        self,
        frame_data: Any,
        slots: dict[str, Slot],
    ) -> list[Constraint]:
        """Generate multi-slot constraints from frame specifications.

        Converts frame-specific restrictions into DSL Constraint objects
        that enforce relationships between slots.

        Parameters
        ----------
        frame_data : Any
            Frame specification from the external inventory.
        slots : dict[str, Slot]
            Slots that have been created for this frame.

        Returns
        -------
        list[Constraint]
            Multi-slot constraints for the template.

        Examples
        --------
        Generating constraints from VerbNet restrictions:
        >>> constraints = mapper.generate_constraints(
        ...     frame_data={"restrictions": [...]},
        ...     slots={"subject": ..., "verb": ...}
        ... )  # doctest: +SKIP
        """
        ...

    def create_template_name(
        self,
        *identifiers: str,
        separator: str = "_",
    ) -> str:
        """Create a unique template name from identifiers.

        Utility method for generating consistent template names.
        Sanitizes identifiers by replacing spaces, dots, and hyphens.

        Parameters
        ----------
        *identifiers : str
            Components to include in the name (e.g., verb, class, frame).
        separator : str
            Separator between components (default: "_").

        Returns
        -------
        str
            Sanitized template name.

        Examples
        --------
        >>> mapper = ConcreteMapper()
        >>> mapper.create_template_name("think", "29.9", "that-clause")
        'think_29_9_that_clause'
        """
        # Sanitize each identifier
        sanitized: list[str] = []
        for identifier in identifiers:
            safe: str = (
                identifier.replace(" ", separator)
                .replace(".", separator)
                .replace("-", separator)
            )
            sanitized.append(safe)

        return separator.join(sanitized)

    def create_template_metadata(
        self,
        frame_data: dict[str, Any],
        **additional_metadata: Any,
    ) -> dict[str, Any]:
        """Create metadata dictionary for template.

        Utility method for extracting and organizing frame metadata.
        Subclasses can override to add resource-specific metadata.

        Parameters
        ----------
        frame_data : dict[str, Any]
            Frame specification from the external inventory.
        **additional_metadata : Any
            Additional metadata to include.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary for Template.metadata field.

        Examples
        --------
        >>> mapper = ConcreteMapper()
        >>> metadata = mapper.create_template_metadata(
        ...     frame_data={"id": "29.9-1", "examples": [...]},
        ...     verb_lemma="think"
        ... )  # doctest: +SKIP
        """
        metadata: dict[str, Any] = {}

        # Add frame data
        metadata.update(frame_data)

        # Add additional metadata
        metadata.update(additional_metadata)

        return metadata


class MultiFrameMapper(FrameToTemplateMapper):
    """Mapper that generates multiple template variants from a single frame.

    Some frame specifications support multiple realizations (e.g., different
    complementizer types, voice alternations). This class provides a framework
    for generating all variants.

    Examples
    --------
    >>> class ClausalMapper(MultiFrameMapper):
    ...     def get_frame_variants(self, frame_data):
    ...         # Return list of variant specifications
    ...         return [
    ...             {"comp": "that", "mood": "declarative"},
    ...             {"comp": "whether", "mood": "interrogative"},
    ...         ]
    ...
    ...     def generate_from_frame(self, verb, frame_data):
    ...         variants = self.get_frame_variants(frame_data)
    ...         return [self._generate_variant(verb, v) for v in variants]
    ...
    ...     def map_frame_to_slots(self, frame_data):
    ...         return {}
    ...
    ...     def generate_constraints(self, frame_data, slots):
    ...         return []
    """

    @abstractmethod
    def get_frame_variants(
        self,
        frame_data: Any,
    ) -> list[Any]:
        """Extract all variants from frame specification.

        Parameters
        ----------
        frame_data : Any
            Frame specification from the external inventory.

        Returns
        -------
        list[Any]
            List of variant specifications, each representing one possible
            realization of the frame.

        Examples
        --------
        >>> variants = mapper.get_frame_variants({
        ...     "complementizers": ["that", "whether", "if"]
        ... })  # doctest: +SKIP
        >>> len(variants)
        3
        """
        ...

    def generate_from_frame(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[Template]:
        """Generate templates for all frame variants.

        Default implementation calls get_frame_variants() and generates
        a template for each variant. Subclasses can override for custom logic.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to variant generation.
        **kwargs : Any
            Keyword arguments passed to variant generation.

        Returns
        -------
        list[Template]
            Templates for all variants.
        """
        # Extract frame_data from kwargs
        frame_data = kwargs.get("frame_data")
        if frame_data is None:
            raise ValueError("frame_data must be provided in kwargs")

        variants = self.get_frame_variants(frame_data)

        templates: list[Template] = []
        for variant in variants:
            # Create a modified kwargs with variant info
            variant_kwargs = kwargs.copy()
            variant_kwargs["variant_data"] = variant

            template: Template = self._generate_variant(*args, **variant_kwargs)
            templates.append(template)

        return templates

    @abstractmethod
    def _generate_variant(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Template:
        """Generate template for a single variant.

        Parameters
        ----------
        *args : Any
            Positional arguments.
        **kwargs : Any
            Keyword arguments, including variant_data.

        Returns
        -------
        Template
            Template for this variant.
        """
        ...
