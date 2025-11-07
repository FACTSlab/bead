"""Template generation from VerbNet frames.

This module generates Template objects from VerbNet frame data. It combines:
- VerbNet frame structures (via GlazingAdapter)
- MegaAttitude clausal complement mappings
- DSL constraint generation
- Morphological paradigms (via UniMorphAdapter)

The generated templates are compatible with the bead template infrastructure
and can be filled using the template filling system.
"""

from __future__ import annotations

from pathlib import Path

from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.template import Slot, Template

from .clausal_frames import ClausalTemplate, map_verbnet_to_clausal_templates
from .constraint_builder import (
    build_determiner_constraint,
)


class TemplateGenerator:
    """Generate Template objects from VerbNet frames.

    This class orchestrates the generation of templates from VerbNet data,
    applying appropriate constraints and slot definitions based on the
    frame structure.

    Parameters
    ----------
    resource_dir : str | Path | None
        Directory containing bleached lexicon CSV files.
        Defaults to "resources/" relative to this file.

    Examples
    --------
    >>> generator = TemplateGenerator()
    >>> frame_data = {
    ...     "primary": "NP V that S",
    ...     "examples": ["She thinks that he left."]
    ... }
    >>> templates = generator.generate_from_frame(
    ...     verb_lemma="think",
    ...     verbnet_class="29.9",
    ...     frame_data=frame_data
    ... )
    >>> len(templates) > 0
    True
    """

    def __init__(self, resource_dir: str | Path | None = None) -> None:
        """Initialize template generator.

        Parameters
        ----------
        resource_dir : str | Path | None
            Directory containing bleached lexicon CSV files.
        """
        if resource_dir is None:
            # Default to resources/ relative to this file
            resource_dir = Path(__file__).parent.parent / "resources"
        self.resource_dir = Path(resource_dir)

    def generate_from_frame(
        self,
        verb_lemma: str,
        verbnet_class: str,
        frame_data: dict,
    ) -> list[Template]:
        """Generate Template objects from a VerbNet frame.

        Parameters
        ----------
        verb_lemma : str
            The verb lemma (e.g., "think", "believe")
        verbnet_class : str
            VerbNet class ID (e.g., "29.9")
        frame_data : dict
            Frame data with keys:
            - primary: Primary frame description (e.g., "NP V that S")
            - secondary: Secondary description (optional)
            - syntax: List of (pos, value) tuples (optional)
            - examples: List of example sentences (optional)

        Returns
        -------
        list[Template]
            Generated templates for this frame (may return multiple
            templates for frames with multiple clausal variants).

        Examples
        --------
        >>> generator = TemplateGenerator()
        >>> frame = {"primary": "NP V NP", "examples": ["She broke the vase."]}
        >>> templates = generator.generate_from_frame("break", "45.1", frame)
        >>> templates[0].name
        'break_45.1_NP_V_NP'
        """
        frame_primary = frame_data.get("primary", "")

        # Try to map to clausal templates
        clausal_templates = map_verbnet_to_clausal_templates(frame_primary)

        if clausal_templates:
            # Generate templates for clausal frames
            return self._generate_clausal_templates(
                verb_lemma, verbnet_class, frame_data, clausal_templates
            )
        else:
            # Generate simple transitive/intransitive template
            return self._generate_simple_templates(
                verb_lemma, verbnet_class, frame_data
            )

    def _generate_clausal_templates(
        self,
        verb_lemma: str,
        verbnet_class: str,
        frame_data: dict,
        clausal_templates: list[ClausalTemplate],
    ) -> list[Template]:
        """Generate templates for clausal complement frames.

        Parameters
        ----------
        verb_lemma : str
            Verb lemma
        verbnet_class : str
            VerbNet class ID
        frame_data : dict
            Frame data
        clausal_templates : list[ClausalTemplate]
            Clausal template mappings

        Returns
        -------
        list[Template]
            Generated templates for each clausal variant
        """
        templates = []

        for clausal_template in clausal_templates:
            # Create slots based on clausal template
            slots = self._create_slots_from_clausal_template(clausal_template)

            # Generate constraints
            constraints = self._generate_clausal_constraints(clausal_template)

            # Create template name
            template_name = self._create_template_name(
                verb_lemma, verbnet_class, clausal_template.frame_type
            )

            # Create description
            description = self._create_description(
                verb_lemma, verbnet_class, frame_data, clausal_template
            )

            # Build template
            template = Template(
                name=template_name,
                template_string=clausal_template.template_string,
                slots=slots,
                constraints=constraints,
                description=description,
                language_code="eng",
                tags=self._generate_tags(frame_data, clausal_template),
                metadata={
                    "verb_lemma": verb_lemma,
                    "verbnet_class": verbnet_class,
                    "frame_primary": frame_data.get("primary", ""),
                    "frame_type": clausal_template.frame_type,
                    "complementizer": clausal_template.complementizer,
                    "mood": clausal_template.mood,
                    "examples": frame_data.get("examples", []),
                },
            )

            templates.append(template)

        return templates

    def _generate_simple_templates(
        self,
        verb_lemma: str,
        verbnet_class: str,
        frame_data: dict,
    ) -> list[Template]:
        """Generate templates for simple (non-clausal) frames.

        Parameters
        ----------
        verb_lemma : str
            Verb lemma
        verbnet_class : str
            VerbNet class ID
        frame_data : dict
            Frame data

        Returns
        -------
        list[Template]
            Generated template
        """
        frame_primary = frame_data.get("primary", "")

        # Infer slots from frame description
        slots, template_string = self._infer_slots_from_frame(frame_primary)

        # Generate constraints
        constraints = self._generate_simple_constraints(slots)

        # Create template name
        template_name = self._create_template_name(
            verb_lemma, verbnet_class, frame_primary
        )

        # Create description
        description = f"VerbNet frame: {verb_lemma} ({verbnet_class}) - {frame_primary}"

        # Build template
        template = Template(
            name=template_name,
            template_string=template_string,
            slots=slots,
            constraints=constraints,
            description=description,
            language_code="eng",
            tags=self._generate_tags(frame_data, None),
            metadata={
                "verb_lemma": verb_lemma,
                "verbnet_class": verbnet_class,
                "frame_primary": frame_primary,
                "examples": frame_data.get("examples", []),
            },
        )

        return [template]

    def _create_slots_from_clausal_template(
        self, clausal_template: ClausalTemplate
    ) -> dict[str, Slot]:
        """Create Slot objects from clausal template specification.

        Parameters
        ----------
        clausal_template : ClausalTemplate
            Clausal template specification

        Returns
        -------
        dict[str, Slot]
            Slots keyed by name
        """
        slots = {}

        for slot_name, slot_type in clausal_template.slots.items():
            # Create basic slot with POS constraint
            constraints = []

            if slot_type == "noun":
                constraints = [Constraint(expression="self.pos == 'NOUN'")]
            elif slot_type.startswith("verb"):
                constraints = [Constraint(expression="self.pos == 'VERB'")]
                # Add form constraints for specific verb forms
                if slot_type == "verb_past":
                    constraints.append(
                        Constraint(expression="self.features.tense == 'PST'")
                    )
                elif slot_type == "verb_base":
                    # Base form - could add constraint for infinitive
                    pass
                elif slot_type == "verb_present_participle":
                    constraints.append(
                        Constraint(expression="self.features.verb_form == 'V.PTCP'")
                    )
                    constraints.append(
                        Constraint(expression="self.features.tense == 'PRS'")
                    )
                elif slot_type == "verb_past_participle":
                    constraints.append(
                        Constraint(expression="self.features.verb_form == 'V.PTCP'")
                    )
                    constraints.append(
                        Constraint(expression="self.features.tense == 'PST'")
                    )

            # Create slot
            slots[slot_name] = Slot(
                name=slot_name,
                description=f"{slot_type} slot",
                constraints=constraints,
                required=True,
            )

        return slots

    def _generate_clausal_constraints(
        self, clausal_template: ClausalTemplate
    ) -> list[Constraint]:
        """Generate multi-slot constraints for clausal templates.

        Parameters
        ----------
        clausal_template : ClausalTemplate
            Clausal template specification

        Returns
        -------
        list[Constraint]
            Multi-slot constraints
        """
        constraints = []

        # Check for determiner-noun pairs
        slot_names = list(clausal_template.slots.keys())

        # Pattern: det + noun combinations
        for i, slot_name in enumerate(slot_names):
            if slot_name.endswith("_det") or slot_name == "det":
                # Find corresponding noun
                noun_slot = slot_name.replace("_det", "_noun")
                if noun_slot == "det":
                    noun_slot = "noun"

                if noun_slot in slot_names:
                    det_constraint = build_determiner_constraint(slot_name, noun_slot)
                    constraints.append(det_constraint)

        return constraints

    def _generate_simple_constraints(self, slots: dict[str, Slot]) -> list[Constraint]:
        """Generate constraints for simple templates.

        Parameters
        ----------
        slots : dict[str, Slot]
            Slot definitions

        Returns
        -------
        list[Constraint]
            Multi-slot constraints
        """
        constraints = []

        # Check for determiner-noun pairs
        slot_names = list(slots.keys())

        for i, slot_name in enumerate(slot_names):
            if "det" in slot_name:
                # Find corresponding noun
                for noun_slot in slot_names:
                    if "noun" in noun_slot and "det" not in noun_slot:
                        det_constraint = build_determiner_constraint(
                            slot_name, noun_slot
                        )
                        constraints.append(det_constraint)
                        break

        return constraints

    def _infer_slots_from_frame(
        self, frame_primary: str
    ) -> tuple[dict[str, Slot], str]:
        """Infer slots and template string from frame description.

        Parameters
        ----------
        frame_primary : str
            Primary frame description (e.g., "NP V NP", "NP V NP PP")

        Returns
        -------
        tuple[dict[str, Slot], str]
            Slots dict and template string
        """
        # Simple heuristic mapping
        frame_lower = frame_primary.lower()
        slots = {}
        template_parts = []

        # Track slot counters for multiple instances
        noun_counter = 0
        verb_counter = 0
        pp_counter = 0

        # Parse frame elements - first count total NPs to determine argument structure
        parts = frame_primary.split()
        total_nps = sum(1 for part in parts if part.lower() == "np")

        for part in parts:
            part_lower = part.lower()

            if part_lower == "np":
                # Noun phrase - create determiner + noun slots
                noun_counter += 1
                if noun_counter == 1:
                    # Always subject
                    det_slot_name = "det_subj"
                    noun_slot_name = "noun_subj"
                    det_description = "Subject determiner"
                    noun_description = "Subject noun"
                elif noun_counter == 2:
                    # Second NP: dobj in transitive, iobj in ditransitive
                    if total_nps >= 3:
                        det_slot_name = "det_iobj"
                        noun_slot_name = "noun_iobj"
                        det_description = "Indirect object determiner"
                        noun_description = "Indirect object noun"
                    else:
                        det_slot_name = "det_dobj"
                        noun_slot_name = "noun_dobj"
                        det_description = "Direct object determiner"
                        noun_description = "Direct object noun"
                elif noun_counter == 3:
                    # Third NP: always dobj in ditransitive
                    det_slot_name = "det_dobj"
                    noun_slot_name = "noun_dobj"
                    det_description = "Direct object determiner"
                    noun_description = "Direct object noun"
                else:
                    # Additional NPs (rare)
                    det_slot_name = f"det_noun{noun_counter}"
                    noun_slot_name = f"noun{noun_counter}"
                    det_description = f"Determiner for noun phrase {noun_counter}"
                    noun_description = f"Noun {noun_counter}"

                # Create determiner slot
                slots[det_slot_name] = Slot(
                    name=det_slot_name,
                    description=det_description,
                    constraints=[Constraint(expression="self.pos == 'DET'")],
                    required=True,
                )

                # Create noun slot
                slots[noun_slot_name] = Slot(
                    name=noun_slot_name,
                    description=noun_description,
                    constraints=[Constraint(expression="self.pos == 'NOUN'")],
                    required=True,
                )

                # Add both to template string
                template_parts.append(f"{{{det_slot_name}}} {{{noun_slot_name}}}")

            elif part_lower == "v":
                # Verb
                verb_counter += 1
                slot_name = "verb" if verb_counter == 1 else f"verb{verb_counter}"

                slots[slot_name] = Slot(
                    name=slot_name,
                    description="Main verb",
                    constraints=[Constraint(expression="self.pos == 'VERB'")],
                    required=True,
                )
                template_parts.append(f"{{{slot_name}}}")

            elif part_lower == "pp" or part_lower.startswith("pp."):
                # Prepositional phrase - preposition + determiner + noun
                pp_counter += 1
                prep_slot = f"prep{pp_counter}" if pp_counter > 1 else "prep"
                det_slot = f"det_pobj{pp_counter}" if pp_counter > 1 else "det_pobj"
                obj_slot = f"noun_pobj{pp_counter}" if pp_counter > 1 else "noun_pobj"

                # Preposition slot
                slots[prep_slot] = Slot(
                    name=prep_slot,
                    description="Preposition",
                    constraints=[Constraint(expression="self.pos == 'ADP'")],
                    required=True,
                )

                # PP object determiner slot
                slots[det_slot] = Slot(
                    name=det_slot,
                    description="Prepositional phrase object determiner",
                    constraints=[Constraint(expression="self.pos == 'DET'")],
                    required=True,
                )

                # PP object noun slot
                slots[obj_slot] = Slot(
                    name=obj_slot,
                    description="Prepositional phrase object noun",
                    constraints=[Constraint(expression="self.pos == 'NOUN'")],
                    required=True,
                )

                template_parts.append(f"{{{prep_slot}}} {{{det_slot}}} {{{obj_slot}}}")

        # Join template parts
        template_string = " ".join(template_parts)

        # Add period if not present
        if not template_string.endswith("."):
            template_string += "."

        return slots, template_string

    def _create_template_name(
        self, verb_lemma: str, verbnet_class: str, frame_info: str
    ) -> str:
        """Create a unique template name.

        Parameters
        ----------
        verb_lemma : str
            Verb lemma
        verbnet_class : str
            VerbNet class ID
        frame_info : str
            Frame type or primary description

        Returns
        -------
        str
            Template name
        """
        # Sanitize frame info for use in name
        safe_frame = frame_info.replace(" ", "_").replace(".", "").replace("-", "_")
        safe_verb = verb_lemma.replace(" ", "_").replace("-", "_")
        safe_class = verbnet_class.replace(".", "_").replace("-", "_")

        return f"{safe_verb}_{safe_class}_{safe_frame}"

    def _create_description(
        self,
        verb_lemma: str,
        verbnet_class: str,
        frame_data: dict,
        clausal_template: ClausalTemplate,
    ) -> str:
        """Create template description.

        Parameters
        ----------
        verb_lemma : str
            Verb lemma
        verbnet_class : str
            VerbNet class ID
        frame_data : dict
            Frame data
        clausal_template : ClausalTemplate
            Clausal template

        Returns
        -------
        str
            Description
        """
        parts = [
            f"VerbNet frame: {verb_lemma} ({verbnet_class})",
            f"Frame type: {clausal_template.frame_type}",
            f"Primary: {frame_data.get('primary', 'N/A')}",
        ]

        if clausal_template.complementizer:
            parts.append(f"Complementizer: {clausal_template.complementizer}")

        if clausal_template.mood:
            parts.append(f"Mood: {clausal_template.mood}")

        return " | ".join(parts)

    def _generate_tags(
        self, frame_data: dict, clausal_template: ClausalTemplate | None
    ) -> list[str]:
        """Generate tags for template categorization.

        Parameters
        ----------
        frame_data : dict
            Frame data
        clausal_template : ClausalTemplate | None
            Clausal template (if applicable)

        Returns
        -------
        list[str]
            Tags
        """
        tags = ["verbnet"]

        frame_primary = frame_data.get("primary", "").lower()

        # Add frame structure tags
        if "np" in frame_primary:
            tags.append("np")
        if "pp" in frame_primary:
            tags.append("pp")

        # Add clausal tags
        if clausal_template:
            tags.append("clausal")

            if clausal_template.complementizer:
                tags.append(f"comp_{clausal_template.complementizer}")

            if clausal_template.mood:
                tags.append(f"mood_{clausal_template.mood}")

            if "finite" in clausal_template.frame_type:
                tags.append("finite")
            elif "nonfinite" in clausal_template.frame_type:
                tags.append("nonfinite")

        return tags


def generate_templates_for_verb(
    verb_item: LexicalItem, include_frames: bool = True
) -> list[Template]:
    """Generate all templates for a VerbNet verb.

    Parameters
    ----------
    verb_item : LexicalItem
        VerbNet verb with frame data in attributes.
        Should have been fetched with include_frames=True.
    include_frames : bool
        Whether to expect frame data in attributes.

    Returns
    -------
    list[Template]
        All generated templates for this verb.

    Examples
    --------
    >>> from gallery.eng.argument_structure.utils.verbnet_parser import VerbNetExtractor
    >>> extractor = VerbNetExtractor()
    >>> verbs = extractor.get_verb_with_frames("think")
    >>> templates = generate_templates_for_verb(verbs[0])
    >>> len(templates) > 0
    True
    """
    generator = TemplateGenerator()

    if not include_frames or "frames" not in verb_item.attributes:
        return []

    templates = []
    frames = verb_item.attributes["frames"]
    verbnet_class = verb_item.attributes.get("verbnet_class", "unknown")

    for frame in frames:
        frame_templates = generator.generate_from_frame(
            verb_lemma=verb_item.lemma,
            verbnet_class=verbnet_class,
            frame_data=frame,
        )
        templates.extend(frame_templates)

    return templates


def generate_templates_for_all_verbs(
    verb_items: list[LexicalItem],
) -> list[Template]:
    """Generate templates for all VerbNet verbs.

    Parameters
    ----------
    verb_items : list[LexicalItem]
        VerbNet verbs with frame data.

    Returns
    -------
    list[Template]
        All generated templates.

    Examples
    --------
    >>> from gallery.eng.argument_structure.utils.verbnet_parser import VerbNetExtractor
    >>> extractor = VerbNetExtractor()
    >>> verbs = extractor.extract_all_verbs_with_frames()[:10]
    >>> templates = generate_templates_for_all_verbs(verbs)
    >>> len(templates) > 0
    True
    """
    all_templates = []

    for verb_item in verb_items:
        templates = generate_templates_for_verb(verb_item)
        all_templates.extend(templates)

    return all_templates
