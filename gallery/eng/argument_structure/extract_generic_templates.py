#!/usr/bin/env python3
"""
Extract generic frame templates from verb-specific VerbNet templates.

This script analyzes the 21,453 verb-specific templates and extracts the
26 unique structural patterns (template_strings). Each generic template
includes metadata about which VerbNet frames map to it.

Output: templates/generic_frames.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console

from bead.resources.constraints import Constraint
from bead.resources.template import Slot, Template

console = Console()


def create_progressive_variant(base_template: Template, tense: str) -> Template:
    """Create progressive variant of a template.

    Parameters
    ----------
    base_template : Template
        Base template with finite verb.
    tense : str
        "present" or "past" for the auxiliary.

    Returns
    -------
    Template
        Progressive variant with {be} and {verb} slots.
    """
    # Only create progressive for templates with {verb} slot
    if "verb" not in base_template.slots:
        return None

    # Create new template string by replacing {verb} with {be} {verb}
    new_template_string = base_template.template_string.replace("{verb}", "{be} {verb}")

    # Create new slots dict
    new_slots = dict(base_template.slots)

    # Add be slot with tense constraint
    if tense == "present":
        be_constraint_expr = "self.lemma == 'be' and self.features.tense == 'PRS'"
        variant_name = "present_progressive"
        description_tense = "present progressive"
    else:  # past
        be_constraint_expr = "self.lemma == 'be' and self.features.tense == 'PST'"
        variant_name = "past_progressive"
        description_tense = "past progressive"

    new_slots["be"] = Slot(
        name="be",
        description=f"Auxiliary 'be' ({tense})",
        constraints=[Constraint(expression=be_constraint_expr)],
        required=True,
    )

    # Update verb slot to require present participle
    verb_slot = new_slots["verb"]
    new_slots["verb"] = Slot(
        name=verb_slot.name,
        description=f"{verb_slot.description} (present participle)",
        constraints=[
            Constraint(expression="self.features.get('verb_form') == 'V.PTCP'"),
            Constraint(expression="self.features.get('tense') == 'PRS'"),
        ],
        required=verb_slot.required,
        default_value=verb_slot.default_value,
    )

    # Create new template
    progressive_template = Template(
        name=f"{base_template.name}_{variant_name}",
        template_string=new_template_string,
        slots=new_slots,
        constraints=base_template.constraints,
        description=f"{base_template.description} ({description_tense})",
        language_code=base_template.language_code,
        tags=base_template.tags + [variant_name, "progressive"],
        metadata={
            **base_template.metadata,
            "base_template_id": str(base_template.id),
            "tense_aspect_variant": variant_name,
            "is_progressive": True,
        },
    )

    return progressive_template


def main(input_file: str = "templates/verbnet_frames.jsonl") -> None:
    """Extract generic templates from verb-specific templates.

    Parameters
    ----------
    input_file : str
        Path to verb-specific templates file.
    """
    base_dir = Path(__file__).parent
    input_path = base_dir / input_file
    output_path = base_dir / "templates" / "generic_frames.jsonl"

    # Group templates by template_string
    template_groups: dict[str, list[dict]] = defaultdict(list)

    console.rule("[1/4] Reading Verb-Specific Templates")

    with console.status("[bold]Loading templates...[/bold]"):
        with open(input_path) as f:
            for line in f:
                template = json.loads(line)
                template_string = template["template_string"]
                template_groups[template_string].append(template)

    total = sum(len(g) for g in template_groups.values())
    console.print(f"[green]✓[/green] Loaded {total:,} verb-specific templates")
    console.print(f"[green]✓[/green] Found {len(template_groups)} unique template structures\n")

    # Create generic templates
    console.rule("[2/4] Creating Generic Templates")

    generic_templates = []

    for template_string, specific_templates in sorted(
        template_groups.items(), key=lambda x: -len(x[1])
    ):
        # Use first template as prototype
        prototype = specific_templates[0]

        # Collect all VerbNet frames that use this structure
        frame_primaries = set()
        verbnet_classes = set()
        example_verbs = []

        for spec_template in specific_templates:
            frame_primaries.add(spec_template["metadata"]["frame_primary"])
            verbnet_classes.add(spec_template["metadata"]["verbnet_class"])
            verb = spec_template["metadata"]["verb_lemma"]
            if len(example_verbs) < 10:  # Keep 10 example verbs
                example_verbs.append(verb)

        # Create generic template name
        # Use first few words of template_string as name
        name_parts = template_string.replace("{", "").replace("}", "").split()[:5]
        template_name = "_".join(name_parts[:3])  # Use first 3 slots

        # Create generic template (without verb-specific constraints)
        generic_template = Template(
            name=template_name,
            template_string=template_string,
            slots=prototype["slots"],  # Reuse slots structure
            constraints=[],  # Generic templates have no multi-slot constraints yet
            description=f"Generic frame structure: {template_string}",
            language_code="eng",
            tags=["verbnet", "generic_frame"],
            metadata={
                "template_structure": template_string,
                "verb_count": len(specific_templates),
                "frame_primaries": sorted(frame_primaries),
                "verbnet_class_count": len(verbnet_classes),
                "example_verbs": example_verbs,
                "is_generic": True,
            },
        )

        generic_templates.append(generic_template)

        console.print(
            f"  {len(generic_templates):2d}. {template_name:30s} "
            f"[{len(specific_templates):5d} verbs, {len(frame_primaries):3d} frames]"
        )

    console.print(f"\n[green]✓[/green] Created {len(generic_templates)} base templates\n")

    # Generate progressive variants
    console.rule("[3/4] Generating Progressive Variants")

    progressive_templates = []
    with console.status("[bold]Creating progressive variants...[/bold]"):
        for base_template in generic_templates:
            # Create present progressive variant
            present_prog = create_progressive_variant(base_template, "present")
            if present_prog:
                progressive_templates.append(present_prog)

            # Create past progressive variant
            past_prog = create_progressive_variant(base_template, "past")
            if past_prog:
                progressive_templates.append(past_prog)

    console.print(f"[green]✓[/green] Generated {len(progressive_templates)} progressive variants\n")

    # Combine base and progressive templates
    all_templates = generic_templates + progressive_templates

    # Save generic templates
    console.rule("[4/4] Saving Templates")

    with open(output_path, "w") as f:
        for template in all_templates:
            template_json = template.model_dump_json()
            f.write(template_json + "\n")

    console.print(f"[green]✓[/green] Saved {len(all_templates)} templates to {output_path}\n")

    # Summary statistics
    console.rule("[bold]Summary[/bold]")
    from rich.table import Table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Base generic frames:", f"[cyan]{len(generic_templates)}[/cyan]")
    table.add_row("Progressive variants:", f"[cyan]{len(progressive_templates)}[/cyan]")
    table.add_row("Total templates:", f"[cyan]{len(all_templates)}[/cyan]")
    table.add_row("", "")
    table.add_row("Cross-product size:", f"[cyan]~2,880 verbs × {len(all_templates)} templates[/cyan]")
    table.add_row("Total combinations:", f"[cyan]≈ {2880 * len(all_templates):,}[/cyan]")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract generic templates from verb-specific templates"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="templates/verbnet_frames.jsonl",
        help="Input file with verb-specific templates",
    )
    args = parser.parse_args()

    main(input_file=args.input)
