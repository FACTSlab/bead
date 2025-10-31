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

from bead.resources.structures import Template


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

    print("=" * 80)
    print("READING VERB-SPECIFIC TEMPLATES")
    print("=" * 80)

    with open(input_path) as f:
        for line in f:
            template = json.loads(line)
            template_string = template["template_string"]
            template_groups[template_string].append(template)

    total = sum(len(g) for g in template_groups.values())
    print(f"Total verb-specific templates: {total}")
    print(f"Unique template structures: {len(template_groups)}")

    # Create generic templates
    print("\n" + "=" * 80)
    print("CREATING GENERIC TEMPLATES")
    print("=" * 80)

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

        print(
            f"  {len(generic_templates):2d}. {template_name:30s} "
            f"[{len(specific_templates):5d} verbs, {len(frame_primaries):3d} frames]"
        )

    # Save generic templates
    print("\n" + "=" * 80)
    print("SAVING GENERIC TEMPLATES")
    print("=" * 80)

    with open(output_path, "w") as f:
        for template in generic_templates:
            template_json = template.model_dump_json()
            f.write(template_json + "\n")

    print(f"✓ Saved {len(generic_templates)} generic templates to {output_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Input: {input_path}")
    print("  - Verb-specific templates: 21,453")
    print("  - Unique verbs: ~4,789")
    print()
    print(f"Output: {output_path}")
    print(f"  - Generic frame structures: {len(generic_templates)}")
    print()
    print("Cross-product for experiment:")
    print(f"  - ~4,789 verbs × {len(generic_templates)} frames")
    print(f"  - ≈ {4789 * len(generic_templates):,} total combinations")
    print()
    print("This enables testing ALL verbs in ALL frames!")


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
