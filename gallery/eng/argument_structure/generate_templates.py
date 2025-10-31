#!/usr/bin/env python3
"""
Generate Template objects from VerbNet frames.

Output: templates/verbnet_frames.jsonl
"""

import argparse
from pathlib import Path

from utils.template_generator import generate_templates_for_all_verbs
from utils.verbnet_parser import VerbNetExtractor

from bead.resources.adapters.cache import AdapterCache


def main(verb_limit: int | None = None) -> None:
    """Generate and save templates from VerbNet frames.

    Parameters
    ----------
    verb_limit : int | None
        Limit number of verb-class pairs to process (for testing).
    """
    # Set up paths
    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    # Initialize with caching
    cache = AdapterCache()
    extractor = VerbNetExtractor(cache=cache)

    # Get all verbs with detailed frame information
    print("=" * 80)
    print("EXTRACTING VERBNET VERBS WITH FRAMES")
    print("=" * 80)
    verbs_with_frames = extractor.extract_all_verbs_with_frames()
    print(f"Found {len(verbs_with_frames)} verb-class pairs")

    # Apply limit if specified
    if verb_limit is not None:
        print(f"[TEST MODE] Limiting to first {verb_limit} verb-class pairs")
        verbs_with_frames = verbs_with_frames[:verb_limit]

    # Generate templates for all verbs
    print("\n" + "=" * 80)
    print("GENERATING TEMPLATES")
    print("=" * 80)
    print(f"Processing {len(verbs_with_frames)} verb-class pairs...\n")

    templates = generate_templates_for_all_verbs(verbs_with_frames)
    print(f"\n✓ Generated {len(templates)} templates")

    # Save templates to JSONL
    print("\n" + "=" * 80)
    print("SAVING TEMPLATES")
    print("=" * 80)
    output_path = templates_dir / "verbnet_frames.jsonl"

    with open(output_path, "w") as f:
        for template in templates:
            # Convert Template object to JSON string (handles UUID serialization)
            template_json = template.model_dump_json()
            f.write(template_json + "\n")

    print(f"✓ Saved {len(templates)} templates to {output_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("TEMPLATE GENERATION COMPLETE")
    print("=" * 80)
    print(f"Verb-class pairs processed: {len(verbs_with_frames)}")
    print(f"Templates generated: {len(templates)}")
    if verbs_with_frames:
        avg_templates = len(templates) / len(verbs_with_frames)
        print(f"Average templates per verb-class: {avg_templates:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Template objects from VerbNet frames"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of verb-class pairs to process (for testing)",
    )
    args = parser.parse_args()

    main(verb_limit=args.limit)
