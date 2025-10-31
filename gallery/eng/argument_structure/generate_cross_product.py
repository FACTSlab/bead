#!/usr/bin/env python3
"""
Generate cross-product of all verbs × all generic templates.

This script creates the foundational item set for the argument structure
experiment by testing every VerbNet verb in every generic frame structure.

Output: items/cross_product_items.jsonl
"""

import argparse
import json
from pathlib import Path

from bead.items.models import Item
from bead.resources.lexicon import Lexicon


def main(
    templates_file: str = "templates/generic_frames.jsonl",
    verbs_file: str = "lexicons/verbnet_verbs.jsonl",
    output_limit: int | None = None,
) -> None:
    """Generate cross-product items.

    Parameters
    ----------
    templates_file : str
        Path to generic templates file.
    verbs_file : str
        Path to verb lexicon file.
    output_limit : int | None
        Limit output to first N items (for testing).
    """
    base_dir = Path(__file__).parent
    templates_path = base_dir / templates_file
    verbs_path = base_dir / verbs_file
    output_dir = base_dir / "items"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "cross_product_items.jsonl"

    # Load generic templates
    print("=" * 80)
    print("LOADING GENERIC TEMPLATES")
    print("=" * 80)

    templates = []
    with open(templates_path) as f:
        for line in f:
            template = json.loads(line)
            templates.append(template)

    print(f"Loaded {len(templates)} generic templates")

    # Load verb lexicon
    print("\n" + "=" * 80)
    print("LOADING VERB LEXICON")
    print("=" * 80)

    verb_lexicon = Lexicon.from_jsonl(str(verbs_path), "verbnet_verbs")
    print(f"Loaded {len(verb_lexicon.items)} verb forms")

    # Get unique verb lemmas (we only need base forms for cross-product)
    verb_lemmas = set()
    for item in verb_lexicon.items.values():
        verb_lemmas.add(item.lemma)

    print(f"Unique verb lemmas: {len(verb_lemmas)}")

    # Generate cross-product
    print("\n" + "=" * 80)
    print("GENERATING CROSS-PRODUCT")
    print("=" * 80)

    if output_limit:
        print(f"[TEST MODE] Limiting output to {output_limit} items")

    items_generated = 0
    total_combinations = len(verb_lemmas) * len(templates)

    with open(output_path, "w") as f:
        for _template_idx, template in enumerate(templates, 1):
            template_id = template["id"]
            template_name = template["name"]
            template_string = template["template_string"]

            for _verb_idx, verb_lemma in enumerate(sorted(verb_lemmas), 1):
                # Create Item for this verb×template combination
                item = Item(
                    item_template_id=template_id,
                    rendered_elements={
                        "template_name": template_name,
                        "template_string": template_string,
                        "verb_lemma": verb_lemma,
                    },
                    item_metadata={
                        "verb_lemma": verb_lemma,
                        "template_id": str(template_id),
                        "template_name": template_name,
                        "template_structure": template_string,
                        "combination_type": "verb_frame_cross_product",
                    },
                )

                # Write to file
                f.write(item.model_dump_json() + "\n")
                items_generated += 1

                # Progress reporting
                if items_generated % 10000 == 0:
                    pct = (items_generated / total_combinations) * 100
                    print(
                        f"  Progress: {items_generated:,}/"
                        f"{total_combinations:,} ({pct:.1f}%)"
                    )

                # Check limit
                if output_limit and items_generated >= output_limit:
                    break

            if output_limit and items_generated >= output_limit:
                break

    print(f"\n✓ Generated {items_generated:,} cross-product items")
    print(f"✓ Saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Verb lemmas: {len(verb_lemmas)}")
    print(f"Generic templates: {len(templates)}")
    print(f"Cross-product items: {items_generated:,}")
    print()
    print("Next steps:")
    print("  1. Fill templates with lexicons (using MixedFillingStrategy)")
    print("  2. Score filled items with language model")
    print("  3. Create 2AFC pairs based on LM scores")
    print("  4. Partition pairs into balanced lists")
    print("  5. Deploy and collect human judgments")
    print("  6. Train model and iterate with active learning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cross-product of verbs × templates"
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="templates/generic_frames.jsonl",
        help="Path to generic templates file",
    )
    parser.add_argument(
        "--verbs",
        type=str,
        default="lexicons/verbnet_verbs.jsonl",
        help="Path to verb lexicon file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit output to first N items (for testing)",
    )
    args = parser.parse_args()

    main(
        templates_file=args.templates,
        verbs_file=args.verbs,
        output_limit=args.limit,
    )
