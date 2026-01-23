#!/usr/bin/env python3
"""Generate cross-product of all verbs × all generic templates.

This script creates the foundational item set for the argument structure
experiment by testing every VerbNet verb in every generic frame structure.

Output: items/cross_product_items.jsonl
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import track
from rich.table import Table

from bead.items.item import Item
from bead.resources.lexicon import Lexicon

console = Console()


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

    console.rule("[bold]Cross-Product Generation[/bold]")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Templates: [cyan]{templates_path}[/cyan]")
    console.print(f"Verbs: [cyan]{verbs_path}[/cyan]")
    console.print(f"Output: [cyan]{output_path}[/cyan]\n")

    # Load generic templates
    console.rule("[1/3] Loading Generic Templates")
    with console.status("[bold]Loading templates...[/bold]"):
        templates = []
        with open(templates_path) as f:
            for line in f:
                template = json.loads(line)
                templates.append(template)

    console.print(f"[green]✓[/green] Loaded {len(templates)} generic templates\n")

    # Load verb lexicon
    console.rule("[2/3] Loading Verb Lexicon")
    with console.status("[bold]Loading verb lexicon...[/bold]"):
        verb_lexicon = Lexicon.from_jsonl(str(verbs_path), "verbnet_verbs")

    console.print(f"[green]✓[/green] Loaded {len(verb_lexicon.items)} verb forms")

    # Get unique verb lemmas (we only need base forms for cross-product)
    verb_lemmas = sorted({item.lemma for item in verb_lexicon.items.values()})
    console.print(f"[green]✓[/green] Found {len(verb_lemmas)} unique verb lemmas\n")

    # Generate cross-product
    console.rule("[3/3] Generating Cross-Product")
    total_combinations = len(verb_lemmas) * len(templates)

    if output_limit:
        console.print(
            f"[yellow]⚠[/yellow]  Test mode: Limiting output to {output_limit:,} items"
        )
        total_combinations = min(output_limit, total_combinations)

    items_generated = 0

    with open(output_path, "w") as f:
        for template in track(templates, description="Processing templates"):
            template_id = template["id"]
            template_name = template["name"]
            template_string = template["template_string"]

            for verb_lemma in verb_lemmas:
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

                # Check limit
                if output_limit and items_generated >= output_limit:
                    break

            if output_limit and items_generated >= output_limit:
                break

    console.print(
        f"[green]✓[/green] Generated {items_generated:,} cross-product items\n"
    )

    # Summary
    console.rule("[bold]Summary[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Verb lemmas:", f"[cyan]{len(verb_lemmas)}[/cyan]")
    table.add_row("Generic templates:", f"[cyan]{len(templates)}[/cyan]")
    table.add_row("Cross-product items:", f"[cyan]{items_generated:,}[/cyan]")
    table.add_row("Output file:", f"[cyan]{output_path}[/cyan]")
    console.print(table)

    console.print(
        "\n[dim]Next: Run create_2afc_pairs.py to generate forced-choice pairs[/dim]"
    )


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
