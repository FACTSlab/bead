#!/usr/bin/env python3
"""Generate 2AFC (two-alternative forced choice) pairs from cross-product items.

This script:
1. Loads cross-product items (verb × template combinations)
2. Fills templates using MixedFillingStrategy
3. Scores filled items with language model (uses bead/items/scoring.py)
4. Creates forced-choice items (uses bead/items/forced_choice.py)
5. Assigns quantiles (uses bead/lists/stratification.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.table import Table

from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer
from bead.lists.stratification import assign_quantiles_by_uuid
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template

console = Console()


def load_cross_product_items(path: str, limit: int | None = None) -> list[Item]:
    """Load cross-product items from JSONL."""
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            items.append(Item(**data))
    return items


def load_templates(path: str) -> dict[str, Template]:
    """Load templates from JSONL and return dict keyed by ID."""
    templates = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            template = Template(**data)
            templates[str(template.id)] = template
    return templates


def load_lexicons(lexicon_dir: Path) -> dict[str, Lexicon]:
    """Load all lexicons from directory."""
    lexicons = {}

    # Map slot names to lexicon files
    slot_to_lexicon = {
        "subj": ("bleached_nouns.jsonl", "bleached_nouns"),
        "obj": ("bleached_nouns.jsonl", "bleached_nouns"),
        "noun": ("bleached_nouns.jsonl", "bleached_nouns"),
        "verb": ("verbnet_verbs.jsonl", "verbnet_verbs"),
        "adj": ("bleached_adjectives.jsonl", "bleached_adjectives"),
        "adjective": ("bleached_adjectives.jsonl", "bleached_adjectives"),
        "det": ("determiners.jsonl", "determiners"),
        "prep": ("prepositions.jsonl", "prepositions"),
        "be": ("be_forms.jsonl", "be_forms"),
    }

    # Load unique lexicon files
    loaded_files = {}
    for slot_name, (filename, lex_name) in slot_to_lexicon.items():
        if filename not in loaded_files:
            path = lexicon_dir / filename
            if path.exists():
                loaded_files[filename] = Lexicon.from_jsonl(str(path), name=lex_name)

        # Map slot name to loaded lexicon
        if filename in loaded_files:
            lexicons[slot_name] = loaded_files[filename]

    return lexicons


def fill_templates_with_mixed_strategy(
    items: list[Item],
    templates: dict[str, Template],
    lexicons: dict[str, Lexicon],
) -> dict[str, str]:
    """Fill templates and return mapping of item_id -> filled_text.

    Handles both base templates (simple tense) and progressive templates (be + participle).
    Selects inflected verb forms from lexicon based on template type.
    """
    filled_texts = {}

    for item in items:
        template_id = str(item.item_template_id)
        if template_id not in templates:
            continue

        template = templates[template_id]

        # Get verb from item metadata
        verb_lemma = item.item_metadata.get("verb_lemma")
        if not verb_lemma:
            continue

        # Detect template type
        is_progressive = "be" in template.slots

        # Build slot values
        slot_values = {}

        # Handle verb slot (select inflected form)
        if "verb" in lexicons:
            # Get all forms of this verb
            verb_forms = [
                item for item in lexicons["verb"].items.values()
                if item.lemma == verb_lemma
            ]

            if is_progressive:
                # Select present participle (V.PTCP;PRS)
                participles = [
                    vf for vf in verb_forms
                    if vf.features.get("verb_form") == "V.PTCP"
                    and vf.features.get("tense") == "PRS"
                ]
                if participles:
                    slot_values["verb"] = participles[0].form or verb_lemma
                else:
                    # Fallback: use lemma + "ing"
                    slot_values["verb"] = f"{verb_lemma}ing"
            else:
                # Select 3sg present form for base templates
                present_3sg = [
                    vf for vf in verb_forms
                    if vf.features.get("tense") == "PRS"
                    and vf.features.get("person") == "3"
                    and vf.features.get("number") == "SG"
                ]
                if present_3sg:
                    slot_values["verb"] = present_3sg[0].form or verb_lemma
                else:
                    # Fallback: use lemma
                    slot_values["verb"] = verb_lemma
        else:
            slot_values["verb"] = verb_lemma

        # Handle be slot (for progressive templates)
        if is_progressive and "be" in lexicons:
            # Select present tense 3sg form ("is")
            be_forms = [
                item for item in lexicons["be"].items.values()
                if item.lemma == "be"
                and item.features.get("tense") == "PRS"
                and item.features.get("person") == "3"
                and item.features.get("number") == "SG"
            ]
            if be_forms:
                slot_values["be"] = be_forms[0].form
            else:
                slot_values["be"] = "is"  # Fallback

        # Fill other slots with first available lexicon entry
        for slot_name in template.slots.keys():
            if slot_name in ["verb", "be"]:
                continue

            # Determine lexicon type from slot name
            if slot_name.startswith("det"):
                # Determiner slot
                if "det" in lexicons and len(lexicons["det"]) > 0:
                    first_item = next(iter(lexicons["det"].items.values()))
                    slot_values[slot_name] = first_item.lemma
            elif "noun" in slot_name or slot_name in ["subj", "obj", "comp_subj", "comp_obj"]:
                # Noun slot
                if "noun" in lexicons and len(lexicons["noun"]) > 0:
                    first_item = next(iter(lexicons["noun"].items.values()))
                    slot_values[slot_name] = first_item.lemma
            elif slot_name.startswith("prep"):
                # Preposition slot
                if "prep" in lexicons and len(lexicons["prep"]) > 0:
                    first_item = next(iter(lexicons["prep"].items.values()))
                    slot_values[slot_name] = first_item.lemma
            elif "verb" in slot_name and slot_name != "verb":
                # Other verb slots (comp_verb, etc.)
                # For now, use same logic as main verb slot
                if is_progressive:
                    # Use present participle
                    verb_forms = [
                        vf for vf in lexicons.get("verb", Lexicon(name="empty")).items.values()
                        if vf.lemma == verb_lemma
                        and vf.features.get("verb_form") == "V.PTCP"
                        and vf.features.get("tense") == "PRS"
                    ]
                    slot_values[slot_name] = verb_forms[0].form if verb_forms else f"{verb_lemma}ing"
                else:
                    # Use 3sg present
                    verb_forms = [
                        vf for vf in lexicons.get("verb", Lexicon(name="empty")).items.values()
                        if vf.lemma == verb_lemma
                        and vf.features.get("tense") == "PRS"
                        and vf.features.get("person") == "3"
                        and vf.features.get("number") == "SG"
                    ]
                    slot_values[slot_name] = verb_forms[0].form if verb_forms else verb_lemma
            elif "adj" in slot_name or slot_name == "adjective":
                # Adjective slot
                if "adj" in lexicons and len(lexicons["adj"]) > 0:
                    first_item = next(iter(lexicons["adj"].items.values()))
                    slot_values[slot_name] = first_item.lemma

        # Use template.fill_with_values() to create FilledTemplate
        filled_template = template.fill_with_values(slot_values, strategy_name="simple_fill")

        # Check if all required slots are filled
        if not filled_template.is_complete:
            # Skip items with unfilled required slots
            continue

        filled_texts[str(item.id)] = filled_template.rendered_text

    return filled_texts


def score_filled_items_with_lm(
    items: list[Item],
    cache_dir: Path,
    model_name: str = "gpt2",
) -> dict[str, float]:
    """Score filled items with language model using bead/items/scoring.py."""
    # Use bead's LanguageModelScorer
    scorer = LanguageModelScorer(
        model_name=model_name,
        cache_dir=cache_dir,
        device="cpu",
        text_key="text",
    )

    # Create temporary items with filled text in rendered_elements
    temp_items = []
    item_id_map = {}
    for item in items:
        temp_item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": item.rendered_elements.get("text", "")},
        )
        temp_items.append(temp_item)
        item_id_map[temp_item.id] = str(item.id)

    # Score batch (progress bar shown by scorer)
    scores_list = scorer.score_batch(temp_items)

    # Map back to original item IDs
    scores = {}
    for temp_item, score in zip(temp_items, scores_list, strict=True):
        original_id = item_id_map[temp_item.id]
        scores[original_id] = score

    return scores


def create_forced_choice_pairs(
    items: list[Item],
    lm_scores: dict[str, float],
) -> list[Item]:
    """Create 2AFC items using bead/items/forced_choice.py.

    Creates two types of forced-choice items:
    1. Same-verb pairs (same verb, different frames)
    2. Different-verb pairs (different verbs, same frame)
    """
    # Add scores to item metadata
    for item in items:
        item.item_metadata["lm_score"] = lm_scores.get(str(item.id), float("-inf"))

    # Helper to extract text from items
    def extract_text(item: Item) -> str:
        return item.rendered_elements.get("text", "")

    # 1. Create same-verb pairs (group by verb_lemma)
    with console.status("[bold]Creating same-verb pairs...[/bold]"):
        same_verb_items = create_forced_choice_items_from_groups(
            items=items,
            group_by=lambda item: item.item_metadata.get("verb_lemma", "unknown"),
            n_alternatives=2,
            extract_text=extract_text,
            include_group_metadata=True,
        )

    # Add pair_type and additional metadata
    for fc_item in same_verb_items:
        item1_id = fc_item.item_metadata.get("source_item_0_id")
        item2_id = fc_item.item_metadata.get("source_item_1_id")

        source_items = [i for i in items if str(i.id) in [item1_id, item2_id]]
        if len(source_items) == 2:
            fc_item.item_metadata.update({
                "pair_type": "same_verb",
                "verb": source_items[0].item_metadata.get("verb_lemma"),
                "template1": source_items[0].item_metadata.get("template_structure"),
                "template2": source_items[1].item_metadata.get("template_structure"),
                "lm_score_a": lm_scores.get(str(source_items[0].id), float("-inf")),
                "lm_score_b": lm_scores.get(str(source_items[1].id), float("-inf")),
                "lm_score_diff": abs(
                    lm_scores.get(str(source_items[0].id), 0) -
                    lm_scores.get(str(source_items[1].id), 0)
                ),
            })

    console.print(f"[green]✓[/green] Created {len(same_verb_items):,} same-verb pairs")

    # 2. Create different-verb pairs (group by template_id)
    with console.status("[bold]Creating different-verb pairs...[/bold]"):
        different_verb_items = create_forced_choice_items_from_groups(
            items=items,
            group_by=lambda item: str(item.item_template_id),
            n_alternatives=2,
            extract_text=extract_text,
            include_group_metadata=True,
        )

    # Add pair_type and additional metadata
    for fc_item in different_verb_items:
        item1_id = fc_item.item_metadata.get("source_item_0_id")
        item2_id = fc_item.item_metadata.get("source_item_1_id")

        source_items = [i for i in items if str(i.id) in [item1_id, item2_id]]
        if len(source_items) == 2:
            fc_item.item_metadata.update({
                "pair_type": "different_verb",
                "template_id": str(source_items[0].item_template_id),
                "template_structure": source_items[0].item_metadata.get("template_structure"),
                "verb1": source_items[0].item_metadata.get("verb_lemma"),
                "verb2": source_items[1].item_metadata.get("verb_lemma"),
                "lm_score_a": lm_scores.get(str(source_items[0].id), float("-inf")),
                "lm_score_b": lm_scores.get(str(source_items[1].id), float("-inf")),
                "lm_score_diff": abs(
                    lm_scores.get(str(source_items[0].id), 0) -
                    lm_scores.get(str(source_items[1].id), 0)
                ),
            })

    console.print(f"[green]✓[/green] Created {len(different_verb_items):,} different-verb pairs")

    return same_verb_items + different_verb_items


def assign_quantiles_to_pairs(
    pair_items: list[Item],
    n_quantiles: int = 10,
) -> list[Item]:
    """Assign quantile bins using bead/lists/stratification.py.

    Stratifies by pair_type so same-verb and different-verb pairs
    get separate quantile distributions.
    """
    with console.status("[bold]Assigning quantiles (stratified by pair_type)...[/bold]"):
        # Build metadata dict for quantile assignment
        item_metadata = {
            item.id: item.item_metadata
            for item in pair_items
        }

        # Get item IDs
        item_ids = [item.id for item in pair_items]

        # Assign quantiles stratified by pair_type
        quantile_assignments = assign_quantiles_by_uuid(
            item_ids=item_ids,
            item_metadata=item_metadata,
            property_key="lm_score_diff",
            n_quantiles=n_quantiles,
            stratify_by_key="pair_type",
        )

        # Add quantile to each item's metadata
        for item in pair_items:
            item.item_metadata["quantile"] = quantile_assignments[item.id]

    console.print(f"[green]✓[/green] Assigned quantiles to {len(pair_items):,} pairs")
    return pair_items


def main(item_limit: int | None = None) -> None:
    """Generate 2AFC pairs from cross-product.

    Parameters
    ----------
    item_limit : int | None
        Limit number of cross-product items to process.
        If None, process all items.
    """
    # Paths
    base_dir = Path(__file__).parent
    items_path = base_dir / "items" / "cross_product_items.jsonl"
    templates_path = base_dir / "templates" / "generic_frames.jsonl"
    lexicons_dir = base_dir / "lexicons"
    output_path = base_dir / "items" / "2afc_pairs.jsonl"

    console.rule("[bold]2AFC Pair Generation[/bold]")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Cross-product items: [cyan]{items_path}[/cyan]")
    console.print(f"Output: [cyan]{output_path}[/cyan]\n")

    if item_limit:
        console.print(f"[yellow]⚠[/yellow]  Test mode: Limiting to {item_limit:,} items\n")

    # Load cross-product items
    console.rule("[1/7] Loading Cross-Product Items")
    with console.status("[bold]Loading items...[/bold]"):
        items = load_cross_product_items(str(items_path), limit=item_limit)
    console.print(f"[green]✓[/green] Loaded {len(items):,} cross-product items\n")

    # Load templates
    console.rule("[2/7] Loading Templates")
    with console.status("[bold]Loading templates...[/bold]"):
        templates = load_templates(str(templates_path))
    console.print(f"[green]✓[/green] Loaded {len(templates)} templates\n")

    # Load lexicons
    console.rule("[3/7] Loading Lexicons")
    with console.status("[bold]Loading lexicons...[/bold]"):
        lexicons = load_lexicons(lexicons_dir)
    console.print(f"[green]✓[/green] Loaded {len(lexicons)} lexicons")
    for name, lexicon in lexicons.items():
        console.print(f"  • [cyan]{name}[/cyan]: {len(lexicon)} entries")
    console.print()

    # Fill templates
    console.rule("[4/7] Filling Templates")
    with console.status("[bold]Filling templates...[/bold]"):
        filled_texts = fill_templates_with_mixed_strategy(items, templates, lexicons)

    if not filled_texts:
        console.print("[red]✗[/red] No items were successfully filled. Exiting.")
        return

    console.print(f"[green]✓[/green] Filled {len(filled_texts):,} items")

    # Create Items with filled text in rendered_elements
    filled_items = []
    for item in items:
        if str(item.id) in filled_texts:
            item.rendered_elements["text"] = filled_texts[str(item.id)]
            filled_items.append(item)

    # Show examples
    console.print("\n[dim]Example filled texts:[/dim]")
    for i, item in enumerate(filled_items[:3]):
        console.print(f"  [dim]{i + 1}.[/dim] {item.rendered_elements['text']}")
    console.print()

    # Score with LM
    console.rule("[5/7] Scoring with Language Model")
    cache_dir = base_dir / ".cache"
    lm_scores = score_filled_items_with_lm(
        filled_items, cache_dir=cache_dir, model_name="gpt2"
    )
    console.print(f"[green]✓[/green] Scored {len(lm_scores):,} items\n")

    # Create forced-choice pairs
    console.rule("[6/7] Creating Forced-Choice Pairs")
    pair_items = create_forced_choice_pairs(filled_items, lm_scores)

    if not pair_items:
        console.print("[red]✗[/red] No pairs were created. Exiting.")
        return

    console.print()

    # Assign quantiles
    console.rule("[7/7] Assigning Quantiles")
    pair_items = assign_quantiles_to_pairs(pair_items, n_quantiles=10)
    console.print()

    # Save
    console.rule("Saving Results")
    with console.status(f"[bold]Writing to {output_path}...[/bold]"):
        with open(output_path, "w") as f:
            for item in pair_items:
                f.write(item.model_dump_json() + "\n")

    console.print(f"[green]✓[/green] Saved {len(pair_items):,} 2AFC pairs\n")

    # Summary
    console.rule("[bold]Summary[/bold]")
    same_verb_count = sum(
        1 for item in pair_items if item.item_metadata.get("pair_type") == "same_verb"
    )
    different_verb_count = sum(
        1 for item in pair_items if item.item_metadata.get("pair_type") == "different_verb"
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Same-verb pairs:", f"[cyan]{same_verb_count:,}[/cyan]")
    table.add_row("Different-verb pairs:", f"[cyan]{different_verb_count:,}[/cyan]")
    table.add_row("Total pairs:", f"[cyan]{len(pair_items):,}[/cyan]")
    table.add_row("Output file:", f"[cyan]{output_path}[/cyan]")
    console.print(table)

    console.print("\n[dim]Next: Run generate_lists.py to partition pairs into experiment lists[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 2AFC pairs from cross-product items"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cross-product items to process (default: all)",
    )
    args = parser.parse_args()

    main(item_limit=args.limit)
