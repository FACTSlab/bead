#!/usr/bin/env python3
"""Generate 2AFC (two-alternative forced choice) pairs from cross-product items.

This script:
1. Loads cross-product items (verb × template combinations)
2. Fills templates using MixedFillingStrategy
3. Scores filled items with language model
4. Creates minimal pairs (same-verb and different-verb)
5. Computes likelihood differences and assigns quantiles
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np

from bead.items.adapters.huggingface import HuggingFaceLanguageModel
from bead.items.cache import ModelOutputCache
from bead.items.models import Item
from bead.resources.lexicon import Lexicon
from bead.resources.structures import Template


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

    For now, uses simple exhaustive strategy with first available entry.
    """
    filled_texts = {}

    for item in items:
        template_id = str(item.item_template_id)
        if template_id not in templates:
            print(f"Warning: Template {template_id} not found, skipping item {item.id}")
            continue

        template = templates[template_id]

        # Get verb from item metadata
        verb_lemma = item.item_metadata.get("verb_lemma")
        if not verb_lemma:
            print(f"Warning: No verb_lemma in item {item.id}, skipping")
            continue

        # Build slot values
        slot_values = {"verb": verb_lemma}

        # Fill other slots with first available lexicon entry
        for slot_name in template.slots.keys():
            if slot_name == "verb":
                continue  # Already set

            # Get lexicon for this slot
            if slot_name in lexicons:
                lexicon = lexicons[slot_name]
                if len(lexicon) > 0:
                    # Use first entry (items is a dict)
                    first_item = next(iter(lexicon.items.values()))
                    # Use form if available, otherwise use lemma
                    slot_values[slot_name] = first_item.form or first_item.lemma
            else:
                # Try to infer lexicon from slot name
                if slot_name in ["subj", "obj", "noun"]:
                    if "noun" in lexicons and len(lexicons["noun"]) > 0:
                        first_item = next(iter(lexicons["noun"].items.values()))
                        slot_values[slot_name] = first_item.form or first_item.lemma
                elif slot_name in ["adj", "adjective"]:
                    if "adjective" in lexicons and len(lexicons["adjective"]) > 0:
                        first_item = next(iter(lexicons["adjective"].items.values()))
                        slot_values[slot_name] = first_item.form or first_item.lemma

        # Fill template
        try:
            filled_text = template.template_string
            for slot_name, value in slot_values.items():
                filled_text = filled_text.replace(f"{{{slot_name}}}", value)

            # Check if any slots remain unfilled
            if "{" in filled_text and "}" in filled_text:
                print(
                    f"Warning: Unfilled slots remain in item {item.id}: {filled_text}"
                )
                # Skip items with unfilled slots
                continue

            filled_texts[str(item.id)] = filled_text
        except Exception as e:
            print(f"Warning: Error filling item {item.id}: {e}")
            continue

    return filled_texts


def score_with_language_model(
    filled_items: dict[str, str],
    cache_dir: Path,
    model_name: str = "gpt2",
) -> dict[str, float]:
    """Score filled items with language model (get log likelihood)."""
    print(f"  Loading model {model_name}...")
    cache = ModelOutputCache(cache_dir=cache_dir)
    adapter = HuggingFaceLanguageModel(model_name=model_name, cache=cache, device="cpu")
    scores = {}

    print(f"  Scoring {len(filled_items)} items...")
    for i, (item_id, text) in enumerate(filled_items.items()):
        if i % 10 == 0:
            print(f"    Progress: {i}/{len(filled_items)}")

        try:
            # Get log likelihood from model
            log_prob = adapter.compute_log_probability(text)
            scores[item_id] = log_prob
        except Exception as e:
            print(f"Warning: Error scoring item {item_id}: {e}")
            scores[item_id] = float("-inf")

    print("  Scoring complete.")
    return scores


def create_pairs(
    items: list[Item],
    filled_texts: dict[str, str],
    lm_scores: dict[str, float],
) -> list[dict]:
    """Create same-verb and different-verb minimal pairs."""
    pairs = []

    # Filter items to only those that were successfully filled
    valid_items = [item for item in items if str(item.id) in filled_texts]

    # Group by verb for same-verb pairs
    by_verb = defaultdict(list)
    for item in valid_items:
        verb = item.item_metadata["verb_lemma"]
        by_verb[verb].append(item)

    # Create same-verb pairs (same verb, different frames)
    for verb, verb_items in by_verb.items():
        for i, item1 in enumerate(verb_items):
            for item2 in verb_items[i + 1 :]:
                score1 = lm_scores[str(item1.id)]
                score2 = lm_scores[str(item2.id)]

                pairs.append(
                    {
                        "item1_id": str(item1.id),
                        "item2_id": str(item2.id),
                        "text1": filled_texts[str(item1.id)],
                        "text2": filled_texts[str(item2.id)],
                        "pair_type": "same_verb",
                        "verb": verb,
                        "template1": item1.item_metadata["template_structure"],
                        "template2": item2.item_metadata["template_structure"],
                        "lm_score1": score1,
                        "lm_score2": score2,
                        "lm_score_diff": abs(score1 - score2),
                    }
                )

    # Group by template for different-verb pairs
    by_template = defaultdict(list)
    for item in valid_items:
        template_id = str(item.item_template_id)
        by_template[template_id].append(item)

    # Create different-verb pairs (different verbs, same frame)
    for template_id, template_items in by_template.items():
        for i, item1 in enumerate(template_items):
            for item2 in template_items[i + 1 :]:
                score1 = lm_scores[str(item1.id)]
                score2 = lm_scores[str(item2.id)]

                pairs.append(
                    {
                        "item1_id": str(item1.id),
                        "item2_id": str(item2.id),
                        "text1": filled_texts[str(item1.id)],
                        "text2": filled_texts[str(item2.id)],
                        "pair_type": "different_verb",
                        "template_id": template_id,
                        "template_structure": item1.item_metadata["template_structure"],
                        "verb1": item1.item_metadata["verb_lemma"],
                        "verb2": item2.item_metadata["verb_lemma"],
                        "lm_score1": score1,
                        "lm_score2": score2,
                        "lm_score_diff": abs(score1 - score2),
                    }
                )

    return pairs


def assign_quantiles(pairs: list[dict], n_quantiles: int = 10) -> list[dict]:
    """Assign quantile bins to pairs based on LM score difference."""
    # Separate by pair type
    same_verb = [p for p in pairs if p["pair_type"] == "same_verb"]
    different_verb = [p for p in pairs if p["pair_type"] == "different_verb"]

    # Compute quantiles separately for each type
    for pair_list in [same_verb, different_verb]:
        if not pair_list:
            continue

        scores = np.array([p["lm_score_diff"] for p in pair_list])
        quantile_edges = np.quantile(scores, np.linspace(0, 1, n_quantiles + 1))

        for pair in pair_list:
            score = pair["lm_score_diff"]
            quantile = np.searchsorted(quantile_edges[1:], score)
            pair["quantile"] = int(quantile)

    return same_verb + different_verb


def convert_to_items(pairs: list[dict]) -> list[Item]:
    """Convert pair dicts to Item objects for 2AFC."""
    items = []

    # Create a simple 2AFC template ID (reuse for all pairs)
    template_id = uuid4()

    for pair in pairs:
        item = Item(
            item_template_id=template_id,
            rendered_elements={
                "option_a": pair["text1"],
                "option_b": pair["text2"],
            },
            item_metadata={
                "pair_type": pair["pair_type"],
                "item1_id": pair["item1_id"],
                "item2_id": pair["item2_id"],
                "lm_score1": pair["lm_score1"],
                "lm_score2": pair["lm_score2"],
                "lm_score_diff": pair["lm_score_diff"],
                "quantile": pair["quantile"],
                **{
                    k: v
                    for k, v in pair.items()
                    if k
                    not in [
                        "text1",
                        "text2",
                        "item1_id",
                        "item2_id",
                        "lm_score1",
                        "lm_score2",
                        "lm_score_diff",
                        "quantile",
                    ]
                },
            },
        )
        items.append(item)

    return items


def main(item_limit: int | None = None) -> None:
    """Generate 2AFC pairs from cross-product.

    Parameters
    ----------
    item_limit : int | None
        Limit number of cross-product items to process.
        If None, process all items.
    """
    print("=" * 60)
    print("2AFC PAIR GENERATION")
    print("=" * 60)

    # Paths
    base_dir = Path(__file__).parent
    items_path = base_dir / "items" / "cross_product_items.jsonl"
    templates_path = base_dir / "templates" / "generic_frames.jsonl"
    lexicons_dir = base_dir / "lexicons"
    output_path = base_dir / "items" / "2afc_pairs.jsonl"

    print("\n[1/6] Loading cross-product items...")
    items = load_cross_product_items(str(items_path), limit=item_limit)
    print(f"✓ Loaded {len(items)} cross-product items")

    print("\n[2/6] Loading templates...")
    templates = load_templates(str(templates_path))
    print(f"✓ Loaded {len(templates)} templates")

    print("\n[3/6] Loading lexicons...")
    lexicons = load_lexicons(lexicons_dir)
    print(f"✓ Loaded {len(lexicons)} lexicons")
    for name, lexicon in lexicons.items():
        print(f"  - {name}: {len(lexicon)} entries")

    print("\n[4/6] Filling templates...")
    filled_texts = fill_templates_with_mixed_strategy(items, templates, lexicons)
    print(f"✓ Filled {len(filled_texts)} items")

    if not filled_texts:
        print("Error: No items were successfully filled. Exiting.")
        return

    # Show examples
    print("\nExample filled texts:")
    for i, (_item_id, text) in enumerate(list(filled_texts.items())[:3]):
        print(f"  {i + 1}. {text}")

    print("\n[5/6] Scoring with language model...")
    cache_dir = base_dir / ".cache"
    lm_scores = score_with_language_model(
        filled_texts, cache_dir=cache_dir, model_name="gpt2"
    )
    print(f"✓ Scored {len(lm_scores)} items")

    print("\n[6/6] Creating minimal pairs...")
    pairs = create_pairs(items, filled_texts, lm_scores)
    print(f"✓ Created {len(pairs)} raw pairs")

    if not pairs:
        print("Error: No pairs were created. Exiting.")
        return

    pairs = assign_quantiles(pairs, n_quantiles=10)
    print(f"✓ Assigned quantiles to {len(pairs)} pairs")

    # Convert to Items
    pair_items = convert_to_items(pairs)

    # Save
    print(f"\n[7/7] Saving to {output_path}...")
    with open(output_path, "w") as f:
        for item in pair_items:
            f.write(item.model_dump_json() + "\n")

    print(f"✓ Saved {len(pair_items)} 2AFC pairs")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    same_verb_count = sum(1 for p in pairs if p["pair_type"] == "same_verb")
    different_verb_count = sum(1 for p in pairs if p["pair_type"] == "different_verb")
    print(f"  - Same-verb pairs: {same_verb_count}")
    print(f"  - Different-verb pairs: {different_verb_count}")
    print(f"  - Total pairs: {len(pairs)}")
    print(f"  - Output: {output_path}")
    print("=" * 60)


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
