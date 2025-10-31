#!/usr/bin/env python3
"""
Generate JSONL lexicon files for the argument structure alternations dataset.

This script creates all required lexicons using the bead adapter infrastructure:
1. verbnet_verbs.jsonl - All VerbNet verbs with inflected forms
2. bleached_nouns.jsonl - Controlled noun inventory from CSV
3. bleached_verbs.jsonl - Controlled verb inventory from CSV
4. bleached_adjectives.jsonl - Controlled adjective inventory from CSV
5. prepositions.jsonl - Comprehensive English preposition list
6. determiners.jsonl - Basic determiner inventory [a, the, some]
"""

import argparse
import csv
from pathlib import Path

from utils.morphology import MorphologyExtractor
from utils.verbnet_parser import VerbNetExtractor

from bead.resources.adapters.cache import AdapterCache
from bead.resources.lexicon import Lexicon
from bead.resources.models import LexicalItem


def main(verb_limit: int | None = None):
    # Set up paths
    base_dir = Path(__file__).parent
    lexicons_dir = base_dir / "lexicons"
    resources_dir = base_dir / "resources"

    # Ensure directories exist
    lexicons_dir.mkdir(exist_ok=True)

    # Initialize adapters with caching
    cache = AdapterCache()
    verbnet = VerbNetExtractor(cache=cache)
    morph = MorphologyExtractor(cache=cache)

    # 1. Generate VerbNet verbs lexicon
    print("=" * 80)
    print("GENERATING VERBNET VERBS LEXICON")
    print("=" * 80)
    print("Extracting VerbNet verbs...")

    verb_items_dict: dict[str, LexicalItem] = {}
    base_verbs = verbnet.extract_all_verbs()

    print(f"Found {len(base_verbs)} verb-class pairs from VerbNet")

    # Apply limit if specified
    if verb_limit is not None:
        print(f"[TEST MODE] Limiting to first {verb_limit} verbs")
        base_verbs = base_verbs[:verb_limit]

    print(f"\nGetting inflected forms for {len(base_verbs)} verbs...")

    for i, base_verb in enumerate(base_verbs, 1):
        lemma = base_verb.lemma

        if i % 10 == 0 or verb_limit is not None:
            print(f"  Processed {i}/{len(base_verbs)} verbs... (current: {lemma})")

        # Get all inflected forms
        forms = morph.get_all_required_forms(lemma)

        # Add VerbNet metadata to each form
        for form_item in forms:
            # Merge attributes from base verb and morphology
            form_item.attributes.update(
                {
                    "verbnet_class": base_verb.attributes.get("verbnet_class", ""),
                    "themroles": base_verb.attributes.get("themroles", []),
                    "frame_count": base_verb.attributes.get("frame_count", 0),
                }
            )

            # Use LexicalItem's UUID as key
            verb_items_dict[str(form_item.id)] = form_item

    print(f"\nCreated {len(verb_items_dict)} verb form entries")

    verb_lexicon = Lexicon(
        name="verbnet_verbs",
        description="All VerbNet verbs with inflected forms",
        language_code="eng",
        items=verb_items_dict,
    )

    output_path = lexicons_dir / "verbnet_verbs.jsonl"
    verb_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 2. Generate bleached nouns lexicon
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED NOUNS LEXICON")
    print("=" * 80)

    noun_items: dict[str, LexicalItem] = {}
    csv_path = resources_dir / "bleached_nouns.csv"

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                pos="NOUN",
                language_code="eng",
                features={
                    "number": row["number"],
                    "countability": row["countability"],
                },
                attributes={
                    "semantic_class": row["semantic_class"],
                },
            )
            noun_items[str(item.id)] = item

    print(f"Loaded {len(noun_items)} bleached nouns from {csv_path}")

    noun_lexicon = Lexicon(
        name="bleached_nouns",
        description="Controlled noun inventory for templates",
        language_code="eng",
        items=noun_items,
    )

    output_path = lexicons_dir / "bleached_nouns.jsonl"
    noun_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 3. Generate bleached verbs lexicon
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED VERBS LEXICON")
    print("=" * 80)

    bleached_verb_items: dict[str, LexicalItem] = {}
    csv_path = resources_dir / "bleached_verbs.csv"

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                pos="VERB",
                language_code="eng",
                features={
                    "tense": row.get("tense", ""),
                },
                attributes={
                    "semantic_class": row.get("semantic_class", ""),
                },
            )
            bleached_verb_items[str(item.id)] = item

    print(f"Loaded {len(bleached_verb_items)} bleached verbs from {csv_path}")

    bleached_verb_lexicon = Lexicon(
        name="bleached_verbs",
        description="Controlled verb inventory for templates",
        language_code="eng",
        items=bleached_verb_items,
    )

    output_path = lexicons_dir / "bleached_verbs.jsonl"
    bleached_verb_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 4. Generate bleached adjectives lexicon
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED ADJECTIVES LEXICON")
    print("=" * 80)

    adj_items: dict[str, LexicalItem] = {}
    csv_path = resources_dir / "bleached_adjectives.csv"

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                pos="ADJ",
                language_code="eng",
                features={},
                attributes={
                    "semantic_class": row.get("semantic_class", ""),
                },
            )
            adj_items[str(item.id)] = item

    print(f"Loaded {len(adj_items)} bleached adjectives from {csv_path}")

    adj_lexicon = Lexicon(
        name="bleached_adjectives",
        description="Controlled adjective inventory for templates",
        language_code="eng",
        items=adj_items,
    )

    output_path = lexicons_dir / "bleached_adjectives.jsonl"
    adj_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 5. Generate prepositions lexicon
    print("\n" + "=" * 80)
    print("GENERATING PREPOSITIONS LEXICON")
    print("=" * 80)

    # Comprehensive English preposition list (52 prepositions from constraint_builder.py)
    prepositions = [
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "by",
        "concerning",
        "despite",
        "down",
        "during",
        "except",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "like",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "out",
        "outside",
        "over",
        "past",
        "regarding",
        "round",
        "since",
        "through",
        "throughout",
        "to",
        "toward",
        "towards",
        "under",
        "underneath",
        "until",
        "up",
        "upon",
        "with",
        "within",
        "without",
    ]

    prep_items: dict[str, LexicalItem] = {}

    for prep in prepositions:
        item = LexicalItem(
            lemma=prep, pos="ADP", language_code="eng", features={}, attributes={}
        )
        prep_items[str(item.id)] = item

    print(f"Created {len(prep_items)} preposition entries")

    prep_lexicon = Lexicon(
        name="prepositions",
        description="Comprehensive English preposition inventory",
        language_code="eng",
        items=prep_items,
    )

    output_path = lexicons_dir / "prepositions.jsonl"
    prep_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 6. Generate determiners lexicon
    print("\n" + "=" * 80)
    print("GENERATING DETERMINERS LEXICON")
    print("=" * 80)

    determiners = ["a", "the", "some"]
    det_items: dict[str, LexicalItem] = {}

    for det in determiners:
        item = LexicalItem(
            lemma=det, pos="DET", language_code="eng", features={}, attributes={}
        )
        det_items[str(item.id)] = item

    print(f"Created {len(det_items)} determiner entries")

    det_lexicon = Lexicon(
        name="determiners",
        description="Basic determiner inventory",
        language_code="eng",
        items=det_items,
    )

    output_path = lexicons_dir / "determiners.jsonl"
    det_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("LEXICON GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {6} lexicon files:")
    print(f"  1. verbnet_verbs.jsonl:       {len(verb_items_dict)} entries")
    print(f"  2. bleached_nouns.jsonl:      {len(noun_items)} entries")
    print(f"  3. bleached_verbs.jsonl:      {len(bleached_verb_items)} entries")
    print(f"  4. bleached_adjectives.jsonl: {len(adj_items)} entries")
    print(f"  5. prepositions.jsonl:        {len(prep_items)} entries")
    print(f"  6. determiners.jsonl:         {len(det_items)} entries")
    print(f"\nAll files saved to: {lexicons_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JSONL lexicon files for argument structure dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of VerbNet verbs to process (for testing)",
    )
    args = parser.parse_args()

    main(verb_limit=args.limit)
