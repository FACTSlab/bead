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
from pathlib import Path

from utils.morphology import MorphologyExtractor
from utils.verbnet_parser import VerbNetExtractor

from bead.resources.adapters.cache import AdapterCache
from bead.resources.lexicon import Lexicon
from bead.resources.lexical_item import LexicalItem
from bead.resources.loaders import from_csv  # NEW: Use bead loader utilities


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

    csv_path = resources_dir / "bleached_nouns.csv"

    # REFACTORED: Use bead/resources/loaders.py
    noun_lexicon = from_csv(
        path=csv_path,
        name="bleached_nouns",
        column_mapping={"word": "lemma"},  # CSV "word" → LexicalItem "lemma"
        feature_columns=["number", "countability"],
        attribute_columns=["semantic_class"],
        language_code="eng",
        description="Controlled noun inventory for templates",
    )

    print(f"Loaded {len(noun_lexicon.items)} bleached nouns from {csv_path}")

    output_path = lexicons_dir / "bleached_nouns.jsonl"
    noun_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 3. Generate bleached verbs lexicon
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED VERBS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "bleached_verbs.csv"

    # REFACTORED: Use bead/resources/loaders.py
    bleached_verb_lexicon = from_csv(
        path=csv_path,
        name="bleached_verbs",
        column_mapping={"word": "lemma"},
        feature_columns=["tense"],
        attribute_columns=["semantic_class"],
        language_code="eng",
        description="Controlled verb inventory for templates",
    )

    print(f"Loaded {len(bleached_verb_lexicon.items)} bleached verbs from {csv_path}")

    output_path = lexicons_dir / "bleached_verbs.jsonl"
    bleached_verb_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 4. Generate bleached adjectives lexicon
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED ADJECTIVES LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "bleached_adjectives.csv"

    # REFACTORED: Use bead/resources/loaders.py
    adj_lexicon = from_csv(
        path=csv_path,
        name="bleached_adjectives",
        column_mapping={"word": "lemma"},
        attribute_columns=["semantic_class"],
        language_code="eng",
        description="Controlled adjective inventory for templates",
    )

    print(f"Loaded {len(adj_lexicon.items)} bleached adjectives from {csv_path}")

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

    # 7. Generate "be" verb forms lexicon
    print("\n" + "=" * 80)
    print("GENERATING BE VERB LEXICON")
    print("=" * 80)

    # Manually create forms of "be" with proper features
    # UniMorph doesn't have "be" so we create them manually
    be_forms_data = [
        # Present tense forms
        {"form": "am", "tense": "PRS", "person": "1", "number": "SG"},
        {"form": "is", "tense": "PRS", "person": "3", "number": "SG"},
        {"form": "are", "tense": "PRS", "person": "2", "number": "SG"},
        {"form": "are", "tense": "PRS", "person": "1", "number": "PL"},
        {"form": "are", "tense": "PRS", "person": "2", "number": "PL"},
        {"form": "are", "tense": "PRS", "person": "3", "number": "PL"},
        # Past tense forms
        {"form": "was", "tense": "PST", "person": "1", "number": "SG"},
        {"form": "was", "tense": "PST", "person": "3", "number": "SG"},
        {"form": "were", "tense": "PST", "person": "2", "number": "SG"},
        {"form": "were", "tense": "PST", "person": "1", "number": "PL"},
        {"form": "were", "tense": "PST", "person": "2", "number": "PL"},
        {"form": "were", "tense": "PST", "person": "3", "number": "PL"},
        # Participles
        {"form": "being", "verb_form": "V.PTCP", "tense": "PRS"},
        {"form": "been", "verb_form": "V.PTCP", "tense": "PST"},
    ]

    be_items: dict[str, LexicalItem] = {}
    for be_data in be_forms_data:
        form = be_data.pop("form")
        item = LexicalItem(
            lemma="be",
            form=form,
            pos="V",
            language_code="eng",
            features={"pos": "V", **be_data},
            attributes={},
        )
        # Use unique key combining form and features for deduplication
        key = str(item.id)
        be_items[key] = item

    print(f"Created {len(be_items)} forms of 'be'")

    be_lexicon = Lexicon(
        name="be_forms",
        description="Inflected forms of auxiliary 'be'",
        language_code="eng",
        items=be_items,
    )

    output_path = lexicons_dir / "be_forms.jsonl"
    be_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("LEXICON GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {7} lexicon files:")
    print(f"  1. verbnet_verbs.jsonl:       {len(verb_items_dict)} entries")
    print(f"  2. bleached_nouns.jsonl:      {len(noun_lexicon.items)} entries")
    print(f"  3. bleached_verbs.jsonl:      {len(bleached_verb_lexicon.items)} entries")
    print(f"  4. bleached_adjectives.jsonl: {len(adj_lexicon.items)} entries")
    print(f"  5. prepositions.jsonl:        {len(prep_items)} entries")
    print(f"  6. determiners.jsonl:         {len(det_items)} entries")
    print(f"  7. be_forms.jsonl:            {len(be_items)} entries")
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
