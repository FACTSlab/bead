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
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.loaders import from_csv  # NEW: Use bead loader utilities


def main(verb_limit: int | None = None) -> None:
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
            form_item.features.update(
                {
                    "verbnet_class": base_verb.features.get("verbnet_class", ""),
                    "themroles": base_verb.features.get("themroles", []),
                    "frame_count": base_verb.features.get("frame_count", 0),
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

    noun_lexicon = from_csv(
        path=csv_path,
        name="bleached_nouns",
        column_mapping={"word": "lemma"},
        feature_columns=["number", "countability", "semantic_class"],
        language_code="eng",
        description="Controlled noun inventory for templates",
        pos="NOUN",
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

    bleached_verb_lexicon = from_csv(
        path=csv_path,
        name="bleached_verbs",
        column_mapping={"word": "lemma"},
        feature_columns=["tense", "semantic_class"],
        language_code="eng",
        description="Controlled verb inventory for templates",
        pos="V",
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

    adj_lexicon = from_csv(
        path=csv_path,
        name="bleached_adjectives",
        column_mapping={"word": "lemma"},
        feature_columns=["semantic_class"],
        language_code="eng",
        description="Controlled adjective inventory for templates",
        pos="ADJ",
    )

    print(f"Loaded {len(adj_lexicon.items)} bleached adjectives from {csv_path}")

    output_path = lexicons_dir / "bleached_adjectives.jsonl"
    adj_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 5. Generate prepositions lexicon
    print("\n" + "=" * 80)
    print("GENERATING PREPOSITIONS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "prepositions.csv"

    prep_lexicon = from_csv(
        path=csv_path,
        name="prepositions",
        column_mapping={"lemma": "lemma"},
        feature_columns=["pos"],
        language_code="eng",
        description="Comprehensive English preposition inventory",
        pos="ADP",
    )

    print(f"Loaded {len(prep_lexicon.items)} prepositions from {csv_path}")

    output_path = lexicons_dir / "prepositions.jsonl"
    prep_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 6. Generate determiners lexicon
    print("\n" + "=" * 80)
    print("GENERATING DETERMINERS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "determiners.csv"

    det_lexicon = from_csv(
        path=csv_path,
        name="determiners",
        column_mapping={"lemma": "lemma"},
        feature_columns=["pos"],
        language_code="eng",
        description="Basic determiner inventory",
        pos="DET",
    )

    print(f"Loaded {len(det_lexicon.items)} determiners from {csv_path}")

    output_path = lexicons_dir / "determiners.jsonl"
    det_lexicon.to_jsonl(str(output_path))
    print(f"✓ Saved to {output_path}")

    # 7. Generate "be" verb forms lexicon
    print("\n" + "=" * 80)
    print("GENERATING BE VERB LEXICON")
    print("=" * 80)

    # Load from CSV (with custom handling for form column)
    csv_path = resources_dir / "be_forms.csv"

    import pandas as pd

    df = pd.read_csv(csv_path)

    be_items: dict[str, LexicalItem] = {}
    for _, row in df.iterrows():
        # Build features dict from all columns except lemma and form
        features = {"pos": str(row["pos"])}

        # Add optional feature columns if not empty
        for col in ["tense", "person", "number", "verb_form"]:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                features[col] = str(row[col])

        item = LexicalItem(
            lemma=str(row["lemma"]),
            form=str(row["form"]),
            language_code="eng",
            features=features,
            source="csv",
        )
        be_items[str(item.id)] = item

    print(f"Loaded {len(be_items)} forms of 'be' from {csv_path}")

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
    print(f"  5. prepositions.jsonl:        {len(prep_lexicon.items)} entries")
    print(f"  6. determiners.jsonl:         {len(det_lexicon.items)} entries")
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
