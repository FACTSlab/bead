#!/usr/bin/env python3
"""
Generate JSONL lexicon files for argument structure alternation dataset

This script creates all required lexicons:
Output:
    lexicons/verbs.jsonl              - Active verbs with simplePast forms
    lexicons/bleached_nouns.jsonl     - Semantically light nouns
    lexicons/bleached_verbs.jsonl     - Semantically light verbs
    lexicons/bleached_adjectives.jsonl - Semantically light adjectives
    lexicons/prepositions.jsonl       - Igbo prepositions (nà)
    lexicons/determiners.jsonl        - Igbo demonstrative determiners (ahụ̀)
"""

import argparse
import csv
import json                          
from typing import List
import pandas as pd
from pathlib import Path

from bead.resources.lexicon import Lexicon
from bead.resources.lexical_item import LexicalItem


def main(verb_limit: int | None = None, save_csv: bool = True) -> None:
    """
    Generate all lexicon files for the Igbo argument structure dataset.
    
    Args:
        verb_limit: Optional limit on number of verbs to process (for testing)
        save_csv: Whether to save intermediate CSV files to resources/
    """
    # Set up paths
    base_dir = Path(__file__).parent
    lexicons_dir = base_dir / "lexicons"
    resources_dir = base_dir / "resources"
    
    # The dataset used for the project is from Nkowaokwu
    # https://www.kaggle.com/organizations/nkowaokwu/datasets
    dataset_path = base_dir / "igbo-api-dataset.json"

    # Ensure directories exist
    lexicons_dir.mkdir(exist_ok=True)
    resources_dir.mkdir(exist_ok=True)

    # 1. Generate verbs lexicon from local JSON file
    print("=" * 80)
    print("GENERATING VERBS LEXICON FROM NKỌWA OKWU IGBO DATASET")
    print("=" * 80)

    print(f"Loading Igbo dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        igbo_data = json.load(f)

    active_verbs_data = [
        entry for entry in igbo_data 
        if entry.get('wordClass') == 'Active verb'
    ]
    print(f"Found {len(active_verbs_data)} active verbs")

    lexicon = Lexicon(name="verbs")
    verbs: List[LexicalItem] = []

    if verb_limit is not None:
        print(f"[TEST MODE] Limiting to first {verb_limit} verbs")

    processed_count = 0

    for entry in active_verbs_data:
        if verb_limit is not None and processed_count >= verb_limit:
            break

        
        # get the simplepast verbs: entry['tenses']['simplePast']
        tenses = entry.get('tenses', {})
        simple_past = tenses.get('simplePast', '')
        
        # Skip entries without simplePast form
        if not simple_past or not simple_past.strip():
            continue

        verb = LexicalItem(
            lemma=entry['word'],                    # Base form (lemma)
            form=simple_past,                       # simplePast form
            language_code="ibo",                    
            features={"pos": "V", "tense": "PST", "definition": entry.get('definitions', '')},
            source="NkowaOkwu"                      
        )
        verbs.append(verb)
        processed_count += 1

    lexicon.add_many(verbs)

    print(f"Total verbs with simplePast extracted: {len(verbs)}")

    lexicon.to_jsonl(str(lexicons_dir / "verbs.jsonl"))
    print(f"✓ Saved to {lexicons_dir / 'verbs.jsonl'}")

    
    # 2. Generate bleached nouns
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED NOUNS LEXICON")
    print("=" * 80)

    bleached_nouns = pd.DataFrame(columns=['word', 'semantic_class', 'number', 'countability'])
   
    # These are semantically light nouns commonly used in Igbo
    word = [
        'mmadụ',      # person
        'ndị mmadụ',  # people
        'ihe',        # thing
        'ihe ndị',    # things
        'ebe',        # place
        'ebe ndị',    # places
        'onye',       # someone/person
        'ndị'         # people/group
    ]
    semantic_class = [
        'animate', 'animate',           # mmadụ, ndị mmadụ
        'inanimate_object', 'inanimate_object',  # ihe, ihe ndịa
        'location', 'location',         # ebe, ebe ndịa
        'animate', 'animate'          # onye, ndị
    ]
    number = ['singular', 'plural', 'singular', 'plural', 'singular', 'plural', 'singular', 'singular']
    countability = ['countable'] * len(word)

    bleached_nouns['word'] = word
    bleached_nouns['semantic_class'] = semantic_class
    bleached_nouns['number'] = number
    bleached_nouns['countability'] = countability

    if save_csv:
        bleached_nouns.to_csv(resources_dir / 'bleached_nouns.csv', index=False)

    lexicon = Lexicon(name="bleached_nouns")

    with open(resources_dir / "bleached_nouns.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="ibo",
                features={"pos": "NOUN", "number": row["number"], "countability": row["countability"], "semantic_class": row["semantic_class"]},
            )
            lexicon.add(item)

    lexicon.to_jsonl(str(lexicons_dir / "bleached_nouns.jsonl"))
    print(f"Created {len(bleached_nouns)} bleached nouns.")
    print(f"✓ Saved to {lexicons_dir / 'bleached_nouns.jsonl'}")


    # 3. Generate bleached verbs
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED VERBS LEXICON")
    print("=" * 80)

    bleached_verbs = pd.DataFrame(columns=['word', 'simplePast', 'semantic_class', 'valency'])

    
    word = ['me', 'ga', 'bịa', 'nweta', 'nye', 'hụ', 'mara']
    # Translations: do, go, come, get/obtain, give, see, know
    
    simple_past = ['mere', 'gara', 'bịara', 'nwetara', 'nyere', 'hụrụ', 'maara']
    
    semantic_class = ['activity', 'motion', 'motion', 'transfer', 'transfer', 'perception', 'cognition']
    valency = ['transitive', 'intransitive', 'intransitive', 'transitive', 'ditransitive', 'transitive', 'transitive']

    bleached_verbs['word'] = word
    bleached_verbs['simplePast'] = simple_past          
    bleached_verbs['semantic_class'] = semantic_class
    bleached_verbs['valency'] = valency

    if save_csv:
        bleached_verbs.to_csv(resources_dir / "bleached_verbs.csv", index=False)

    lexicon = Lexicon(name="bleached_verbs")

    with open(resources_dir / "bleached_verbs.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                form=row["simplePast"],                 
                language_code="ibo",                   
                features={"pos": "V", "tense": "PST", "semantic_class": row["semantic_class"], "valency": row["valency"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl(str(lexicons_dir / "bleached_verbs.jsonl"))
    print(f"Created {len(bleached_verbs)} bleached verbs.")
    print(f"✓ Saved to {lexicons_dir / 'bleached_verbs.jsonl'}")


    # 4. Generate bleached adjectives lexicon
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED ADJECTIVES LEXICON")
    print("=" * 80)

    bleached_adjectives = pd.DataFrame(columns=['word', 'semantic_class'])

    # words from the dataset
    word = ['ọma', 'ọjọọ', 'ukwu', 'nta', 'ọcha', 'ojii', 'ọhụrụ', 'ochie']
    # Translations: good, bad, big, small, white/fair, black/dark, new, old

    semantic_class = [
        'evaluation', 'evaluation',     # ọma, ajọ
        'dimension', 'dimension',       # ukwu, nta
        'color', 'color',               # ọcha, ojii
        'age', 'age',                   # ọhụrụ, ochie
    ]

    bleached_adjectives['word'] = word
    bleached_adjectives['semantic_class'] = semantic_class

    if save_csv:
        bleached_adjectives.to_csv(resources_dir / "bleached_adjectives.csv", index=False)

    lexicon = Lexicon(name="bleached_adjectives")

    with open(resources_dir / "bleached_adjectives.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="ibo",                    
                features={"pos": "ADJ", "semantic_class": row["semantic_class"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl(str(lexicons_dir / "bleached_adjectives.jsonl"))
    print(f"Created {len(bleached_adjectives)} bleached adjectives.")
    print(f"✓ Saved to {lexicons_dir / 'bleached_adjectives.jsonl'}")


    # 5. Generate prepositions lexicon
    print("\n" + "=" * 80)
    print("GENERATING PREPOSITIONS LEXICON")                
    print("=" * 80)

    prepositions = pd.DataFrame(columns=['word', 'pos'])

    word = ['nà']
    # Translations: in/on/at
    
    pos = ['ADP'] * len(word)

    prepositions['word'] = word
    prepositions['pos'] = pos

    if save_csv:
        prepositions.to_csv(resources_dir / "prepositions.csv", index=False)

    lexicon = Lexicon(name="prepositions")

    with open(resources_dir / "prepositions.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="ibo",
                features={"pos": row["pos"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl(str(lexicons_dir / "prepositions.jsonl"))
    print(f"Created {len(prepositions)} prepositions.")
    print(f"✓ Saved to {lexicons_dir / 'prepositions.jsonl'}")


    # 6. Generate determiners lexicon
    print("\n" + "=" * 80)
    print("GENERATING DETERMINERS LEXICON")             
    print("=" * 80)

    determiners = pd.DataFrame(columns=['word', 'pos'])

    word = ['ahụ']
    # Translations: the/that
    
    pos = ['DET']

    determiners['word'] = word
    determiners['pos'] = pos

    if save_csv:
        determiners.to_csv(resources_dir / "determiners.csv", index=False)

    lexicon = Lexicon(name="determiners")

    with open(resources_dir / "determiners.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="ibo",
                features={"pos": row["pos"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl(str(lexicons_dir / "determiners.jsonl"))
    print(f"Created {len(determiners)} determiners.")
    print(f"✓ Saved to {lexicons_dir / 'determiners.jsonl'}")


    # SUMMARY
    print("\n" + "=" * 80)
    print("LEXICON GENERATION COMPLETE")
    print("=" * 80)
  
    print(f"\nGenerated {6} lexicon files:")                   
    print(f"  1. verbs.jsonl:              {len(verbs)} entries")
    print(f"  2. bleached_nouns.jsonl:     {len(bleached_nouns)} entries")
    print(f"  3. bleached_verbs.jsonl:     {len(bleached_verbs)} entries")
    print(f"  4. bleached_adjectives.jsonl:{len(bleached_adjectives)} entries")
    print(f"  5. prepositions.jsonl:       {len(prepositions)} entries")      
    print(f"  6. determiners.jsonl:        {len(determiners)} entries")  
    print(f"\nAll files saved to: {lexicons_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JSONL lexicon files for Igbo argument structure dataset"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of verbs to process (for testing)",
    )

    parser.add_argument(
        "--save_csv",
        type=bool,
        default=True,
        help="Whether to save CSV files during processing (default: True)",
    )
    
    # specify dataset path
    parser.add_argument(
        "--dataset",
        type=str,
        default="igbo-api-dataset.json",
        help="Path to the Nkọwa okwu Igbo dataset JSON file",
    )

    args = parser.parse_args()

    main(verb_limit=args.limit, save_csv=args.save_csv)