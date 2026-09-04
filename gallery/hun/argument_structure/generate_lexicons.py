#!/usr/bin/env python3
"""Generate Hungarian JSONL lexicons used by the argument-structure pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.morphology import MorphologyExtractor
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon


# HELPERS

def save_lexicon(name: str, description: str, items: list[LexicalItem], output_path: Path):
    lexicon = Lexicon(
        name=name,
        description=description,
        language_code="hun",
        items=tuple(items),
    )

    lexicon.to_jsonl(str(output_path))
    return lexicon


def clean_value(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    return value


def load_csv_items(csv_path: Path, pos: str, feature_columns: list[str]) -> list[LexicalItem]:
    df = pd.read_csv(csv_path)
    items = []

    for _, row in df.iterrows():
        # `form` is last so that a CSV carrying both a lemma and an inflected
        # form still keys on the lemma. Uninflected inventories such as the
        # postposition tables only have `form`, and key on that.
        for identifier_column in ("lemma", "word", "marker", "form"):
            if identifier_column in df.columns:
                lemma = str(row[identifier_column]).strip()
                break
        else:
            raise ValueError(
                f"{csv_path.name} must contain one of these columns: "
                "'lemma', 'word', 'marker', or 'form'."
            )

        if "form" in df.columns and pd.notna(row["form"]) and str(row["form"]).strip():
            form = str(row["form"]).strip()
        else:
            form = lemma

        features = {"pos": pos}

        for column in feature_columns:
            if column not in df.columns:
                continue

            value = clean_value(row[column])

            if value is not None:
                features[column] = value

        items.append(
            LexicalItem(
                lemma=lemma,
                form=form,
                language_code="hun",
                features=features,
                source="csv",
            )
        )

    return items


# MAIN

def main(verb_limit: int | None = None):
    base_dir = Path(__file__).parent
    lexicons_dir = base_dir / "lexicons"
    resources_dir = base_dir / "resources"

    lexicons_dir.mkdir(exist_ok=True)

    morph = MorphologyExtractor()
    # 1. VERBS
    print("=" * 80)
    print("GENERATING HUNGARIAN VERBS LEXICON")
    print("=" * 80)

    base_verbs = morph.get_verb_lemmas(limit=verb_limit)

    print(f"Found {len(base_verbs)} verb lemmas")

    if verb_limit is not None:
        print(f"Limited to first {verb_limit} lemmas")

    verb_items = []

    for i, lemma in enumerate(base_verbs, 1):
        if i % 100 == 0 or verb_limit is not None:
            print(f"  Processed {i}/{len(base_verbs)} verbs (current: {lemma})")

        forms = morph.get_all_verb_forms(lemma)
        verb_items.extend(forms)

    print(f"\nCreated {len(verb_items)} verb-form entries")

    verb_lexicon = save_lexicon(
        name="verbs",
        description="Hungarian UniMorph verbs with complete inflected forms",
        items=verb_items,
        output_path=lexicons_dir / "verbs.jsonl",
    )

    print("✓ Saved verbs.jsonl")
    # 2. BLEACHED NOUNS
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED NOUNS LEXICON")
    print("=" * 80)

    noun_csv = resources_dir / "bleached_nouns.csv"

    noun_items = load_csv_items(
        noun_csv,
        pos="NOUN",
        feature_columns=[
            "case",
            "number",
            "semantic_class",
            "countability",
            "animacy",
            "harmony",
            "role",
        ],
    )

    noun_lexicon = save_lexicon(
        name="bleached_nouns",
        description="Controlled Hungarian bleached nouns with complete case-inflected surface forms",
        items=noun_items,
        output_path=lexicons_dir / "bleached_nouns.jsonl",
    )

    print(f"✓ Saved {len(noun_items)} inflected noun entries")
    # 3. BLEACHED VERBS
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED VERBS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "bleached_verbs.csv"

    bleached_verb_items = load_csv_items(
        csv_path,
        pos="V",
        feature_columns=[
            "tense",
            "semantic_class",
            "aspectual_class",
            "valency",
            "frame_type",
            "subject_role",
            "object_role",
            "oblique_role",
            "telicity",
            "lemma_vp",
            "infinitive_vp",
            "present_clause",
            "past_clause",
        ],
    )

    bleached_verb_lexicon = save_lexicon(
        name="bleached_verbs",
        description="Controlled Hungarian bleached verb inventory",
        items=bleached_verb_items,
        output_path=lexicons_dir / "bleached_verbs.jsonl",
    )

    print(f"✓ Saved {len(bleached_verb_items)} bleached verbs")
    # 4. BLEACHED ADJECTIVES
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED ADJECTIVES LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "bleached_adjectives.csv"

    adjective_items = load_csv_items(
        csv_path,
        pos="ADJ",
        feature_columns=["semantic_class", "gradability", "stage_vs_individual"],
    )

    adjective_lexicon = save_lexicon(
        name="bleached_adjectives",
        description="Controlled Hungarian adjective inventory",
        items=adjective_items,
        output_path=lexicons_dir / "bleached_adjectives.jsonl",
    )

    print(f"✓ Saved {len(adjective_items)} adjectives")
    # 5. CASE METADATA
    print("\n" + "=" * 80)
    print("GENERATING CASE METADATA LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "case_markers.csv"

    case_items = load_csv_items(
        csv_path,
        pos="CASE_SCHEMA",
        feature_columns=[
            "case",
            "harmony_pattern",
            "morphophonology",
            "concatenable_directly",
        ],
    )

    case_lexicon = save_lexicon(
        name="case_markers",
        description="Hungarian case morphology metadata. Not directly concatenated onto nouns.",
        items=case_items,
        output_path=lexicons_dir / "case_markers.jsonl",
    )

    print(f"✓ Saved {len(case_items)} case metadata entries")
    # 6. DETERMINERS
    print("\n" + "=" * 80)
    print("GENERATING DETERMINERS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "determiners.csv"

    determiner_items = load_csv_items(
        csv_path,
        pos="DET",
        feature_columns=["definiteness", "following_segment_type"],
    )

    determiner_lexicon = save_lexicon(
        name="determiners",
        description="Controlled Hungarian determiner inventory",
        items=determiner_items,
        output_path=lexicons_dir / "determiners.jsonl",
    )

    print(f"✓ Saved {len(determiner_items)} determiners")
    # 7. SUBJECT PRONOUNS
    print("\n" + "=" * 80)
    print("GENERATING SUBJECT PRONOUNS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "subject_pronouns.csv"

    pronoun_items = load_csv_items(
        csv_path,
        pos="PRON",
        feature_columns=[
            "pronoun_type",
            "case",
            "person",
            "number",
            "definiteness",
        ],
    )

    pronoun_lexicon = save_lexicon(
        name="subject_pronouns",
        description="Hungarian nominative personal pronouns",
        items=pronoun_items,
        output_path=lexicons_dir / "subject_pronouns.jsonl",
    )

    print(f"✓ Saved {len(pronoun_items)} pronouns")
    # 8. AUXILIARY VERBS
    print("\n" + "=" * 80)
    print("GENERATING AUXILIARY VERBS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "auxiliary_verbs.csv"

    auxiliary_items = load_csv_items(
        csv_path,
        pos="AUX",
        feature_columns=[
            "finiteness",
            "mood",
            "tense",
            "person",
            "number",
            "function",
        ],
    )

    auxiliary_lexicon = save_lexicon(
        name="auxiliary_verbs",
        description="Controlled Hungarian auxiliary/copular forms",
        items=auxiliary_items,
        output_path=lexicons_dir / "auxiliary_verbs.jsonl",
    )

    print(f"✓ Saved {len(auxiliary_items)} auxiliary forms")
    # 9. PREVERBS
    print("\n" + "=" * 80)
    print("GENERATING PREVERBS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "preverbs.csv"

    preverb_items = load_csv_items(
        csv_path,
        pos="PREVERB",
        feature_columns=["separable", "aspectual_effect", "stage1_manipulated"],
    )

    preverb_lexicon = save_lexicon(
        name="preverbs",
        description="Hungarian verbal prefix/preverb inventory",
        items=preverb_items,
        output_path=lexicons_dir / "preverbs.jsonl",
    )

    print(f"✓ Saved {len(preverb_items)} preverbs")
    # 10. PARTICLES
    print("\n" + "=" * 80)
    print("GENERATING PARTICLES LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "particles.csv"

    particle_items = load_csv_items(
        csv_path,
        pos="PART",
        feature_columns=["function", "stage1_manipulated"],
    )

    particle_lexicon = save_lexicon(
        name="particles",
        description="Controlled Hungarian particles and adverbials",
        items=particle_items,
        output_path=lexicons_dir / "particles.jsonl",
    )

    print(f"✓ Saved {len(particle_items)} particles")
    # 11. SPATIAL POSTPOSITIONS
    print("\n" + "=" * 80)
    print("GENERATING SPATIAL POSTPOSITIONS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "spatial_postpositions.csv"

    spatial_items = load_csv_items(
        csv_path,
        pos="POSTP",
        feature_columns=[
            "postp_type",
            "spatial_class",
            "series",
            "gov_case",
            "eng_gloss",
            "stage1",
        ],
    )

    spatial_lexicon = save_lexicon(
        name="spatial_postpositions",
        description=(
            "Hungarian bare postpositions governing a caseless (NOM) complement, "
            "in essive/lative/ablative series"
        ),
        items=spatial_items,
        output_path=lexicons_dir / "spatial_postpositions.jsonl",
    )

    print(f"✓ Saved {len(spatial_items)} spatial postpositions")
    # 12. COMPLEX POSTPOSITIONS
    print("\n" + "=" * 80)
    print("GENERATING COMPLEX POSTPOSITIONS LEXICON")
    print("=" * 80)

    csv_path = resources_dir / "complex_postpositions.csv"

    complex_postposition_items = load_csv_items(
        csv_path,
        pos="POSTP",
        feature_columns=[
            "postp_type",
            "gov_case",
            "semantic_class",
            "eng_gloss",
            "stage1",
        ],
    )

    complex_postposition_lexicon = save_lexicon(
        name="complex_postpositions",
        description=(
            "Hungarian postpositions governing an overtly case-marked complement, "
            "plus non-spatial NOM-governing postpositions"
        ),
        items=complex_postposition_items,
        output_path=lexicons_dir / "complex_postpositions.jsonl",
    )

    print(f"✓ Saved {len(complex_postposition_items)} complex postpositions")
    # SUMMARY
    print("\n" + "=" * 80)
    print("LEXICON GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"""
Generated 12 lexicon files:

  1. verbs.jsonl:                   {len(verb_lexicon.items)}
  2. bleached_nouns.jsonl:          {len(noun_lexicon.items)}
  3. bleached_verbs.jsonl:          {len(bleached_verb_lexicon.items)}
  4. bleached_adjectives.jsonl:     {len(adjective_lexicon.items)}
  5. case_markers.jsonl:            {len(case_lexicon.items)}
  6. determiners.jsonl:             {len(determiner_lexicon.items)}
  7. subject_pronouns.jsonl:        {len(pronoun_lexicon.items)}
  8. auxiliary_verbs.jsonl:         {len(auxiliary_lexicon.items)}
  9. preverbs.jsonl:                {len(preverb_lexicon.items)}
 10. particles.jsonl:               {len(particle_lexicon.items)}
 11. spatial_postpositions.jsonl:   {len(spatial_lexicon.items)}
 12. complex_postpositions.jsonl:   {len(complex_postposition_lexicon.items)}

All files saved to:
    {lexicons_dir}
"""
    )


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hungarian JSONL lexicons for the argument-structure dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of unique UniMorph verb lemmas for testing")
    args = parser.parse_args()

    main(verb_limit=args.limit)