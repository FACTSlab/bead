# Reference Data

This directory contains reference CSV files used to generate the controlled lexicons for the argument structure project.

## Files

### bleached_nouns.csv

This file contains a controlled noun inventory (41 nouns) covering key semantic classes with minimal semantic content. The CSV has columns for `word`, `semantic_class`, `number`, and `countability`. Semantic classes include animates (person, people, group, organization), inanimate objects (thing, stuff), locations (place, area), temporals (time), abstracts (information, idea, reason, matter, situation, way, level, event, activity, amount, part), and other-marked variants. These fill NP slots in templates.

### bleached_verbs.csv

The verb inventory includes columns for `word`, `semantic_class`, `aspect`, `valency`, and other morphological features. These 8 verbs (do, be, have, go, get, make, happen, come) are used to fill the heads of embedded clauses, both finite and infinitival.

### bleached_adjectives.csv

This inventory has 11 adjectives with columns for semantic class, gradability, and stage/individual-level distinctions. After exhaustive noun and verb filling, these adjectives (good, bad, right, wrong, okay, certain, ready, done, different, same, other) fill ADJ slots using MLM filling strategies.

### determiners.csv

A minimal determiner inventory (3 items: a, the, some) with columns for `lemma` and `pos`. These fill determiner slots in NP templates.

### prepositions.csv

A comprehensive English preposition inventory (53 items) with columns for `lemma` and `pos`. Includes spatial (in, on, under), temporal (during, since), and abstract prepositions (about, regarding). These fill PP slots and prepositional arguments.

### be_forms.csv

Inflected forms of the auxiliary verb "be" (14 forms) with columns for `lemma`, `form`, `pos`, `tense`, `person`, `number`, and `verb_form`. Includes all present tense forms (am, is, are), past tense forms (was, were), and participles (being, been). UniMorph doesn't provide "be" inflections, so this file serves as a manual resource.

## Usage

These CSV files are read by `generate_lexicons.py` to create the JSONLines lexicon files used throughout the bead pipeline:

```python
import csv
from bead.resources.models import LexicalItem

with open("resources/bleached_nouns.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        item = LexicalItem(
            lemma=row["word"],
            pos="NOUN",
            features={"number": row["number"], "countability": row["countability"]},
            attributes={"semantic_class": row["semantic_class"]}
        )
```

### Pipeline Integration

The generated lexicons appear throughout the pipeline. During template filling, `bleached_nouns.jsonl` fills NP slots exhaustively, while `bleached_verbs.jsonl` handles embedded clause verbs and `bleached_adjectives.jsonl` uses MLM strategies for ADJ slots. Additional lexicons for determiners and prepositions complete the syntactic frames.

When creating 2AFC pairs, the `create_2afc_pairs.py` script uses these lexicons for inline template filling, with an exhaustive strategy ensuring all noun-verb combinations get tested. The deployment stage references lexicons in metadata for provenance tracking and validates item generation for reproducibility.

### Design Rationale

The bleached lexicon approach minimizes semantic confounds in acceptability judgments. By using generic words with broad meanings (person, thing, do, be), we avoid unusual or domain-specific vocabulary that might trigger pragmatic effects. All items are high-frequency and common in everyday language. This design ensures that judgments reflect syntactic acceptability rather than semantic plausibility or world knowledge, and similar inventories can be created for other languages.

## Source

These files were originally created in the `presentations/` directory as reference data for project planning and have been moved here to be part of the self-contained gallery project structure.
