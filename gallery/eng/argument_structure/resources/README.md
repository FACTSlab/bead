# Reference Data

This directory contains reference CSV files used to generate the controlled lexicons for the argument structure project.

## Files

### bleached_nouns.csv
Controlled noun inventory covering key semantic classes with minimal semantic content:
- **Columns**: `word`, `semantic_class`, `number`, `countability`
- **Purpose**: Fill NP slots in templates with semantically light nouns
- **Examples**: person, people, thing, stuff, place

### bleached_verbs.csv
Controlled verb inventory for filling clausal complement heads:
- **Columns**: `word`, `semantic_class`, `aspect`, `valency`, etc.
- **Purpose**: Fill heads of embedded clauses (finite, infinitival)
- **Examples**: do, be, have, go, get, make

### bleached_adjectives.csv
Controlled adjective inventory for small clause predicates:
- **Columns**: `word`, `semantic_class`, `gradability`, `stage_vs_individual`, `notes`
- **Purpose**: Fill ADJ slots (will use MLM filling after exhaustive noun/verb filling)
- **Examples**: good, bad, right, wrong, certain, ready

## Usage

These CSV files are read by `generate_lexicons.py` to create the JSONLines lexicon files used in the bead pipeline:

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

## Source

These files were originally created in the `presentations/` directory as reference data for project planning and have been moved here to be part of the self-contained gallery project structure.
