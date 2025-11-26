# Items Module

The `bead.items` module provides task-type-specific utilities for creating experimental items.

## Task-Type Utilities

The items module provides 8 task-type-specific utilities for programmatic item creation. All utilities follow a consistent API pattern.

### Forced Choice

Create N-alternative forced choice items (2AFC, 3AFC, etc.):

```python
from bead.items.forced_choice import create_forced_choice_item

# Create 2AFC item
item = create_forced_choice_item(
    "The cat sleeps",
    "The cat sleep",
)

# Create 3AFC item
item = create_forced_choice_item(
    "Option A",
    "Option B",
    "Option C",
)

# With metadata
item = create_forced_choice_item(
    "The cat sleeps",
    "The cat sleep",
    metadata={"condition": "agreement"},
)
```

**Batch creation from groups**:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.items.item import Item

# Load existing source items from cross-product items
# Note: tests cd to fixtures dir, so paths are relative to tests/fixtures/api_docs/
source_items = read_jsonlines(
    Path("items/cross_product_items.jsonl"),
    Item,
)

# Create 2AFC items within groups (group by verb_lemma metadata)
# This will create pairs of items that share the same verb
items = create_forced_choice_items_from_groups(
    items=source_items,
    group_by=lambda item: item.item_metadata["verb_lemma"],
    n_alternatives=2,
    extract_text=lambda item: item.rendered_elements.get("template_string", ""),
)

print(f"Created {len(items)} 2AFC items from {len(source_items)} source items")
```

### Ordinal Scale

Create Likert-scale or slider items:

```python
from bead.items.ordinal_scale import create_ordinal_scale_item

# Create 7-point Likert item
item = create_ordinal_scale_item(
    text="How natural is this sentence?",
    scale_bounds=(1, 7),
    prompt="Rate the sentence:",
    scale_labels={1: "Very unnatural", 7: "Very natural"},
)

# Default 7-point scale
item = create_ordinal_scale_item(
    text="The cat sleeps",
)
```

**Batch creation**:

```python
from bead.items.ordinal_scale import create_ordinal_scale_items_from_texts

sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]

items = create_ordinal_scale_items_from_texts(
    sentences,
    scale_bounds=(1, 7),
    metadata_fn=lambda text: {"length": len(text)},
)
```

### Binary

Create yes/no or true/false items:

```python
from bead.items.binary import create_binary_item

item = create_binary_item(
    text="Is this sentence grammatical?",
    prompt="Judge grammaticality:",
    binary_options=("Yes", "No"),
)

print(f"Created binary item with options: {item.rendered_elements.get('options')}")
```

### Categorical

Create items with unordered categories (NLI, semantic relations):

```python
from bead.items.categorical import create_categorical_item

item = create_categorical_item(
    text="All dogs bark",
    categories=["entailment", "contradiction", "neutral"],
    prompt="What is the relationship?",
)

# Specialized NLI helper
from bead.items.categorical import create_nli_item

item = create_nli_item(
    premise="All dogs bark",
    hypothesis="Some dogs bark",
)
```

### Free Text

Create open-ended text response items:

```python
from bead.items.free_text import create_free_text_item

item = create_free_text_item(
    text="Translate this sentence to Spanish:",
    prompt="Enter translation:",
    max_length=500,
)
```

### Cloze

Create fill-in-the-blank items:

```python
from bead.items.cloze import create_simple_cloze_item

item = create_simple_cloze_item(
    text="The quick brown fox",
    blank_positions=[1],  # "quick" is blank
    blank_labels=["adjective"],
)
```

### Multi-Select

Create checkbox-style items:

```python
from bead.items.multi_select import create_multi_select_item

item = create_multi_select_item(
    "grammatical",
    "natural",
    "formal",
    "colloquial",
    min_selections=1,
    max_selections=3,
)

n_options = len([k for k in item.rendered_elements if k.startswith("option_")])
print(f"Created multi-select item with {n_options} options")
```

### Magnitude

Create unbounded numeric value items:

```python
from bead.items.magnitude import create_magnitude_item

item = create_magnitude_item(
    text="Reading time in milliseconds:",
    unit="ms",
    bounds=(0, 10000),
    prompt="Enter reading time:",
)

print(f"Created magnitude item with unit: {item.item_metadata.get('unit')}")
```

## Language Model Scoring

Score items with language models:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer

# Load items from fixtures
source_items = read_jsonlines(
    Path("items/cross_product_items.jsonl"),
    Item,
)

# Create scorer
scorer = LanguageModelScorer(
    model_name="gpt2",
    cache_dir=Path(".cache/scoring"),
    device="cpu",
    text_key="template_string",
)

# Score first few items
items_to_score = source_items[:3]
scores = scorer.score_batch(items_to_score)

# Add scores to metadata
for item, score in zip(items_to_score, scores, strict=True):
    item.item_metadata["lm_score"] = score

print(f"Scored {len(items_to_score)} items")
```

## Item Validation

Validate items conform to task-type requirements:

```python
from bead.items.ordinal_scale import create_ordinal_scale_item
from bead.items.validation import (
    get_task_type_requirements,
    infer_task_type_from_item,
    validate_item_for_task_type,
)

# Create an item to validate
item = create_ordinal_scale_item(text="The cat sleeps", scale_bounds=(1, 7))

# Validate structure
validate_item_for_task_type(item, "ordinal_scale")  # Raises ValueError if invalid
print("Item is valid for ordinal_scale")

# Infer task type
task_type = infer_task_type_from_item(item)
print(f"Inferred task type: {task_type}")

# Get requirements
reqs = get_task_type_requirements("ordinal_scale")
print(f"Requirements: {list(reqs.keys())}")
```

## Complete Example

From [gallery/eng/argument_structure/create_2afc_pairs.py](../../../gallery/eng/argument_structure/create_2afc_pairs.py):

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer

# Load source items (already in Item format)
source_items = read_jsonlines(
    Path("items/cross_product_items.jsonl"),
    Item,
)

print(f"Loaded {len(source_items)} source items")

# Score with language model (score first 10 for speed)
scorer = LanguageModelScorer(
    model_name="gpt2",
    cache_dir=Path(".cache/scoring"),
    device="cpu",
    text_key="template_string",
)
items_to_score = source_items[:10]
scores = scorer.score_batch(items_to_score)

# Add scores to metadata
for item, score in zip(items_to_score, scores, strict=True):
    item.item_metadata["lm_score"] = score

print(f"Scored {len(items_to_score)} items")

# Create 2AFC items grouped by verb
afc_items = create_forced_choice_items_from_groups(
    items=items_to_score,
    group_by=lambda item: item.item_metadata["verb_lemma"],
    n_alternatives=2,
    extract_text=lambda item: item.rendered_elements.get("template_string", ""),
)

print(f"Created {len(afc_items)} 2AFC items")

# Save example (commented out for testing)
# from bead.data.serialization import write_jsonlines
# write_jsonlines(afc_items, Path("output/2afc_items.jsonl"))
```

## Design Principles

1. **NO Silent Fallbacks**: All errors raise `ValueError` with descriptive messages
2. **Strict Validation**: Use `zip(..., strict=True)`, explicit parameter checks
3. **Consistent API**: Same pattern across all 8 task types
4. **Automatic Metadata**: Utilities populate task-specific metadata (n_options, scale_min/max, etc.)

## Task Type Summary

| Task Type | Use For | Key Function |
|-----------|---------|--------------|
| `forced_choice` | N-AFC items | `create_forced_choice_item()` |
| `ordinal_scale` | Likert, slider | `create_ordinal_scale_item()` |
| `binary` | Yes/No | `create_binary_item()` |
| `categorical` | NLI, relations | `create_categorical_item()` |
| `free_text` | Open-ended | `create_free_text_item()` |
| `cloze` | Fill-in-blank | `create_cloze_item()` |
| `multi_select` | Checkboxes | `create_multi_select_item()` |
| `magnitude` | Numeric | `create_magnitude_item()` |

## Next Steps

- [Lists module](lists.md): Partition items into balanced lists
- [CLI reference](../cli/items.md): Command-line equivalents
- [Gallery example](../../../gallery/eng/argument_structure/create_2afc_pairs.py): Full working script
