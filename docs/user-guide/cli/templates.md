# Templates Module

The templates module handles Stage 2 of the pipeline: filling template slots with lexical items.

## Filling Strategies

Template filling generates experimental stimuli by substituting lexical items into template slots. Different strategies balance coverage, randomness, and computational cost.

### Exhaustive Filling

Generate all valid slot combinations:

```bash
bead templates fill templates/generic_frames.jsonl lexicons/bleached_nouns.jsonl lexicons/determiners.jsonl lexicons/bleached_verbs.jsonl lexicons/prepositions.jsonl lexicons/be_forms.jsonl filled_templates/all_combinations.jsonl \
    --strategy random \
    --max-combinations 10 \
    --random-seed 42
```

For demonstration, we use random sampling with a small limit. Exhaustive filling produces every possible combination respecting constraints but can generate thousands of combinations with multiple lexicons. Multiple lexicons are automatically merged.

### Random Sampling

Sample combinations randomly:

```bash
bead templates fill templates/generic_frames.jsonl lexicons/bleached_nouns.jsonl lexicons/determiners.jsonl lexicons/bleached_verbs.jsonl lexicons/prepositions.jsonl lexicons/be_forms.jsonl filled_templates/random_sample.jsonl \
    --strategy random \
    --max-combinations 10 \
    --random-seed 42
```

Random sampling scales to large combinatorial spaces. The `--random-seed` parameter ensures reproducibility.

### Stratified Filling

Balance samples across strata defined by metadata:

```bash
bead templates fill templates/generic_frames.jsonl lexicons/bleached_nouns.jsonl lexicons/determiners.jsonl lexicons/bleached_verbs.jsonl lexicons/prepositions.jsonl lexicons/be_forms.jsonl filled_templates/stratified_sample.jsonl \
    --strategy stratified \
    --max-combinations 10 \
    --grouping-property pos
```

Stratified filling ensures balanced representation of categories (e.g., part of speech, verb classes, animacy levels).

### Constraint Satisfaction

Apply constraints during filling to filter invalid combinations:

```bash
bead resources create-constraint constraints/verb_constraint.jsonl \
    --type intensional \
    --slot verb \
    --expression "self.pos == 'VERB'"

bead templates fill templates/generic_frames.jsonl lexicons/bleached_nouns.jsonl lexicons/determiners.jsonl lexicons/bleached_verbs.jsonl lexicons/prepositions.jsonl lexicons/be_forms.jsonl filled_templates/constrained.jsonl \
    --strategy random \
    --max-combinations 10 \
    --random-seed 42 \
    --constraints constraints/verb_constraint.jsonl
```

Constraints filter combinations that violate conditions. See [Resources](resources.md#creating-constraints) for creating constraints.

### MLM-Based and Mixed Strategies

**Note**: MLM-based (Masked Language Model) and mixed strategies require HuggingFace model integration and are currently available through the Python API. Use random or stratified sampling for large spaces in CLI workflows.

## Estimating Combinations

Preview combinatorial space size before filling:

```bash
bead templates estimate-combinations templates/generic_frames.jsonl lexicons/bleached_nouns.jsonl lexicons/determiners.jsonl lexicons/bleached_verbs.jsonl lexicons/prepositions.jsonl lexicons/be_forms.jsonl
```

The command outputs the estimated number of combinations and provides recommendations for which filling strategy to use. Multiple lexicons are automatically merged.

## Sampling Combinations

Draw stratified samples from large spaces:

```bash
bead templates sample-combinations templates/generic_frames.jsonl lexicons/bleached_nouns.jsonl lexicons/determiners.jsonl lexicons/bleached_verbs.jsonl lexicons/prepositions.jsonl lexicons/be_forms.jsonl filled_templates/sampled.jsonl \
    --n-samples 10 \
    --seed 42
```

Uses stratified sampling for balanced coverage across the combinatorial space. Multiple lexicons are automatically merged.

## Filtering Filled Templates

Filter existing filled templates by criteria:

```bash
bead templates filter-filled filled_templates/generic_frames_filled.jsonl filled_templates/filtered_length.jsonl \
    --min-length 5 \
    --max-length 15

bead templates filter-filled filled_templates/generic_frames_filled.jsonl filled_templates/transitive_only.jsonl \
    --template-name "transitive"

bead templates filter-filled filled_templates/generic_frames_filled.jsonl filled_templates/exhaustive_only.jsonl \
    --strategy "exhaustive"

bead templates filter-filled filled_templates/generic_frames_filled.jsonl filled_templates/filtered_multi.jsonl \
    --min-length 6 \
    --template-name "transitive"
```

## Merging Filled Templates

Combine multiple filled template files:

```bash
bead templates merge-filled filled_templates/batch1.jsonl filled_templates/batch2.jsonl filled_templates/merged.jsonl
```

With deduplication by ID:

```bash
bead templates merge-filled filled_templates/batch1.jsonl filled_templates/batch2.jsonl filled_templates/merged_unique.jsonl \
    --deduplicate
```

The command reports duplicate count when deduplication is enabled.

## Export Formats

Export filled templates to alternative formats for analysis or external tools.

### Export to CSV

```bash
bead templates export-csv filled_templates/generic_frames_filled.jsonl exports/all.csv
```

Output includes columns: `id`, `template_id`, `template_name`, `rendered_text`, `strategy_name`, `slot_count`.

### Export to JSON

```bash
bead templates export-json filled_templates/generic_frames_filled.jsonl exports/all.json

bead templates export-json filled_templates/generic_frames_filled.jsonl exports/all_pretty.json --pretty
```

Exports as JSON array (not JSONL). Use for compatibility with tools that don't support JSONL.

## Validation

Check filled templates for completeness:

```bash
bead templates validate-filled filled_templates/generic_frames_filled.jsonl
```

Validation checks for all slots filled, nonempty rendered text, and valid template references.

## Statistics

View statistics on filled templates:

```bash
bead templates show-stats filled_templates/generic_frames_filled.jsonl
```

The command displays filled template counts, unique templates, strategies used, and text length statistics.

## Workflow Example

Complete workflow from templates to filled templates:

```bash
bead templates fill templates/generic_frames.jsonl lexicons/bleached_nouns.jsonl lexicons/determiners.jsonl lexicons/bleached_verbs.jsonl lexicons/prepositions.jsonl lexicons/be_forms.jsonl filled_templates/workflow_sample.jsonl \
    --strategy random \
    --max-combinations 20 \
    --random-seed 42

bead templates filter-filled filled_templates/workflow_sample.jsonl filled_templates/workflow_filtered.jsonl \
    --min-length 4 \
    --max-length 10

bead templates validate-filled filled_templates/workflow_filtered.jsonl

bead templates export-csv filled_templates/workflow_filtered.jsonl exports/workflow.csv
```

## Next Steps

Once templates are filled:

1. [Create experimental items](items.md) from filled templates
2. [Partition items into lists](lists.md) with constraints
3. [Deploy experiments](deployment.md) to jsPsych/JATOS

For complete API documentation, see [bead.templates API reference](../api/templates.md).
