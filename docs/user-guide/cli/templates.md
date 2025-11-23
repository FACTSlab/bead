# Templates Module

The templates module handles Stage 2 of the pipeline: filling template slots with lexical items.

## Filling Strategies

Template filling generates experimental stimuli by substituting lexical items into template slots. Different strategies balance coverage, randomness, and computational cost.

### Exhaustive Filling

Generate all valid slot combinations:

```bash
bead templates fill \
    --template templates/frames.jsonl \
    --lexicon lexicons/verbs.jsonl lexicons/nouns.jsonl \
    --strategy exhaustive \
    --output filled_templates/all_combinations.jsonl
```

Exhaustive filling produces every possible combination respecting constraints. Use for small template sets (< 1000 expected combinations).

### Random Sampling

Sample combinations randomly:

```bash
bead templates fill \
    --template templates/frames.jsonl \
    --lexicon lexicons/verbs.jsonl lexicons/nouns.jsonl \
    --strategy random \
    --n-samples 500 \
    --seed 42 \
    --output filled_templates/random_sample.jsonl
```

Random sampling scales to large combinatorial spaces. The `--seed` parameter ensures reproducibility.

### Stratified Filling

Balance samples across strata defined by metadata:

```bash
bead templates fill \
    --template templates/frames.jsonl \
    --lexicon lexicons/verbs.jsonl lexicons/nouns.jsonl \
    --strategy stratified \
    --stratify-by verb_class \
    --n-samples-per-stratum 50 \
    --output filled_templates/stratified_sample.jsonl
```

Stratified filling ensures balanced representation of categories (e.g., verb classes, animacy levels).

### Constraint Satisfaction

Apply constraints during filling to filter invalid combinations:

```bash
bead templates fill \
    --template templates/frames.jsonl \
    --lexicon lexicons/verbs.jsonl lexicons/nouns.jsonl \
    --constraints constraints/slot_restrictions.jsonl \
    --strategy exhaustive \
    --filter-invalid \
    --output filled_templates/constrained.jsonl
```

The `--filter-invalid` flag removes combinations violating constraints. Constraint types:

- **Extensional**: whitelist specific items
- **Intensional**: DSL expressions on features
- **Relational**: cross-slot requirements (e.g., subject ≠ object)

### MLM-Based and Mixed Strategies

**Note**: MLM-based (Masked Language Model) and mixed strategies require HuggingFace model integration and are deferred in the current CLI implementation. Use random or stratified sampling for large spaces.

## Estimating Combinations

Preview combinatorial space size before filling:

```bash
bead templates estimate-combinations \
    --template templates/frames.jsonl \
    --lexicon lexicons/verbs.jsonl lexicons/nouns.jsonl \
    --constraints constraints/slot_restrictions.jsonl
```

Output shows:

```
Template: basic_transitive
  Slots: subj (120 items), verb (45 items), obj (120 items)
  Total combinations (unconstrained): 648,000
  Estimated after constraints: 324,500
  Recommendation: use random or stratified sampling
```

## Sampling Combinations

Draw stratified samples from large spaces:

```bash
bead templates sample-combinations \
    --template templates/frames.jsonl \
    --lexicon lexicons/verbs.jsonl lexicons/nouns.jsonl \
    --n-samples 1000 \
    --seed 42 \
    --language-code eng \
    --output filled_templates/sampled.jsonl
```

Uses Latin hypercube sampling for balanced coverage across the combinatorial space.

## Filtering Filled Templates

Filter existing filled templates by criteria:

```bash
# By text length
bead templates filter-filled \
    --filled-templates filled_templates/all.jsonl \
    --min-length 5 \
    --max-length 15 \
    --output filled_templates/filtered_length.jsonl

# By template name
bead templates filter-filled \
    --filled-templates filled_templates/all.jsonl \
    --template-name "transitive" \
    --output filled_templates/transitive_only.jsonl

# By strategy name
bead templates filter-filled \
    --filled-templates filled_templates/all.jsonl \
    --strategy-name "exhaustive" \
    --output filled_templates/exhaustive_only.jsonl

# Multiple criteria (AND logic)
bead templates filter-filled \
    --filled-templates filled_templates/all.jsonl \
    --min-length 6 \
    --template-name "transitive" \
    --output filled_templates/filtered_multi.jsonl
```

## Merging Filled Templates

Combine multiple filled template files:

```bash
bead templates merge-filled \
    --input filled_templates/batch1.jsonl \
    --input filled_templates/batch2.jsonl \
    --input filled_templates/batch3.jsonl \
    --output filled_templates/merged.jsonl
```

With deduplication by ID:

```bash
bead templates merge-filled \
    --input filled_templates/batch1.jsonl \
    --input filled_templates/batch2.jsonl \
    --deduplicate \
    --output filled_templates/merged_unique.jsonl
```

The command reports duplicate count when deduplication is enabled.

## Export Formats

Export filled templates to alternative formats for analysis or external tools.

### Export to CSV

```bash
bead templates export-csv \
    --filled-templates filled_templates/all.jsonl \
    --output filled_templates/all.csv
```

Output includes columns:

- `id`: UUID
- `template_id`: source template UUID
- `template_name`: template name
- `rendered_text`: final text with slots filled
- `strategy_name`: filling strategy used
- `slot_count`: number of slots in template

### Export to JSON

```bash
bead templates export-json \
    --filled-templates filled_templates/all.jsonl \
    --output filled_templates/all.json

# Pretty-printed
bead templates export-json \
    --filled-templates filled_templates/all.jsonl \
    --pretty \
    --output filled_templates/all_pretty.json
```

Exports as JSON array (not JSONL). Use for compatibility with tools that don't support JSONL.

## Validation

Check filled templates for completeness:

```bash
bead templates validate-filled \
    --filled-templates filled_templates/all.jsonl
```

Validation checks:

- All slots filled (no empty slots)
- Rendered text nonempty
- Template references valid
- Slot filler UUIDs valid

## Statistics

View statistics on filled templates:

```bash
bead templates show-stats \
    --filled-templates filled_templates/all.jsonl
```

Output includes:

```
Filled templates: 1,245
Unique templates: 26
Strategies: exhaustive (980), random (265)
Average text length: 8.3 words
Slot fill distribution:
  verb: 45 unique items
  subj: 98 unique items
  obj: 102 unique items
```

## Workflow Example

Complete workflow from templates to filled templates:

```bash
# 1. Generate templates
bead resources generate-templates \
    --from-pattern "{subj} {verb} {obj}" \
    --output templates/transitive.jsonl

# 2. Estimate combinations
bead templates estimate-combinations \
    --template templates/transitive.jsonl \
    --lexicon lexicons/nouns.jsonl lexicons/verbs.jsonl

# 3. Fill with stratified sampling (large space)
bead templates fill \
    --template templates/transitive.jsonl \
    --lexicon lexicons/nouns.jsonl lexicons/verbs.jsonl \
    --strategy stratified \
    --stratify-by verb_class \
    --n-samples-per-stratum 20 \
    --output filled_templates/stratified.jsonl

# 4. Filter by length
bead templates filter-filled \
    --filled-templates filled_templates/stratified.jsonl \
    --min-length 4 \
    --max-length 10 \
    --output filled_templates/filtered.jsonl

# 5. Validate
bead templates validate-filled \
    --filled-templates filled_templates/filtered.jsonl

# 6. Export to CSV for inspection
bead templates export-csv \
    --filled-templates filled_templates/filtered.jsonl \
    --output filled_templates/filtered.csv
```

## Next Steps

Once templates are filled:

1. [Create experimental items](items.md) from filled templates
2. [Partition items into lists](lists.md) with constraints
3. [Deploy experiments](deployment.md) to jsPsych/JATOS

For complete API documentation, see [bead.templates API reference](../api/templates.md).
