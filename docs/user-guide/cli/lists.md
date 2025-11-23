# Lists Module

The lists module handles Stage 4 of the pipeline: partitioning items into experimental lists with constraint satisfaction.

## List Constraints

List constraints apply to individual lists. Eight constraint types address common experimental design requirements.

### Uniqueness Constraint

Ensure property values appear at most once per list:

```bash
bead lists create-uniqueness \
    --property-expression "item.metadata.verb" \
    --priority 5 \
    --output constraints/unique_verbs.jsonl
```

Each list contains each verb at most once.

### Balance Constraint

Balance distribution of property values:

```bash
bead lists create-balance \
    --property-expression "item.metadata.condition" \
    --target-counts control=10,experimental=10 \
    --tolerance 0.1 \
    --priority 4 \
    --output constraints/balance_condition.jsonl
```

Tolerance allows ±10% deviation from target counts.

### Quantile Constraint

Ensure property values span quantile ranges:

```bash
bead lists create-quantile \
    --property-expression "item.metadata.word_length" \
    --n-quantiles 4 \
    --priority 3 \
    --output constraints/quantile_length.jsonl
```

Each list includes items from all four word length quartiles.

### Grouped Quantile Constraint

Apply quantile constraints within groups:

```bash
bead lists create-grouped-quantile \
    --property-expression "item.metadata.frequency" \
    --group-by-expression "item.metadata.condition" \
    --n-quantiles 3 \
    --priority 3 \
    --output constraints/grouped_quantile_freq.jsonl
```

Each condition has balanced frequency distribution across tertiles.

### Diversity Constraint

Require minimum unique values:

```bash
bead lists create-diversity \
    --property-expression "item.metadata.verb_class" \
    --min-unique 10 \
    --priority 3 \
    --output constraints/diversity_class.jsonl
```

Each list must include at least 10 different verb classes.

### Size Constraint

Constrain list size:

```bash
bead lists create-size \
    --min-size 40 \
    --max-size 60 \
    --priority 5 \
    --output constraints/size.jsonl
```

Lists contain between 40 and 60 items.

## Batch Constraints

Batch constraints apply across all lists collectively. Four constraint types ensure balanced coverage.

### Batch Coverage Constraint

Ensure all target values appear somewhere:

```bash
bead lists create-batch-coverage \
    --property-expression "item.metadata.template_id" \
    --target-values "0,1,2,3,4,5,6,7,8,9" \
    --min-coverage 1.0 \
    --priority 5 \
    --output constraints/batch_coverage_templates.jsonl
```

All 10 templates appear in at least one list.

### Batch Balance Constraint

Balance property values across batch:

```bash
bead lists create-batch-balance \
    --property-expression "item.metadata.condition" \
    --target-distribution control=0.5,experimental=0.5 \
    --tolerance 0.05 \
    --priority 4 \
    --output constraints/batch_balance_condition.jsonl
```

Across all lists, conditions appear in 50/50 ratio (±5%).

### Batch Diversity Constraint

Limit values per list:

```bash
bead lists create-batch-diversity \
    --property-expression "item.metadata.target_word" \
    --max-lists-per-value 3 \
    --priority 3 \
    --output constraints/batch_diversity_word.jsonl
```

Each target word appears in at most 3 lists.

### Batch Min Occurrence Constraint

Ensure minimum occurrences:

```bash
bead lists create-batch-min-occurrence \
    --property-expression "item.metadata.construction" \
    --min-occurrences 5 \
    --priority 4 \
    --output constraints/batch_min_occurrence.jsonl
```

Each construction appears at least 5 times across all lists.

## Partitioning

Divide items into experimental lists.

### Basic Partitioning

```bash
bead lists partition \
    --items items/all.jsonl \
    --n-lists 10 \
    --strategy balanced \
    --output lists/
```

The `balanced` strategy distributes items evenly across lists.

### With List Constraints

```bash
bead lists partition \
    --items items/all.jsonl \
    --n-lists 10 \
    --list-constraints constraints/list_constraints.jsonl \
    --strategy balanced \
    --output lists/
```

### With Batch Constraints

```bash
bead lists partition \
    --items items/all.jsonl \
    --n-lists 10 \
    --batch-constraints constraints/batch_constraints.jsonl \
    --strategy balanced \
    --output lists/
```

### With Both Constraint Types

```bash
bead lists partition \
    --items items/all.jsonl \
    --n-lists 10 \
    --list-constraints constraints/list_constraints.jsonl \
    --batch-constraints constraints/batch_constraints.jsonl \
    --strategy balanced \
    --max-iterations 10000 \
    --output lists/
```

The partitioner uses priority-weighted constraint satisfaction. Higher priority constraints are satisfied first.

## Partitioning Strategies

Three strategies balance different goals:

- **balanced**: even distribution across lists
- **random**: random assignment (respects constraints)
- **stratified**: balance strata defined by metadata

Example with stratified:

```bash
bead lists partition \
    --items items/all.jsonl \
    --n-lists 10 \
    --strategy stratified \
    --stratify-by condition \
    --output lists/
```

## Validation

Verify list file structure:

```bash
bead lists validate list_0.jsonl
```

Validates that the list file contains a valid ExperimentList with proper structure.

## Listing and Statistics

View list statistics:

```bash
# Show statistics
bead lists show-stats lists/
```

Output includes:

```
Experiment List Statistics
┌─────────────────────┬───────┐
│ Metric              │ Value │
├─────────────────────┼───────┤
│ Total Lists         │    10 │
│ Total Items         │   450 │
│                     │       │
│ Avg Items per List  │  45.0 │
│ Min Items per List  │    42 │
│ Max Items per List  │    48 │
└─────────────────────┴───────┘

Per-List Breakdown:
  list_0: 45 items
  list_1: 42 items
  list_2: 48 items
  ...
```

## Workflow Example

Complete workflow from items to lists:

```bash
# 1. Create list constraints
bead lists create-uniqueness \
    --property-expression "item.metadata.verb" \
    --priority 5 \
    --output constraints/unique_verbs.jsonl

bead lists create-balance \
    --property-expression "item.metadata.condition" \
    --target-counts control=20,experimental=20 \
    --tolerance 0.1 \
    --priority 4 \
    --output constraints/balance_condition.jsonl

# 2. Create batch constraint
bead lists create-batch-coverage \
    --property-expression "item.metadata.template_id" \
    --target-values "0,1,2,3,4,5" \
    --min-coverage 1.0 \
    --priority 5 \
    --output constraints/batch_coverage.jsonl

# 3. Partition with constraints
bead lists partition \
    --items items/2afc_pairs.jsonl \
    --n-lists 10 \
    --list-constraints constraints/unique_verbs.jsonl constraints/balance_condition.jsonl \
    --batch-constraints constraints/batch_coverage.jsonl \
    --strategy balanced \
    --output lists/

# 4. Validate a list file
bead lists validate lists/list_0.jsonl

# 5. View statistics
bead lists show-stats lists/
```

## Next Steps

Once lists are partitioned:

1. [Deploy to jsPsych/JATOS](deployment.md) with distribution strategies
2. [Train models](training.md) on collected responses
3. [Analyze results](training.md#evaluation) with convergence detection

For complete API documentation, see [bead.lists API reference](../api/lists.md).
