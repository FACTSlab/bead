# Deployment Module

The deployment module handles Stage 5 of the pipeline: generating jsPsych 8.x experiments for JATOS deployment.

## Batch Mode Architecture

Bead uses batch mode exclusively: all lists are packaged in a single experiment. The system distributes participants to lists via server-side logic using JATOS batch sessions.

Benefits:

- Single experiment URL for all participants
- Server tracks list usage and assignment
- No manual list rotation needed
- Real-time balancing across participants

## Distribution Strategies

Eight strategies control how participants are assigned to lists.

### Random Distribution

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Read and respond to each item." \
    --distribution-strategy random
```

Participants receive lists uniformly at random.

### Sequential Distribution

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy sequential
```

Round-robin assignment: participant 1 gets list 0, participant 2 gets list 1, ..., participant N+1 gets list 0.

### Balanced Distribution

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy balanced
```

Assigns participants to the least-used list, ensuring even distribution.

### Latin Square Distribution

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy latin_square \
    --distribution-config '{"balanced": true}'
```

Counterbalancing using Bradley's balanced Latin square algorithm.

### Stratified Distribution

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy stratified \
    --distribution-config '{"factors": ["condition", "verb_type"]}'
```

Balances assignment across factors in list metadata.

### Weighted Random Distribution

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy weighted_random \
    --distribution-config '{"weight_expression": "list_metadata.priority || 1.0", "normalize_weights": true}'
```

Non-uniform random assignment based on list metadata expressions.

### Quota-Based Distribution

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy quota_based \
    --distribution-config '{"participants_per_list": 25, "allow_overflow": false}'
```

Fixed quota per list. Raises error when quotas filled if `allow_overflow` is false.

### Metadata-Based Distribution

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy metadata_based \
    --distribution-config '{"filter_expression": "list_metadata.difficulty === \"hard\"", "rank_expression": "list_metadata.priority || 0", "rank_ascending": false}'
```

Filter and rank lists by metadata expressions.

### Debug Mode

For development, force assignment to a specific list:

```bash
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy balanced \
    --debug-mode \
    --debug-list-index 0
```

All participants receive list 0. Use only during development.

## Trial Configuration

Customize trial presentation with specialized configuration commands.

### Rating Scale Trials

```bash
bead deployment trials configure-rating trial_config_rating.json \
    --min-value 1 \
    --max-value 7 \
    --step 1 \
    --min-label "Strongly disagree" \
    --max-label "Strongly agree" \
    --show-numeric-labels \
    --required
```

### Choice Trials

```bash
bead deployment trials configure-choice trial_config_choice.json \
    --button-html "<button>%choice%</button>" \
    --enable-keyboard \
    --randomize-position
```

The `%choice%` placeholder is replaced with choice text.

## UI Customization

Apply Material Design themes and custom styling.

### Generate CSS

```bash
bead deployment ui generate-css experiment/css/material.css \
    --theme dark \
    --primary-color "#1976D2" \
    --secondary-color "#FF4081"
```

Themes: `light`, `dark`, `auto` (respects system preference).

### Customize Experiment

Apply theme to existing experiment:

```bash
bead deployment ui customize experiment/ \
    --theme dark \
    --primary-color "#1976D2" \
    --secondary-color "#FF4081"
```

## JATOS Integration

Export and upload experiments to JATOS servers.

### Export to JATOS

Package experiment as `.jzip`:

```bash
bead deployment export-jatos experiment/ argument_structure.jzip \
    --title "Argument Structure Study" \
    --description "Forced choice acceptability judgment task"
```

## Validation

Verify experiment structure before deployment.

### Basic Validation

```bash
bead deployment validate experiment/
```

### With Distribution Check

```bash
bead deployment validate experiment/ \
    --check-distribution
```

### With Trial Config Check

```bash
bead deployment validate experiment/ \
    --check-trials
```

### With Data Structure Check

```bash
bead deployment validate experiment/ \
    --check-data-structure
```

### Strict Mode (All Checks)

```bash
bead deployment validate experiment/ \
    --strict
```

Enables all validation checks.

## Generated File Structure

Batch mode experiments have this structure:

```
experiment/
├── index.html              # Entry point
├── js/
│   ├── experiment.js       # jsPsych 8.x trial logic
│   └── list_distributor.js # Server-side assignment
├── css/
│   └── experiment.css      # Material Design styles
└── data/
    ├── config.json         # Experiment configuration
    ├── lists.jsonl         # All lists (JSONL format)
    ├── items.jsonl         # All items (JSONL format)
    └── distribution.json   # Strategy configuration
```

The `list_distributor.js` file uses JATOS batch sessions for server-side state management, avoiding race conditions when multiple participants join simultaneously.

## Workflow Example

Complete deployment workflow:

```bash
# 1. Generate experiment with balanced strategy
bead deployment generate lists/ items/2afc_pairs.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Verb Argument Structure" \
    --instructions "Choose the more natural sentence." \
    --distribution-strategy balanced

# 2. Configure rating scale trials
bead deployment trials configure-rating experiment/trial_config.json \
    --min-value 1 \
    --max-value 7

# 3. Apply dark theme
bead deployment ui customize experiment/ \
    --theme dark \
    --primary-color "#1976D2"

# 4. Validate
bead deployment validate experiment/ \
    --strict

# 5. Export to JATOS
bead deployment export-jatos experiment/ verb_study.jzip \
    --title "Verb Argument Structure Study" \
    --description "Forced choice acceptability judgment"
```

## Next Steps

After deployment:

1. Collect responses via JATOS
2. [Train models](training.md) on collected data
3. [Evaluate convergence](training.md#convergence) to human agreement

For complete API documentation, see [bead.deployment API reference](../api/deployment.md).
