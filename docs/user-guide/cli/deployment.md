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
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Read and respond to each item." \
    --distribution-strategy random \
    --output experiment/
```

Participants receive lists uniformly at random.

### Sequential Distribution

```bash
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy sequential \
    --output experiment/
```

Round-robin assignment: participant 1 gets list 0, participant 2 gets list 1, ..., participant N+1 gets list 0.

### Balanced Distribution

```bash
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy balanced \
    --output experiment/
```

Assigns participants to the least-used list, ensuring even distribution.

### Latin Square Distribution

```bash
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy latin_square \
    --distribution-config '{"balanced": true}' \
    --output experiment/
```

Counterbalancing using Bradley's balanced Latin square algorithm.

### Stratified Distribution

```bash
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy stratified \
    --distribution-config '{"factors": ["condition", "verb_type"]}' \
    --output experiment/
```

Balances assignment across factors in list metadata.

### Weighted Random Distribution

```bash
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy weighted_random \
    --distribution-config '{"weight_expression": "list_metadata.priority || 1.0", "normalize_weights": true}' \
    --output experiment/
```

Non-uniform random assignment based on list metadata expressions.

### Quota-Based Distribution

```bash
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy quota_based \
    --distribution-config '{"participants_per_list": 25, "allow_overflow": false}' \
    --output experiment/
```

Fixed quota per list. Raises error when quotas filled if `allow_overflow` is false.

### Metadata-Based Distribution

```bash
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy metadata_based \
    --distribution-config '{"filter_expression": "list_metadata.difficulty === \"hard\"", "rank_expression": "list_metadata.priority || 0", "rank_ascending": false}' \
    --output experiment/
```

Filter and rank lists by metadata expressions.

### Debug Mode

For development, force assignment to a specific list:

```bash
bead deployment generate \
    --lists lists/ \
    --items items/all.jsonl \
    --experiment-type forced_choice \
    --title "Experiment Title" \
    --instructions "Instructions here." \
    --distribution-strategy balanced \
    --debug-mode \
    --debug-list-index 0 \
    --output experiment/
```

All participants receive list 0. Use only during development.

## Trial Configuration

Customize trial presentation with specialized configuration commands.

### Rating Scale Trials

```bash
bead deployment trials configure-rating \
    --scale-type likert \
    --min-value 1 \
    --max-value 7 \
    --step 1 \
    --labels "Strongly disagree,Disagree,Somewhat disagree,Neutral,Somewhat agree,Agree,Strongly agree" \
    --show-numeric-labels \
    --required \
    --output trial_config_rating.json
```

### Choice Trials

```bash
bead deployment trials configure-choice \
    --choice-type forced_choice \
    --button-html "<button>%choice%</button>" \
    --enable-keyboard \
    --keyboard-choices j,k,l \
    --randomize-position \
    --output trial_config_choice.json
```

The `%choice%` placeholder is replaced with choice text.

### Timing Configuration

```bash
bead deployment trials configure-timing \
    --timing-type rsvp \
    --duration 500 \
    --isi 100 \
    --mask-char "#" \
    --cumulative \
    --output trial_config_timing.json
```

RSVP (Rapid Serial Visual Presentation) or self-paced reading parameters.

### Display Configuration

View trial configurations:

```bash
bead deployment trials show-config \
    --config trial_config_rating.json \
    --config trial_config_choice.json
```

## UI Customization

Apply Material Design themes and custom styling.

### Generate CSS

```bash
bead deployment ui generate-css \
    --theme dark \
    --primary-color "#1976D2" \
    --secondary-color "#FF4081" \
    --output experiment/css/material.css
```

Themes: `light`, `dark`, `auto` (respects system preference).

### Customize Experiment

Apply theme to existing experiment:

```bash
bead deployment ui customize \
    --experiment experiment/ \
    --theme dark \
    --primary-color "#1976D2" \
    --secondary-color "#FF4081"
```

With custom CSS file:

```bash
bead deployment ui customize \
    --experiment experiment/ \
    --theme dark \
    --css-file custom_styles.css
```

## JATOS Integration

Export and upload experiments to JATOS servers.

### Export to JATOS

Package experiment as `.jzip`:

```bash
bead deployment export-jatos \
    --experiment experiment/ \
    --study-title "Argument Structure Study" \
    --output argument_structure.jzip
```

### Upload to JATOS

Upload directly to server:

```bash
bead deployment upload-jatos \
    --file argument_structure.jzip \
    --server https://jatos.example.com \
    --username researcher \
    --password yourpassword
```

The server URL should point to your JATOS installation. Authentication uses JATOS credentials.

## Validation

Verify experiment structure before deployment.

### Basic Validation

```bash
bead deployment validate \
    --experiment experiment/
```

### With Distribution Check

```bash
bead deployment validate \
    --experiment experiment/ \
    --check-distribution
```

### With Trial Config Check

```bash
bead deployment validate \
    --experiment experiment/ \
    --check-trials
```

### With Data Structure Check

```bash
bead deployment validate \
    --experiment experiment/ \
    --check-data-structure
```

### Strict Mode (All Checks)

```bash
bead deployment validate \
    --experiment experiment/ \
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
bead deployment generate \
    --lists lists/ \
    --items items/2afc_pairs.jsonl \
    --experiment-type forced_choice \
    --title "Verb Argument Structure" \
    --instructions "Choose the more natural sentence." \
    --distribution-strategy balanced \
    --output experiment/

# 2. Configure rating scale trials
bead deployment trials configure-rating \
    --scale-type likert \
    --min-value 1 \
    --max-value 7 \
    --output experiment/trial_config.json

# 3. Apply dark theme
bead deployment ui customize \
    --experiment experiment/ \
    --theme dark \
    --primary-color "#1976D2"

# 4. Validate
bead deployment validate \
    --experiment experiment/ \
    --strict

# 5. Export to JATOS
bead deployment export-jatos \
    --experiment experiment/ \
    --study-title "Verb Argument Structure Study" \
    --output verb_study.jzip

# 6. Upload to server
bead deployment upload-jatos \
    --file verb_study.jzip \
    --server https://jatos.example.com \
    --username researcher \
    --password yourpassword
```

## Next Steps

After deployment:

1. Collect responses via JATOS
2. [Train models](training.md) on collected data
3. [Evaluate convergence](training.md#convergence) to human agreement

For complete API documentation, see [bead.deployment API reference](../api/deployment.md).
