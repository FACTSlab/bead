# CLI Reference

## Overview

The bead CLI provides commands for all stages of the experimental pipeline, from resource creation through deployment and active learning. Commands are organized into logical groups corresponding to pipeline stages.

The CLI uses a hierarchical command structure:

```bash
bead [GLOBAL_OPTIONS] COMMAND_GROUP COMMAND [OPTIONS] [ARGUMENTS]
```

All commands support `--help` for detailed usage information:

```bash
bead --help                    # Show all command groups
bead resources --help          # Show all resource commands
bead resources create-lexicon --help  # Show specific command help
```

## Global Options

These options apply to all bead commands:

| Option | Description |
|--------|-------------|
| `--config-file PATH`, `-c PATH` | Path to configuration file (overrides profile defaults) |
| `--profile NAME`, `-p NAME` | Configuration profile: default, dev, prod, test |
| `--verbose`, `-v` | Enable verbose output with detailed logging |
| `--quiet`, `-q` | Suppress all output except errors |
| `--version` | Show bead version and exit |
| `--help` | Show help message and exit |

**Examples:**

```bash
# Use custom config file
bead --config-file my-config.yaml config show

# Use development profile
bead --profile dev resources list-lexicons

# Verbose output for debugging
bead --verbose templates fill template.jsonl lexicon.jsonl filled.jsonl

# Quiet mode for scripts
bead --quiet lists partition items.jsonl lists/ --n-lists 5
```

## Command Groups

### init

Initialize a new bead project with directory structure and default configuration.

**Commands:**

| Command | Description |
|---------|-------------|
| `init [PROJECT_NAME]` | Create new project directory with scaffolding |

**Options:**

- `--profile NAME`: Initialize with specific profile (default, dev, prod, test)
- `--force`: Overwrite existing directory

**Examples:**

```bash
# Create new project
bead init my-experiment

# Initialize in current directory
bead init

# Use development profile
bead init my-experiment --profile dev

# Overwrite existing directory
bead init my-experiment --force
```

**Generated Structure:**

```
my-experiment/
├── bead.yaml          # Configuration file
├── .gitignore         # Git ignores
├── lexicons/          # Lexical resources
├── templates/         # Template definitions
├── filled_templates/  # Generated filled templates
├── items/             # Generated items
├── lists/             # Experiment lists
├── experiments/       # Generated experiments
├── data/              # Collected data
└── models/            # Trained models
```

### config

Configuration management commands.

**Commands:**

| Command | Description |
|---------|-------------|
| `show` | Display current configuration with optional filtering |
| `validate` | Validate configuration file structure |
| `export` | Export merged configuration to YAML |
| `profiles` | List available configuration profiles |
| `merge` | Merge two configuration files |
| `create-active-learning` | Create active learning configuration template |
| `create-model` | Create model configuration template |
| `create-simulation` | Create simulation configuration template |

**Examples:**

```bash
# Show current configuration
bead config show

# Show specific key
bead config show --key paths.data_dir

# Show as JSON
bead config show --format json

# Validate configuration file
bead config validate

# Export with comments
bead config export --output my-config.yaml --comments

# List available profiles
bead config profiles

# Merge configurations
bead config merge base.yaml overrides.yaml --output merged.yaml

# Create configuration templates
bead config create-active-learning --output al_config.yaml
bead config create-model --output model_config.yaml
```

### resources (Stage 1)

Lexicon and template management commands.

**Commands:**

| Command | Description |
|---------|-------------|
| `create-lexicon` | Create lexicon from CSV or JSON |
| `create-template` | Create template with slots |
| `generate-templates` | Generate templates from pattern |
| `generate-template-variants` | Generate systematic template variations |
| `list-lexicons` | List available lexicons |
| `list-templates` | List available templates |
| `validate-lexicon` | Validate lexicon file |
| `validate-template` | Validate template file |
| `import-verbnet` | Import from VerbNet |
| `import-unimorph` | Import from UniMorph |
| `import-propbank` | Import from PropBank |
| `import-framenet` | Import from FrameNet |
| `create-constraint` | Create constraint expression |

**Common Examples:**

```bash
# Create lexicon from CSV
bead resources create-lexicon lexicon.jsonl --name verbs \
    --from-csv verbs.csv --language-code eng

# Create template
bead resources create-template template.jsonl --name transitive \
    --template-string "{subject} {verb} {object}" \
    --slot subject:true --slot verb:true --slot object:false

# Generate templates from pattern
bead resources generate-templates template.jsonl \
    --pattern "{subject} {verb} {object}" \
    --name simple_transitive

# List lexicons in directory
bead resources list-lexicons --directory lexicons/

# Validate lexicon
bead resources validate-lexicon lexicon.jsonl

# Import from VerbNet
bead resources import-verbnet verbs.jsonl --classes motion-51.1

# Create constraint
bead resources create-constraint --expression "item['pos'] == 'verb'" \
    --output constraints.jsonl
```

### templates (Stage 2)

Template filling commands using various strategies.

**Commands:**

| Command | Description |
|---------|-------------|
| `fill` | Fill templates with lexical items |
| `estimate-combinations` | Estimate total combinations |
| `list-filled` | List filled template files |
| `validate-filled` | Validate filled templates |
| `show-stats` | Show filling statistics |
| `filter-filled` | Filter filled templates by criteria |
| `merge-filled` | Merge multiple filled files |
| `export-csv` | Export to CSV format |
| `export-json` | Export to JSON array |
| `sample-combinations` | Sample with stratification |

**Filling Strategies:**

```bash
# Exhaustive filling (all combinations)
bead templates fill template.jsonl lexicon.jsonl filled.jsonl \
    --strategy exhaustive

# Random sampling
bead templates fill template.jsonl lexicon.jsonl filled.jsonl \
    --strategy random --max-combinations 100 --random-seed 42

# Stratified sampling
bead templates fill template.jsonl lexicon.jsonl filled.jsonl \
    --strategy stratified --max-combinations 100 \
    --grouping-property pos --random-seed 42
```

**Analysis and Processing:**

```bash
# Estimate combinations before filling
bead templates estimate-combinations template.jsonl lexicon.jsonl

# Show statistics
bead templates show-stats filled.jsonl

# Filter by criteria
bead templates filter-filled filled.jsonl filtered.jsonl \
    --min-length 10 --template-name active

# Merge multiple files
bead templates merge-filled file1.jsonl file2.jsonl merged.jsonl \
    --deduplicate

# Export formats
bead templates export-csv filled.jsonl filled.csv
bead templates export-json filled.jsonl filled.json --pretty
```

### items (Stage 3)

Item construction commands with task-type-specific creation functions.

**Commands:**

| Command | Description |
|---------|-------------|
| `construct` | Construct items from filled templates |
| `create-forced-choice` | Create forced-choice item |
| `create-forced-choice-from-texts` | Create forced-choice items from text file |
| `create-nli` | Create NLI item |
| `create-likert-7` | Create 7-point Likert item |
| `create-ordinal-scale-from-texts` | Create ordinal scale items from texts |
| `create-binary-from-texts` | Create binary judgment items from texts |
| `create-categorical` | Create categorical item |
| `create-multi-select-from-texts` | Create multi-select items from texts |
| `create-magnitude-from-texts` | Create magnitude estimation items |
| `create-free-text-from-texts` | Create free text response items |
| `create-simple-cloze` | Create simple cloze item |
| `list` | List item files |
| `validate` | Validate items file |
| `validate-for-task-type` | Validate items for specific task type |
| `show-stats` | Show item statistics |
| `infer-task-type` | Infer task type from item structure |
| `get-task-requirements` | Get requirements for task type |

**Construction Examples:**

```bash
# Basic item construction from templates
bead items construct \
    --item-template templates.jsonl \
    --filled-templates filled.jsonl \
    --output items.jsonl

# With constraints
bead items construct \
    --item-template templates.jsonl \
    --filled-templates filled.jsonl \
    --constraints constraints.jsonl \
    --output items.jsonl

# With caching
bead items construct \
    --item-template templates.jsonl \
    --filled-templates filled.jsonl \
    --output items.jsonl \
    --cache-dir .cache/models

# Create forced-choice items from texts
bead items create-forced-choice-from-texts texts.txt items.jsonl \
    --n-alternatives 2 --group-by line

# Create Likert items
bead items create-ordinal-scale-from-texts sentences.txt items.jsonl \
    --scale-min 1 --scale-max 7

# Create NLI items
bead items create-nli items.jsonl \
    --premise "All dogs bark" \
    --hypothesis "Some dogs bark"
```

**Analysis:**

```bash
# List items
bead items list --directory items/

# Validate items
bead items validate items.jsonl

# Validate for specific task type
bead items validate-for-task-type items.jsonl --task-type forced_choice

# Show statistics
bead items show-stats items.jsonl

# Infer task type
bead items infer-task-type items.jsonl

# Get task type requirements
bead items get-task-requirements forced_choice
```

### lists (Stage 4)

List partitioning commands with constraint creation.

**Commands:**

| Command | Description |
|---------|-------------|
| `partition` | Partition items into lists |
| `list` | List available experiment lists |
| `validate` | Validate list file |
| `show-stats` | Show list statistics |
| `create-uniqueness` | Create uniqueness constraint |
| `create-balance` | Create balance constraint |
| `create-quantile` | Create quantile constraint |
| `create-grouped-quantile` | Create grouped quantile constraint |
| `create-diversity` | Create diversity constraint |
| `create-size` | Create size constraint |
| `create-batch-coverage` | Create batch coverage constraint |
| `create-batch-balance` | Create batch balance constraint |
| `create-batch-diversity` | Create batch diversity constraint |
| `create-batch-min-occurrence` | Create batch min occurrence constraint |

**Partitioning:**

```bash
# Balanced partitioning
bead lists partition items.jsonl lists/ --n-lists 5 \
    --strategy balanced --random-seed 42

# Random partitioning
bead lists partition items.jsonl lists/ --n-lists 5 \
    --strategy random

# Stratified partitioning
bead lists partition items.jsonl lists/ --n-lists 5 \
    --strategy stratified

# Dry run preview
bead lists partition items.jsonl lists/ --n-lists 5 \
    --strategy balanced --dry-run
```

**Creating Constraints:**

```bash
# List constraints (per-list)
bead lists create-uniqueness --property "verb_lemma" \
    --output constraints.jsonl

bead lists create-balance --property "condition" \
    --tolerance 0.1 --output constraints.jsonl

# Batch constraints (across all lists)
bead lists create-batch-coverage --property "template_id" \
    --min-coverage 1.0 --output constraints.jsonl

bead lists create-batch-balance --property "verb_type" \
    --tolerance 0.15 --output constraints.jsonl
```

**Analysis:**

```bash
# List available lists
bead lists list --directory lists/

# Validate list
bead lists validate lists/list_0.jsonl

# Show statistics
bead lists show-stats lists/
```

### deployment (Stage 5)

Experiment generation and deployment commands for jsPsych/JATOS.

**Commands:**

| Command | Description |
|---------|-------------|
| `generate` | Generate jsPsych experiment |
| `export-jatos` | Export to JATOS .jzip format |
| `upload-jatos` | Upload to JATOS server |
| `validate` | Validate experiment directory |
| `ui` | UI customization subgroup |
| `trials` | Trial configuration subgroup |

**UI Subgroup Commands:**

- `ui customize` - Apply UI customization to experiment
- `ui generate-css` - Generate Material Design CSS

**Trials Subgroup Commands:**

- `trials configure-choice` - Configure choice trial parameters
- `trials configure-rating` - Configure rating scale parameters
- `trials configure-timing` - Configure timing parameters
- `trials show-config` - Display trial configuration

**Generation:**

```bash
# Basic generation with balanced distribution
bead deployment generate lists/ items.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Acceptability Study" \
    --distribution-strategy balanced

# Quota-based distribution
bead deployment generate lists/ items.jsonl experiment/ \
    --experiment-type forced_choice \
    --distribution-strategy quota_based \
    --distribution-config '{"participants_per_list": 25, "allow_overflow": false}'

# Stratified distribution
bead deployment generate lists/ items.jsonl experiment/ \
    --experiment-type forced_choice \
    --distribution-strategy stratified \
    --distribution-config '{"factors": ["condition", "verb_type"]}'

# Debug mode (always use same list)
bead deployment generate lists/ items.jsonl experiment/ \
    --experiment-type forced_choice \
    --distribution-strategy balanced \
    --debug-mode --debug-list-index 0
```

**Distribution Strategies:**

1. **random**: Random list selection
2. **sequential**: Round-robin assignment
3. **balanced**: Assign to least-used list
4. **quota_based**: Fixed quota per list
5. **latin_square**: Counterbalancing with Bradley's algorithm
6. **weighted_random**: Non-uniform probabilities
7. **stratified**: Balance across metadata factors
8. **metadata_based**: Filter and rank by metadata

**UI Customization:**

```bash
# Apply UI theme
bead deployment ui customize experiment/ \
    --theme dark --primary-color "#1976D2"

# Generate custom CSS
bead deployment ui generate-css experiment/css/custom.css \
    --theme dark --primary-color "#1976D2"
```

**Trial Configuration:**

```bash
# Configure rating scale
bead deployment trials configure-rating \
    --min-value 1 --max-value 7 \
    --min-label "Completely unnatural" \
    --max-label "Completely natural" \
    --output rating_config.json

# Configure choice trials
bead deployment trials configure-choice \
    --button-html '<button class="choice-btn">%choice%</button>' \
    --output choice_config.json
```

**JATOS Integration:**

```bash
# Export to .jzip
bead deployment export-jatos experiment/ study.jzip \
    --title "My Study" \
    --description "Description text"

# Upload to JATOS server
bead deployment upload-jatos study.jzip \
    --jatos-url https://jatos.example.com \
    --api-token YOUR_API_TOKEN

# Validate experiment
bead deployment validate experiment/
```

### models

GLMM model training commands for all 8 task types.

**Commands:**

| Command | Description |
|---------|-------------|
| `train-model` | Train GLMM model for judgment prediction |
| `predict` | Make predictions with trained model |
| `predict-proba` | Predict class probabilities |

**Mixed Effects Modes:**

- `fixed`: Fixed effects only (no participant variability)
- `random_intercepts`: Participant-specific biases
- `random_slopes`: Participant-specific model parameters

**Examples:**

```bash
# Train forced choice model with fixed effects
bead models train-model \
    --task-type forced_choice \
    --items items.jsonl \
    --labels labels.jsonl \
    --output-dir models/fc_model/

# Train with random intercepts
bead models train-model \
    --task-type ordinal_scale \
    --items items.jsonl \
    --labels labels.jsonl \
    --participant-ids participant_ids.txt \
    --mixed-effects-mode random_intercepts \
    --output-dir models/os_model/

# Make predictions
bead models predict \
    --model-dir models/fc_model/ \
    --items test_items.jsonl \
    --output predictions.jsonl

# Predict probabilities
bead models predict-proba \
    --model-dir models/fc_model/ \
    --items test_items.jsonl \
    --output probabilities.jsonl
```

### active-learning (Stage 6)

Active learning commands for convergence detection.

**Note**: This module is **PARTIALLY IMPLEMENTED**. Currently only convergence checking is available.

**Commands:**

| Command | Description |
|---------|-------------|
| `check-convergence` | Check if model converged to human agreement |

**Deferred Commands** (awaiting implementation):

- `select-items` - Requires model.predict_proba implementation
- `run` - Requires data collection infrastructure
- `monitor-convergence` - Requires checkpoint loading

**Examples:**

```bash
# Check convergence
bead active-learning check-convergence \
    --predictions predictions.jsonl \
    --human-labels labels.jsonl \
    --metric krippendorff_alpha \
    --threshold 0.85
```

### training

Data collection and model evaluation commands.

**Commands:**

| Command | Description |
|---------|-------------|
| `collect-data` | Collect judgment data from JATOS |
| `show-data-stats` | Show statistics about collected data |
| `compute-agreement` | Compute inter-annotator agreement |
| `cross-validate` | Perform K-fold cross-validation |
| `evaluate` | Evaluate trained model on test set |
| `learning-curve` | Generate learning curve |

**Examples:**

```bash
# Collect data from JATOS
bead training collect-data results.jsonl \
    --jatos-url https://jatos.example.com \
    --api-token TOKEN --study-id 123

# Show data statistics
bead training show-data-stats results.jsonl

# Compute agreement
bead training compute-agreement results.jsonl \
    --metric krippendorff_alpha

# Cross-validation
bead training cross-validate items.jsonl labels.jsonl \
    --n-folds 5 --task-type forced_choice

# Evaluate model
bead training evaluate model_dir/ test_data.jsonl

# Generate learning curve
bead training learning-curve items.jsonl labels.jsonl \
    --output learning_curve.png
```

### simulate

Simulation framework for testing active learning strategies.

**Commands:**

| Command | Description |
|---------|-------------|
| `run` | Run simulation with configured annotators |
| `configure` | Create simulation configuration file |
| `analyze` | Analyze simulation results |
| `list-annotators` | List available annotator types |
| `list-noise-models` | List available noise models |

**Annotator Types:**

- `oracle` - Perfect annotations
- `random` - Random responses
- `lm_score` - Language model based scoring
- `distance` - Distance-based similarity

**Examples:**

```bash
# List available annotators
bead simulate list-annotators

# List noise models
bead simulate list-noise-models

# Create configuration
bead simulate configure \
    --strategy lm_score \
    --noise-type temperature \
    --temperature 1.5 \
    --output simulation_config.yaml

# Run simulation
bead simulate run \
    --items items.jsonl \
    --templates templates.jsonl \
    --annotator lm_score \
    --n-annotators 5 \
    --output results.jsonl

# Analyze results
bead simulate analyze results.jsonl \
    --metrics agreement accuracy convergence \
    --output analysis.json
```

### workflow

High-level workflow commands for complete pipeline execution.

**Commands:**

| Command | Description |
|---------|-------------|
| `run` | Run complete pipeline workflow |
| `init` | Initialize new project from template |
| `status` | Show current workflow status |
| `resume` | Resume interrupted workflow |
| `rollback` | Rollback to previous stage |
| `list-templates` | List available workflow templates |

**Pipeline Stages:**

1. resources - Create lexicons and templates
2. templates - Fill templates with lexicon items
3. items - Construct experimental items
4. lists - Partition items into experiment lists
5. deployment - Generate jsPsych experiments
6. training - Train models with active learning (optional)

**Examples:**

```bash
# Run all stages
bead workflow run --config bead.yaml

# Run specific stages
bead workflow run --stages resources,templates,items

# Start from items stage
bead workflow run --from-stage items

# Dry run to preview
bead workflow run --dry-run

# Initialize from template
bead workflow init acceptability-study

# Check status
bead workflow status

# Resume interrupted workflow
bead workflow resume

# Rollback to deployment stage
bead workflow rollback deployment

# List available templates
bead workflow list-templates
```

### completion

Shell completion setup.

**Commands:**

| Command | Description |
|---------|-------------|
| `install` | Install shell completion |
| `uninstall` | Remove shell completion |
| `show` | Show completion script |

**Examples:**

```bash
# Install for bash
bead completion install --shell bash

# Install for zsh
bead completion install --shell zsh

# Show completion script
bead completion show --shell bash
```

## Common Workflows

### Complete Pipeline (Stages 1-5)

Build experiment from scratch:

```bash
# Stage 1: Create resources
bead resources create-lexicon lexicons/verbs.jsonl --name verbs \
    --from-csv data/verbs.csv --language-code eng

bead resources create-template templates/transitive.jsonl \
    --name transitive \
    --template-string "{subject} {verb} {object}"

# Stage 2: Fill templates
bead templates fill templates/transitive.jsonl \
    lexicons/verbs.jsonl \
    filled_templates/filled.jsonl \
    --strategy exhaustive

# Stage 3: Construct items
bead items construct \
    --item-template item_templates/templates.jsonl \
    --filled-templates filled_templates/filled.jsonl \
    --output items/items.jsonl

# Stage 4: Partition into lists
bead lists partition items/items.jsonl lists/ \
    --n-lists 5 --strategy balanced --random-seed 42

# Stage 5: Generate experiment
bead deployment generate lists/ items/items.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Acceptability Study" \
    --distribution-strategy balanced
```

### Active Learning Workflow

Train model with convergence detection:

```bash
# Collect initial data
bead training collect-data data/initial_labels.jsonl \
    --jatos-url https://jatos.example.com \
    --api-token TOKEN --study-id 123

# Train initial model
bead models train-model \
    --task-type forced_choice \
    --items items/items.jsonl \
    --labels data/initial_labels.jsonl \
    --output-dir models/model_v1/

# Check convergence
bead active-learning check-convergence \
    --predictions models/model_v1/predictions.jsonl \
    --human-labels data/initial_labels.jsonl \
    --metric krippendorff_alpha \
    --threshold 0.8

# Generate learning curve
bead training learning-curve items/items.jsonl data/all_labels.jsonl \
    --output plots/learning_curve.png
```

### Simulation Testing

Test strategies before deployment:

```bash
# List available annotators
bead simulate list-annotators

# Create simulation configuration
bead simulate configure \
    --strategy lm_score \
    --noise-type temperature \
    --temperature 0.7 \
    --output sims/config.yaml

# Run simulation
bead simulate run \
    --items items/items.jsonl \
    --templates templates/templates.jsonl \
    --annotator lm_score \
    --n-annotators 5 \
    --output sims/results.jsonl

# Analyze results
bead simulate analyze sims/results.jsonl \
    --metrics agreement accuracy convergence \
    --output sims/evaluation.json
```

### JATOS Deployment

Deploy to JATOS server:

```bash
# Generate experiment
bead deployment generate lists/ items/items.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Acceptability Study" \
    --description "Judge sentence acceptability" \
    --distribution-strategy balanced

# Customize UI
bead deployment ui customize experiment/ \
    --theme dark --primary-color "#1976D2"

# Validate before export
bead deployment validate experiment/

# Export to .jzip
bead deployment export-jatos experiment/ acceptability_study.jzip \
    --title "Acceptability Study" \
    --description "Version 1.0"

# Upload to JATOS
bead deployment upload-jatos acceptability_study.jzip \
    --jatos-url https://jatos.example.com \
    --api-token YOUR_API_TOKEN
```

### Configuration-Driven Workflow

Use YAML configuration for entire pipeline:

```bash
# Create project with config
bead init my-experiment --profile prod

# Edit config file (bead.yaml)
# Specify all pipeline parameters

# Validate configuration
bead config validate

# Run entire pipeline from config
bead workflow run --config bead.yaml

# Check pipeline status
bead workflow status

# Resume if interrupted
bead workflow resume
```

## Tips and Best Practices

### Configuration Management

Use profiles for different environments:

```bash
# Development: verbose output, small samples
bead --profile dev templates fill ... --max-combinations 10

# Production: optimized, full datasets
bead --profile prod workflow run --config bead.yaml

# Show configuration with specific key
bead config show --key paths.data_dir

# Export with comments for documentation
bead config export --output full-config.yaml --comments
```

### Error Handling

Use validation commands before expensive operations:

```bash
# Validate inputs before filling
bead resources validate-lexicon lexicon.jsonl
bead resources validate-template template.jsonl

# Estimate before exhaustive filling
bead templates estimate-combinations template.jsonl lexicon.jsonl

# Dry-run before partitioning
bead lists partition items.jsonl lists/ --n-lists 5 --dry-run

# Validate before deployment
bead deployment validate experiment/
```

### Performance Optimization

```bash
# Use caching for repeated operations
bead items construct ... --cache-dir .cache/models

# Use random sampling for large spaces
bead templates fill ... --strategy random --max-combinations 1000

# Preview with dry-run before expensive operations
bead workflow run --dry-run
```

### Reproducibility

Always set random seeds:

```bash
# Template filling
bead templates fill ... --strategy random --random-seed 42

# List partitioning
bead lists partition ... --random-seed 42

# Model training
bead models train-model ... --random-seed 42
```

### Debugging

Use verbose mode and dry-runs:

```bash
# Verbose output
bead --verbose templates fill ...

# Dry-run preview
bead lists partition ... --dry-run
bead workflow run --dry-run

# Debug mode for deployment
bead deployment generate ... --debug-mode --debug-list-index 0

# Show configuration for debugging
bead config show --format json --no-redact
```

### File Organization

Follow recommended directory structure:

```
project/
├── bead.yaml           # Configuration
├── lexicons/           # Stage 1 outputs
├── templates/          # Stage 1 outputs
├── filled_templates/   # Stage 2 outputs
├── items/              # Stage 3 outputs
├── lists/              # Stage 4 outputs
├── experiments/        # Stage 5 outputs
├── data/               # Collected responses
├── models/             # Trained models (Stage 6)
└── .cache/             # Model output cache
```

### Workflow Management

```bash
# Check status before resuming
bead workflow status

# Resume interrupted workflow
bead workflow resume

# Rollback if needed
bead workflow rollback deployment

# Run specific stages only
bead workflow run --stages items,lists,deployment
```
