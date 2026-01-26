# CLI Reference

## Overview

The bead CLI provides commands for all stages of the experimental pipeline, from resource creation through deployment and active learning. Commands are organized into logical groups corresponding to pipeline stages.

The CLI uses a hierarchical command structure:

```bash
uv run bead [GLOBAL_OPTIONS] COMMAND_GROUP COMMAND [OPTIONS] [ARGUMENTS]
```

All commands support `--help` for detailed usage information:

```bash
uv run bead --help                    # Show all command groups
uv run bead resources --help          # Show all resource commands
uv run bead resources create-lexicon --help  # Show specific command help
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
uv run bead --config-file my-config.yaml config show

# Use development profile
uv run bead --profile dev resources list-lexicons

# Verbose output for debugging
uv run bead --verbose templates fill template.jsonl lexicon.jsonl filled.jsonl

# Quiet mode for scripts
uv run bead --quiet lists partition items.jsonl lists/ --n-lists 5
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
uv run bead init my-experiment

# Initialize in current directory
uv run bead init

# Use development profile
uv run bead init my-experiment --profile dev

# Overwrite existing directory
uv run bead init my-experiment --force
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
uv run bead config show

# Show specific key
uv run bead config show --key paths.data_dir

# Show as JSON
uv run bead config show --format json

# Validate configuration file
uv run bead config validate

# Export with comments
uv run bead config export --output my-config.yaml --comments

# List available profiles
uv run bead config profiles

# Merge configurations
uv run bead config merge base.yaml overrides.yaml --output merged.yaml

# Create configuration templates
uv run bead config create-active-learning --output al_config.yaml
uv run bead config create-model --output model_config.yaml
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
uv run bead resources create-lexicon lexicon.jsonl --name verbs \
    --from-csv verbs.csv --language-code eng

# Create template
uv run bead resources create-template template.jsonl --name transitive \
    --template-string "{subject} {verb} {object}" \
    --slot subject:true --slot verb:true --slot object:false

# Generate templates from pattern
uv run bead resources generate-templates template.jsonl \
    --pattern "{subject} {verb} {object}" \
    --name simple_transitive

# List lexicons in directory
uv run bead resources list-lexicons --directory lexicons/

# Validate lexicon
uv run bead resources validate-lexicon lexicon.jsonl

# Import from VerbNet
uv run bead resources import-verbnet verbs.jsonl --classes motion-51.1

# Create constraint
uv run bead resources create-constraint --expression "item['pos'] == 'verb'" \
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
uv run bead templates fill template.jsonl lexicon.jsonl filled.jsonl \
    --strategy exhaustive

# Random sampling
uv run bead templates fill template.jsonl lexicon.jsonl filled.jsonl \
    --strategy random --max-combinations 100 --random-seed 42

# Stratified sampling
uv run bead templates fill template.jsonl lexicon.jsonl filled.jsonl \
    --strategy stratified --max-combinations 100 \
    --grouping-property pos --random-seed 42
```

**Analysis and Processing:**

```bash
# Estimate combinations before filling
uv run bead templates estimate-combinations template.jsonl lexicon.jsonl

# Show statistics
uv run bead templates show-stats filled.jsonl

# Filter by criteria
uv run bead templates filter-filled filled.jsonl filtered.jsonl \
    --min-length 10 --template-name active

# Merge multiple files
uv run bead templates merge-filled file1.jsonl file2.jsonl merged.jsonl \
    --deduplicate

# Export formats
uv run bead templates export-csv filled.jsonl filled.csv
uv run bead templates export-json filled.jsonl filled.json --pretty
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
uv run bead items construct \
    --item-template templates.jsonl \
    --filled-templates filled.jsonl \
    --output items.jsonl

# With constraints
uv run bead items construct \
    --item-template templates.jsonl \
    --filled-templates filled.jsonl \
    --constraints constraints.jsonl \
    --output items.jsonl

# With caching
uv run bead items construct \
    --item-template templates.jsonl \
    --filled-templates filled.jsonl \
    --output items.jsonl \
    --cache-dir .cache/models

# Create forced-choice items from texts
uv run bead items create-forced-choice-from-texts \
    --texts-file sentences.txt \
    --n-alternatives 2 \
    --output items.jsonl

# Create Likert items
uv run bead items create-ordinal-scale-from-texts \
    --texts-file sentences.txt \
    --scale-min 1 --scale-max 7 \
    --output items.jsonl

# Create NLI items
uv run bead items create-nli \
    --premise "All dogs bark" \
    --hypothesis "Some dogs bark" \
    --output items.jsonl
```

**Analysis:**

```bash
# List items
uv run bead items list --directory items/

# Validate items
uv run bead items validate items.jsonl

# Validate for specific task type
uv run bead items validate-for-task-type items.jsonl --task-type forced_choice

# Show statistics
uv run bead items show-stats items.jsonl

# Infer task type
uv run bead items infer-task-type items.jsonl

# Get task type requirements
uv run bead items get-task-requirements forced_choice
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
uv run bead lists partition items.jsonl lists/ --n-lists 5 \
    --strategy balanced --random-seed 42

# Random partitioning
uv run bead lists partition items.jsonl lists/ --n-lists 5 \
    --strategy random

# Stratified partitioning
uv run bead lists partition items.jsonl lists/ --n-lists 5 \
    --strategy stratified

# Dry run preview
uv run bead lists partition items.jsonl lists/ --n-lists 5 \
    --strategy balanced --dry-run
```

**Creating Constraints:**

```bash
# List constraints (per-list)
uv run bead lists create-uniqueness \
    --property-expression "item.metadata.verb_lemma" \
    -o constraints.jsonl

uv run bead lists create-balance \
    --property-expression "item.metadata.condition" \
    --tolerance 0.1 -o constraints.jsonl

# Batch constraints (across all lists)
uv run bead lists create-batch-coverage \
    --property-expression "item.metadata.template_id" \
    --target-values "0,1,2,3,4,5" \
    --min-coverage 1.0 -o constraints.jsonl

uv run bead lists create-batch-balance \
    --property-expression "item.metadata.verb_type" \
    --target-distribution "transitive=0.5,intransitive=0.5" \
    --tolerance 0.05 -o constraints.jsonl
```

**Analysis:**

```bash
# List available lists
uv run bead lists list --directory lists/

# Validate list
uv run bead lists validate lists/list_0.jsonl

# Show statistics
uv run bead lists show-stats lists/
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
uv run bead deployment generate lists/ items.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Acceptability Study" \
    --distribution-strategy balanced

# Quota-based distribution
uv run bead deployment generate lists/ items.jsonl experiment/ \
    --experiment-type forced_choice \
    --distribution-strategy quota_based \
    --distribution-config '{"participants_per_list": 25, "allow_overflow": false}'

# Stratified distribution
uv run bead deployment generate lists/ items.jsonl experiment/ \
    --experiment-type forced_choice \
    --distribution-strategy stratified \
    --distribution-config '{"factors": ["condition", "verb_type"]}'

# Debug mode (always use same list)
uv run bead deployment generate lists/ items.jsonl experiment/ \
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
uv run bead deployment ui customize experiment/ \
    --theme dark --primary-color "#1976D2"

# Generate custom CSS
uv run bead deployment ui generate-css experiment/css/custom.css \
    --theme dark --primary-color "#1976D2"
```

**Trial Configuration:**

```bash
# Configure rating scale
uv run bead deployment trials configure-rating \
    --min-value 1 --max-value 7 \
    --min-label "Completely unnatural" \
    --max-label "Completely natural" \
    --output rating_config.json

# Configure choice trials
uv run bead deployment trials configure-choice \
    --button-html '<button class="choice-btn">%choice%</button>' \
    --output choice_config.json
```

**JATOS Integration:**

```bash
# Export to .jzip
uv run bead deployment export-jatos experiment/ study.jzip \
    --title "My Study" \
    --description "Description text"

# Upload to JATOS server
uv run bead deployment upload-jatos study.jzip \
    --jatos-url https://jatos.example.com \
    --api-token YOUR_API_TOKEN

# Validate experiment
uv run bead deployment validate experiment/
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
uv run bead models train-model \
    --task-type forced_choice \
    --items items.jsonl \
    --labels labels.jsonl \
    --output-dir models/fc_model/

# Train with random intercepts
uv run bead models train-model \
    --task-type ordinal_scale \
    --items items.jsonl \
    --labels labels.jsonl \
    --participant-ids participant_ids.txt \
    --mixed-effects-mode random_intercepts \
    --output-dir models/os_model/

# Make predictions
uv run bead models predict \
    --model-dir models/fc_model/ \
    --items test_items.jsonl \
    --output predictions.jsonl

# Predict probabilities
uv run bead models predict-proba \
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
uv run bead active-learning check-convergence \
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
uv run bead training collect-data results.jsonl \
    --jatos-url https://jatos.example.com \
    --api-token TOKEN --study-id 123

# Show data statistics
uv run bead training show-data-stats results.jsonl

# Compute agreement
uv run bead training compute-agreement \
    --annotations annotations.jsonl \
    --metric krippendorff_alpha

# Cross-validation
uv run bead training cross-validate \
    --items items.jsonl \
    --labels labels.jsonl \
    --model-config model_config.yaml \
    --k-folds 5

# Evaluate model
uv run bead training evaluate \
    --model-dir model_dir/ \
    --test-items test_items.jsonl \
    --test-labels test_labels.jsonl

# Generate learning curve
uv run bead training learning-curve \
    --items items.jsonl \
    --labels labels.jsonl \
    --model-config model_config.yaml \
    --output learning_curve.json
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
uv run bead simulate list-annotators

# List noise models
uv run bead simulate list-noise-models

# Create configuration
uv run bead simulate configure \
    --strategy lm_score \
    --noise-type temperature \
    --temperature 1.5 \
    --output simulation_config.yaml

# Run simulation
uv run bead simulate run \
    --items items.jsonl \
    --templates templates.jsonl \
    --annotator lm_score \
    --n-annotators 5 \
    --output results.jsonl

# Analyze results
uv run bead simulate analyze results.jsonl \
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
uv run bead workflow run --config bead.yaml

# Run specific stages
uv run bead workflow run --stages resources,templates,items

# Start from items stage
uv run bead workflow run --from-stage items

# Dry run to preview
uv run bead workflow run --dry-run

# Initialize from template
uv run bead workflow init acceptability-study

# Check status
uv run bead workflow status

# Resume interrupted workflow
uv run bead workflow resume

# Rollback to deployment stage
uv run bead workflow rollback deployment

# List available templates
uv run bead workflow list-templates
```

### completion

Shell completion setup.

**Commands:**

| Command | Description |
|---------|-------------|
| `bash` | Generate bash completion script |
| `zsh` | Generate zsh completion script |
| `fish` | Generate fish completion script |
| `install` | Auto-detect shell and install completion |

**Examples:**

```bash
# Generate bash completion
uv run bead completion bash > ~/.bash_completion.d/bead

# Generate zsh completion
uv run bead completion zsh > ~/.zsh/completion/_bead

# Generate fish completion
uv run bead completion fish > ~/.config/fish/completions/bead.fish

# Auto-install for current shell
uv run bead completion install
```

### shell

Interactive shell commands.

**Commands:**

| Command | Description |
|---------|-------------|
| `repl` | Start interactive REPL shell |

**Examples:**

```bash
# Start interactive shell
uv run bead shell repl
```

## Common Workflows

### Complete Pipeline (Stages 1-5)

Build experiment from scratch:

```bash
# Stage 1: Create resources
uv run bead resources create-lexicon lexicons/verbs.jsonl --name verbs \
    --from-csv data/verbs.csv --language-code eng

uv run bead resources create-template templates/transitive.jsonl \
    --name transitive \
    --template-string "{subject} {verb} {object}"

# Stage 2: Fill templates
uv run bead templates fill templates/transitive.jsonl \
    lexicons/verbs.jsonl \
    filled_templates/filled.jsonl \
    --strategy exhaustive

# Stage 3: Construct items
uv run bead items construct \
    --item-template item_templates/templates.jsonl \
    --filled-templates filled_templates/filled.jsonl \
    --output items/items.jsonl

# Stage 4: Partition into lists
uv run bead lists partition items/items.jsonl lists/ \
    --n-lists 5 --strategy balanced --random-seed 42

# Stage 5: Generate experiment
uv run bead deployment generate lists/ items/items.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Acceptability Study" \
    --distribution-strategy balanced
```

### Active Learning Workflow

Train model with convergence detection:

```bash
# Collect initial data
uv run bead training collect-data data/initial_labels.jsonl \
    --jatos-url https://jatos.example.com \
    --api-token TOKEN --study-id 123

# Train initial model
uv run bead models train-model \
    --task-type forced_choice \
    --items items/items.jsonl \
    --labels data/initial_labels.jsonl \
    --output-dir models/model_v1/

# Check convergence
uv run bead active-learning check-convergence \
    --predictions models/model_v1/predictions.jsonl \
    --human-labels data/initial_labels.jsonl \
    --metric krippendorff_alpha \
    --threshold 0.8

# Generate learning curve
uv run bead training learning-curve \
    --items items/items.jsonl \
    --labels data/all_labels.jsonl \
    --model-config model_config.yaml \
    --output learning_curve.json
```

### Simulation Testing

Test strategies before deployment:

```bash
# List available annotators
uv run bead simulate list-annotators

# Create simulation configuration
uv run bead simulate configure \
    --strategy lm_score \
    --noise-type temperature \
    --temperature 0.7 \
    --output sims/config.yaml

# Run simulation
uv run bead simulate run \
    --items items/items.jsonl \
    --templates templates/templates.jsonl \
    --annotator lm_score \
    --n-annotators 5 \
    --output sims/results.jsonl

# Analyze results
uv run bead simulate analyze sims/results.jsonl \
    --metrics agreement accuracy convergence \
    --output sims/evaluation.json
```

### JATOS Deployment

Deploy to JATOS server:

```bash
# Generate experiment
uv run bead deployment generate lists/ items/items.jsonl experiment/ \
    --experiment-type forced_choice \
    --title "Acceptability Study" \
    --description "Judge sentence acceptability" \
    --distribution-strategy balanced

# Customize UI
uv run bead deployment ui customize experiment/ \
    --theme dark --primary-color "#1976D2"

# Validate before export
uv run bead deployment validate experiment/

# Export to .jzip
uv run bead deployment export-jatos experiment/ acceptability_study.jzip \
    --title "Acceptability Study" \
    --description "Version 1.0"

# Upload to JATOS
uv run bead deployment upload-jatos acceptability_study.jzip \
    --jatos-url https://jatos.example.com \
    --api-token YOUR_API_TOKEN
```

### Configuration-Driven Workflow

Use YAML configuration for entire pipeline:

```bash
# Create project with config
uv run bead init my-experiment --profile prod

# Edit config file (bead.yaml)
# Specify all pipeline parameters

# Validate configuration
uv run bead config validate

# Run entire pipeline from config
uv run bead workflow run --config bead.yaml

# Check pipeline status
uv run bead workflow status

# Resume if interrupted
uv run bead workflow resume
```

## Tips and Best Practices

### Configuration Management

Use profiles for different environments:

```bash
# Development: verbose output, small samples
uv run bead --profile dev templates fill ... --max-combinations 10

# Production: optimized, full datasets
uv run bead --profile prod workflow run --config bead.yaml

# Show configuration with specific key
uv run bead config show --key paths.data_dir

# Export with comments for documentation
uv run bead config export --output full-config.yaml --comments
```

### Error Handling

Use validation commands before expensive operations:

```bash
# Validate inputs before filling
uv run bead resources validate-lexicon lexicon.jsonl
uv run bead resources validate-template template.jsonl

# Estimate before exhaustive filling
uv run bead templates estimate-combinations template.jsonl lexicon.jsonl

# Dry-run before partitioning
uv run bead lists partition items.jsonl lists/ --n-lists 5 --dry-run

# Validate before deployment
uv run bead deployment validate experiment/
```

### Performance Optimization

```bash
# Use caching for repeated operations
uv run bead items construct ... --cache-dir .cache/models

# Use random sampling for large spaces
uv run bead templates fill ... --strategy random --max-combinations 1000

# Preview with dry-run before expensive operations
uv run bead workflow run --dry-run
```

### Reproducibility

Always set random seeds:

```bash
# Template filling
uv run bead templates fill ... --strategy random --random-seed 42

# List partitioning
uv run bead lists partition ... --random-seed 42

# Model training
uv run bead models train-model ... --random-seed 42
```

### Debugging

Use verbose mode and dry-runs:

```bash
# Verbose output
uv run bead --verbose templates fill ...

# Dry-run preview
uv run bead lists partition ... --dry-run
uv run bead workflow run --dry-run

# Debug mode for deployment
uv run bead deployment generate ... --debug-mode --debug-list-index 0

# Show configuration for debugging
uv run bead config show --format json --no-redact
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
uv run bead workflow status

# Resume interrupted workflow
uv run bead workflow resume

# Rollback if needed
uv run bead workflow rollback deployment

# Run specific stages only
uv run bead workflow run --stages items,lists,deployment
```
