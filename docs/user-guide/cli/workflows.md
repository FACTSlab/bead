# Complete CLI Workflows

This guide demonstrates complete end-to-end workflows using the bead CLI, based on the `gallery/eng/argument_structure/` example project.

## Overview

The bead pipeline consists of 6 stages:

| Stage | Purpose | CLI Command Group |
|-------|---------|-------------------|
| 1 | Create lexicons and templates | `uv run bead resources` |
| 2 | Fill templates with items | `uv run bead templates` |
| 3 | Construct experimental items | `uv run bead items` |
| 4 | Partition into experiment lists | `uv run bead lists` |
| 5 | Generate jsPsych/JATOS experiments | `uv run bead deployment` |
| 6 | Collect data and train models | `uv run bead training` |

## Example: Argument Structure Experiment

This example creates a complete argument structure acceptability judgment experiment.

### Project Structure

```
argument_structure/
├── config.yaml              # Main configuration
├── lexicons/               # Stage 1 output
├── templates/              # Stage 1 output
├── filled_templates/       # Stage 2 output
├── items/                  # Stage 3 output
├── lists/                  # Stage 4 output
└── deployment/            # Stage 5 output
```

### Stage 1: Create Resources

Import verbs from VerbNet:

```bash
uv run bead resources import-verbnet \
  --verb-class "break-45.1" \
  --limit 20 \
  --output lexicons/verbnet_verbs.jsonl
```

Create custom lexicon from CSV:

```bash
uv run bead resources create-lexicon lexicons/bleached_nouns.jsonl \
  --name bleached_nouns \
  --from-csv resources/bleached_nouns.csv \
  --language-code eng
```

Generate templates from pattern:

```bash
uv run bead resources generate-templates templates/transitive.jsonl \
  --pattern "{det} {noun} {verb} {det2} {noun2}" \
  --name transitive \
  --language-code eng \
  --description "Basic transitive frame"
```

### Stage 2: Fill Templates

Fill templates with random strategy:

```bash
uv run bead templates fill templates/transitive.jsonl lexicons/verbnet_verbs.jsonl lexicons/bleached_nouns.jsonl filled_templates/all_combinations.jsonl \
  --strategy random \
  --max-combinations 100 \
  --random-seed 42
```

### Stage 3: Construct Items

Create forced-choice items from filled templates:

```bash
uv run bead items create-forced-choice-from-texts \
  --texts-file filled_templates/all_combinations.jsonl \
  --n-alternatives 2 \
  --sample 10 \
  --output items/2afc_pairs.jsonl
```

Create Likert scale items:

```bash
uv run bead items create-likert-7 \
  --text "The sentence is acceptable" \
  --output items/likert_items.jsonl
```

Validate items for task type:

```bash
uv run bead items validate-for-task-type items/2afc_pairs.jsonl \
  --task-type forced_choice
```

### Stage 4: Partition Lists

Partition items into balanced lists:

```bash
uv run bead lists partition items/2afc_pairs.jsonl lists/ \
  --n-lists 5 \
  --strategy balanced
```

With constraints for balance and coverage:

```bash
# Create uniqueness constraint
uv run bead lists create-uniqueness \
  --property-expression "item['verb']" \
  --output constraints/unique_verbs.jsonl
```

```bash
# Create batch coverage constraint
uv run bead lists create-batch-coverage \
  --property-expression "item['template_id']" \
  --target-values "0,1,2" \
  --min-coverage 1.0 \
  --output constraints/template_coverage.jsonl
```

<!--pytest.mark.skip(reason="requires items from previous pipeline stages")-->
```bash
# Partition with constraints
uv run bead lists partition items/2afc_pairs.jsonl lists/ \
  --n-lists 5 \
  --strategy balanced \
  --list-constraints constraints/unique_verbs.jsonl \
  --batch-constraints constraints/template_coverage.jsonl
```

View list statistics:

<!--pytest.mark.skip(reason="requires lists from previous pipeline stages")-->
```bash
uv run bead lists show-stats lists/
```

### Stage 5: Deploy Experiment

Generate jsPsych experiment:

<!--pytest.mark.skip(reason="requires lists and items from previous pipeline stages")-->
```bash
uv run bead deployment generate lists/ items/2afc_pairs.jsonl deployment/local \
  --experiment-type forced_choice \
  --title "Argument Structure Judgments" \
  --instructions "Choose the more natural sentence." \
  --distribution-strategy balanced
```

Export to JATOS format:

<!--pytest.mark.skip(reason="requires deployment from previous step")-->
```bash
uv run bead deployment export-jatos deployment/local deployment/study.jzip \
  --title "Argument Structure Judgments" \
  --description "Acceptability ratings for verb-frame combinations"
```

### Stage 6: Training and Evaluation

Collect data from JATOS:

<!--pytest.mark.skip(reason="requires external JATOS server")-->
```bash
uv run bead training collect-data results.jsonl \
  --jatos-url https://jatos.example.com \
  --api-token your-api-token \
  --study-id 123
```

View data statistics:

<!--pytest.mark.skip(reason="requires results.jsonl from data collection")-->
```bash
uv run bead training show-data-stats results.jsonl
```

Compute inter-annotator agreement:

<!--pytest.mark.skip(reason="requires results.jsonl from data collection")-->
```bash
uv run bead training compute-agreement \
  --annotations results.jsonl \
  --metric krippendorff_alpha \
  --data-type ordinal
```

## Using the Workflow Command

Run complete pipeline with one command:

<!--pytest.mark.skip(reason="requires full project configuration")-->
```bash
uv run bead workflow run --config config.yaml
```

Run specific stages:

<!--pytest.mark.skip(reason="requires full project configuration")-->
```bash
uv run bead workflow run \
  --config config.yaml \
  --stages resources,templates,items
```

Resume interrupted workflow:

<!--pytest.mark.skip(reason="requires prior workflow state")-->
```bash
uv run bead workflow resume
```

Check workflow status:

<!--pytest.mark.skip(reason="requires prior workflow state")-->
```bash
uv run bead workflow status
```

## Configuration File

Create a `config.yaml` file to configure the entire pipeline:

```yaml
project:
  name: "argument_structure"
  language_code: "eng"

paths:
  lexicons_dir: "lexicons"
  templates_dir: "templates"
  filled_templates_dir: "filled_templates"
  items_dir: "items"
  lists_dir: "lists"

template:
  filling_strategy: "exhaustive"

lists:
  n_lists: 5
  strategy: "balanced"

deployment:
  platform: "jatos"
  distribution_strategy: "balanced"
```

Load configuration:

<!--pytest.mark.skip(reason="requires full project configuration")-->
```bash
uv run bead --config-file config.yaml workflow run
```

## Next Steps

- [Resources Commands](resources.md): Detailed Stage 1 reference
- [Templates Commands](templates.md): Detailed Stage 2 reference
- [Items Commands](items.md): Detailed Stage 3 reference
- [Lists Commands](lists.md): Detailed Stage 4 reference
- [Deployment Commands](deployment.md): Detailed Stage 5 reference
- [Training Commands](training.md): Detailed Stage 6 reference
