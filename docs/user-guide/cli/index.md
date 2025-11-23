# Getting Started with the CLI

The bead CLI provides command-line tools for every stage of the experimental pipeline. This approach uses configuration files and shell commands to avoid Python programming.

## Quick Start

Complete pipeline in 6 stages:

```bash
# Stage 1: Import resources
bead resources import-verbnet --output lexicons/verbs.jsonl

# Stage 2: Fill templates
bead templates fill templates.jsonl lexicons/*.jsonl filled.jsonl \
  --strategy exhaustive

# Stage 3: Construct items
bead items construct --item-template template.jsonl \
  --filled-templates filled.jsonl --output items.jsonl

# Stage 4: Partition into lists
bead lists partition items.jsonl lists/ --n-lists 5 \
  --strategy balanced

# Stage 5: Generate experiment
bead deployment generate lists/ items.jsonl experiment/

# Stage 6: Collect data and train
bead training collect-data results.jsonl
```

## When to Use the CLI

Use the CLI when:
- You prefer configuration-driven workflows
- You want to avoid Python programming
- You're composing operations in shell scripts
- You're working with single resources or templates
- Your workflow is linear and straightforward

For batch operations, complex logic, or dynamic configuration, see the [Python API](../api/index.md).

## Command Groups

The CLI organizes commands into groups by pipeline stage:

- **resources**: Create lexicons and templates (Stage 1)
- **templates**: Fill templates with lexical items (Stage 2)
- **items**: Construct experimental items (Stage 3)
- **lists**: Partition items into experiment lists (Stage 4)
- **deployment**: Generate jsPsych/JATOS experiments (Stage 5)
- **training**: Collect data and train models (Stage 6)
- **workflow**: Run complete multi-stage pipelines
- **config**: Manage configuration files

## Getting Help

View available commands:
```bash
bead --help
bead resources --help
bead templates fill --help
```

## Configuration Files

The CLI uses YAML configuration files to define pipeline parameters:

```yaml
project:
  name: "my_experiment"
  language_code: "eng"

paths:
  lexicons_dir: "lexicons"
  templates_dir: "templates"
  items_dir: "items"
```

See [Configuration Guide](../configuration.md) for complete reference.

## Complete Workflow Example

For a complete working example using all 6 stages, see [CLI Workflows](workflows.md).

## Next Steps

- [CLI Workflows](workflows.md): Complete pipeline examples
- [Resources](resources.md): Stage 1 commands
- [Templates](templates.md): Stage 2 commands
- [Items](items.md): Stage 3 commands
- [Lists](lists.md): Stage 4 commands
- [Deployment](deployment.md): Stage 5 commands
- [Training](training.md): Stage 6 commands
