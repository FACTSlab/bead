# Deployment Module

The `bead.deployment` module provides jsPsych 8.x experiment generation with server-side list distribution via JATOS batch sessions.

## Basic Experiment Generation

Generate a jsPsych experiment from lists:

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jspsych.config import ExperimentConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec
from bead.lists import ExperimentList

# Load lists and items from fixtures
lists = read_jsonlines(Path("lists/experiment_lists.jsonl"), ExperimentList)
items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)

# Create item template
template = ItemTemplate(
    name="forced_choice",
    description="2AFC acceptability",
    judgment_type="acceptability",
    task_type="forced_choice",
    task_spec=TaskSpec(
        prompt="Which sentence sounds more natural?",
        options=["Option A", "Option B"],
    ),
    presentation_spec=PresentationSpec(mode="static"),
)

# Link items to template
items_dict = {item.id: item for item in items}
for item in items_dict.values():
    item.item_template_id = template.id

# Create experiment config
config = ExperimentConfig(
    experiment_type="forced_choice",
    title="Sentence Acceptability Study",
    description="Rate which sentence sounds more natural",
    instructions="You will see pairs of sentences. Select the more natural one.",
    randomize_trial_order=True,
    show_progress_bar=True,
    distribution_strategy=ListDistributionStrategy(
        strategy_type=DistributionStrategyType.BALANCED
    ),
    use_jatos=True,
)

# Generate experiment
generator = JsPsychExperimentGenerator(
    config=config,
    output_dir=Path("/tmp/deployment/experiment"),
)

output_dir = generator.generate(
    lists=lists,
    items=items_dict,
    templates={template.id: template},
)

print(f"Experiment generated in {output_dir}")
```

## Distribution Strategies

The deployment system supports 8 distribution strategies for participant assignment:

**BALANCED**: assign to least-used list

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)

strategy = ListDistributionStrategy(strategy_type=DistributionStrategyType.BALANCED)
```

**SEQUENTIAL**: round-robin (0, 1, 2, ..., N, 0, 1, ...)

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)

strategy = ListDistributionStrategy(strategy_type=DistributionStrategyType.SEQUENTIAL)
```

**RANDOM**: random selection

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)

strategy = ListDistributionStrategy(strategy_type=DistributionStrategyType.RANDOM)
```

**QUOTA_BASED**: fixed quota per list

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)

strategy = ListDistributionStrategy(
    strategy_type=DistributionStrategyType.QUOTA_BASED,
    strategy_config={
        "participants_per_list": 25,
        "allow_overflow": False,
    },
)
```

**LATIN_SQUARE**: counterbalancing

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)

strategy = ListDistributionStrategy(
    strategy_type=DistributionStrategyType.LATIN_SQUARE,
    strategy_config={"balanced": True},
)
```

**WEIGHTED_RANDOM**: non-uniform probabilities

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)

strategy = ListDistributionStrategy(
    strategy_type=DistributionStrategyType.WEIGHTED_RANDOM,
    strategy_config={
        "weight_expression": "list_metadata.priority || 1.0",
        "normalize_weights": True,
    },
)
```

**STRATIFIED**: balance across factors

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)

strategy = ListDistributionStrategy(
    strategy_type=DistributionStrategyType.STRATIFIED,
    strategy_config={
        "factors": ["condition", "verb_type"],
    },
)
```

**METADATA_BASED**: filter and rank by metadata

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)

strategy = ListDistributionStrategy(
    strategy_type=DistributionStrategyType.METADATA_BASED,
    strategy_config={
        "filter_expression": "list_metadata.difficulty === 'hard'",
        "rank_expression": "list_metadata.priority || 0",
        "rank_ascending": False,
    },
)
```

## Experiment Configuration

**ExperimentConfig** parameters:

```python
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jspsych.config import ExperimentConfig

config = ExperimentConfig(
    experiment_type="forced_choice",
    title="Study Title",
    description="Study description",
    instructions="Instructions for participants",
    distribution_strategy=ListDistributionStrategy(
        strategy_type=DistributionStrategyType.BALANCED
    ),
    randomize_trial_order=True,
    show_progress_bar=True,
    ui_theme="light",  # "light", "dark", "auto"
    on_finish_url=None,
    allow_backwards=False,
    show_click_target=False,
    minimum_duration_ms=0,
    use_jatos=True,
    prolific_completion_code=None,  # Auto-generates redirect URL
)
```

## Item Templates

Define task presentation and behavior:

```python
from bead.items.item_template import ItemTemplate, TaskSpec, PresentationSpec

# Forced choice template
template = ItemTemplate(
    name="2afc",
    description="Two-alternative forced choice",
    judgment_type="acceptability",
    task_type="forced_choice",
    task_spec=TaskSpec(
        prompt="Which is more natural?",
        options=["A", "B"],
    ),
    presentation_spec=PresentationSpec(
        mode="static",
    ),
)

# Ordinal scale template
template = ItemTemplate(
    name="likert7",
    description="7-point Likert scale",
    judgment_type="acceptability",
    task_type="ordinal_scale",
    task_spec=TaskSpec(
        prompt="Rate naturalness:",
        scale_bounds=(1, 7),
        scale_labels={1: "Very unnatural", 7: "Very natural"},
    ),
    presentation_spec=PresentationSpec(
        mode="static",
    ),
)
```

## JATOS Export

Export experiments as JATOS study packages (.jzip):

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import ExperimentConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec
from bead.lists import ExperimentList

# Load data
lists = read_jsonlines(Path("lists/experiment_lists.jsonl"), ExperimentList)
items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)

# Create template
template = ItemTemplate(
    name="forced_choice",
    description="2AFC",
    judgment_type="acceptability",
    task_type="forced_choice",
    task_spec=TaskSpec(prompt="Which is more natural?", options=["A", "B"]),
    presentation_spec=PresentationSpec(mode="static"),
)

items_dict = {item.id: item for item in items}
for item in items_dict.values():
    item.item_template_id = template.id

# Generate experiment
config = ExperimentConfig(
    experiment_type="forced_choice",
    title="Study",
    description="Acceptability",
    instructions="Select more natural sentence",
    distribution_strategy=ListDistributionStrategy(
        strategy_type=DistributionStrategyType.BALANCED
    ),
    use_jatos=True,
)

generator = JsPsychExperimentGenerator(config=config, output_dir=Path("/tmp/exp"))
output_dir = generator.generate(
    lists=lists, items=items_dict, templates={template.id: template}
)

# Export to JATOS
exporter = JATOSExporter(
    study_title="Sentence Acceptability Study",
    study_description="Forced choice acceptability judgments",
)

exporter.export(
    experiment_dir=output_dir,
    output_path=Path("/tmp/study.jzip"),
    component_title="Main Experiment",
)

print("JATOS package ready to import")
```

## Generated File Structure

```
output_dir/
├── index.html
├── js/
│   ├── experiment.js
│   └── list_distributor.js
├── css/
│   └── experiment.css
└── data/
    ├── config.json
    ├── lists.jsonl
    ├── items.jsonl
    ├── distribution.json
    └── templates.json
```

## Complete Example

Real working example from [gallery/eng/argument_structure/generate_deployment.py](../../../gallery/eng/argument_structure/generate_deployment.py):

```python
from pathlib import Path

from bead.data.serialization import read_jsonlines
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import ExperimentConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec
from bead.lists import ExperimentList

# Load lists and items from fixtures
lists = read_jsonlines(Path("lists/experiment_lists.jsonl"), ExperimentList)
items = read_jsonlines(Path("items/2afc_pairs.jsonl"), Item)

# Create item template
template = ItemTemplate(
    name="2afc_forced_choice",
    description="Two-alternative forced choice item",
    judgment_type="acceptability",
    task_type="forced_choice",
    task_spec=TaskSpec(
        prompt="Which sentence sounds more natural?",
        options=["Option A", "Option B"],
    ),
    presentation_spec=PresentationSpec(mode="static"),
)

# Link items to template
items_dict = {item.id: item for item in items}
for item in items_dict.values():
    item.item_template_id = template.id

# Create config
config = ExperimentConfig(
    experiment_type="forced_choice",
    title="Sentence Acceptability Study",
    description="Rate which sentence sounds more natural",
    instructions="Select the more natural sentence from each pair.",
    randomize_trial_order=True,
    show_progress_bar=True,
    distribution_strategy=ListDistributionStrategy(
        strategy_type=DistributionStrategyType.BALANCED
    ),
    use_jatos=True,
)

# Generate experiment
generator = JsPsychExperimentGenerator(
    config=config,
    output_dir=Path("/tmp/deployment/experiment"),
)

output_dir = generator.generate(
    lists=lists,
    items=items_dict,
    templates={template.id: template},
)

print(f"Generated experiment in {output_dir}")

# Export to JATOS
exporter = JATOSExporter(
    study_title="Sentence Acceptability",
    study_description="Acceptability judgments via forced choice",
)

exporter.export(
    experiment_dir=output_dir,
    output_path=Path("/tmp/deployment/study.jzip"),
)

print("JATOS package created")
```

## Design Principles

1. **Batch Mode Only**: All experiments package multiple lists with server-side distribution
2. **No Fallbacks**: All required parameters must be explicitly specified
3. **JATOS Integration**: Uses `jatos.batchSession` for server-side state
4. **Race Condition Safety**: Lock mechanism for concurrent participants

## Distribution Strategy Summary

| Strategy | Use For |
|----------|---------|
| `BALANCED` | Equal participants per list |
| `SEQUENTIAL` | Round-robin assignment |
| `RANDOM` | Random selection |
| `QUOTA_BASED` | Fixed quota per list |
| `LATIN_SQUARE` | Counterbalancing |
| `WEIGHTED_RANDOM` | Non-uniform probabilities |
| `STRATIFIED` | Balance across factors |
| `METADATA_BASED` | Custom filtering/ranking |

## Next Steps

- [Training module](training.md): Active learning with convergence detection
- [CLI reference](../cli/deployment.md): Command-line equivalents
- [Gallery example](../../../gallery/eng/argument_structure/generate_deployment.py): Full working script
