# bead

A comprehensive Python framework for constructing, deploying, and analyzing large-scale linguistic judgment experiments with active learning.

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Overview

`bead` implements a complete pipeline for linguistic research, from lexical resource construction through experimental deployment to model training with active learning. The framework is designed for researchers collecting acceptability judgments, inference judgments, and other linguistic annotations at scale.

### Key Features

- **6-Stage Pipeline**: From resources → templates → items → lists → deployment → training
- **Stand-off Annotation**: UUID-based references minimize data duplication and ensure consistency
- **Type-Safe**: Full Python 3.13 type hints with Pydantic v2 validation
- **Configuration-First**: YAML-based orchestration with comprehensive validation
- **Model Integration**: Support for HuggingFace, OpenAI, Anthropic models with caching
- **Active Learning**: Human-in-the-loop training with convergence detection
- **Multi-Language Support**: Language-agnostic core with ISO 639 language codes
- **Beautiful UI**: Material Design interfaces with jsPsych 7.x and JATOS deployment

### Why bead?

The name "bead" refers to the small, decorative elements that form part of window glazing, a play on the [glazing](https://github.com/aaronstevenwhite/glazing) package (which provides access to VerbNet, PropBank, and FrameNet). Just as beads add structure and detail to a window, `bead` provides:

- **Reproducibility**: Full provenance tracking and deterministic pipelines
- **Scalability**: Handles combinatorial explosion with streaming and sampling strategies
- **Flexibility**: Plugin architecture for models, resources, and experiment types
- **Research-Ready**: Designed specifically for linguistic research workflows

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Pipeline Stages](#pipeline-stages)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Requirements

- Python 3.13 or higher
- pip or uv for package management

### Basic Installation

```bash
# Create and activate virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install bead
pip install bead

# Install optional dependencies
pip install bead[api]       # OpenAI, Anthropic, Google APIs
pip install bead[training]  # PyTorch Lightning, TensorBoard
pip install bead[dev]       # Development tools (pytest, ruff, pyright)
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/aaronstevenwhite/bead.git
cd bead

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install in editable mode with all dependencies
pip install -e ".[dev,api,training]"

# Verify installation
bead --version
pytest tests/
```

---

## Quick Start

### 1. Initialize a Project

```bash
# Create new project structure
bead init my-experiment

cd my-experiment
```

This creates a project directory with:
```
my-experiment/
├── config.yaml              # Pipeline configuration
├── lexicons/                # Lexical resources
├── templates/               # Sentence templates
├── filled_templates/        # Generated sentences
├── items/                   # Experimental items
├── lists/                   # Participant lists
└── deployment/              # jsPsych experiments
```

### 2. Define Resources (Stage 1)

Create lexical items and templates:

```python
from bead.resources import LexicalItem, Template, Lexicon, TemplateCollection

# Define lexical items
verb = LexicalItem(
    lemma="walk",
    pos="VERB",
    language_code="en",
    features={"tense": "present", "transitive": False},
    attributes={"frequency": 1000}
)

# Define template
template = Template(
    text="The {det} {noun} {verb}",
    slots=["det", "noun", "verb"],
    language_code="en"
)

# Save to JSONLines
lexicon = Lexicon(items=[verb])
lexicon.save("lexicons/verbs.jsonl")

templates = TemplateCollection(templates=[template])
templates.save("templates/basic.jsonl")
```

### 3. Fill Templates (Stage 2)

Generate sentences from templates:

```python
from bead.templates import TemplateFiller

filler = TemplateFiller(
    strategy="exhaustive",  # or "random", "stratified", "mlm"
    lexicons={"verbs": "lexicons/verbs.jsonl"},
    output_path="filled_templates/output.jsonl"
)

filled = filler.fill(templates)
filled.save("filled_templates/output.jsonl")
```

### 4. Construct Items (Stage 3)

Create experimental items with model-based constraints:

```python
from bead.items import ItemConstructor, ItemTemplate

constructor = ItemConstructor(
    models=["gpt2"],
    cache_enabled=True
)

# Create 2AFC acceptability items
items = constructor.construct_forced_choice_items(
    filled_templates=filled,
    n_alternatives=2,
    pair_type="minimal_pair"
)

items.save("items/2afc_items.jsonl")
```

### 5. Partition Lists (Stage 4)

Create balanced experimental lists:

```python
from bead.lists import ListPartitioner
from bead.lists.constraints import (
    BatchCoverageConstraint,
    BatchBalanceConstraint
)

partitioner = ListPartitioner(random_seed=42)

lists = partitioner.partition_with_batch_constraints(
    items=items.get_uuids(),
    n_lists=8,
    batch_constraints=[
        BatchCoverageConstraint(
            property_expression="item['template_id']",
            target_values=list(range(10)),
            min_coverage=1.0
        ),
        BatchBalanceConstraint(
            property_expression="item['pair_type']",
            target_distribution={"minimal_pair": 0.5, "control": 0.5},
            tolerance=0.05
        )
    ],
    metadata=items.get_metadata()
)

lists.save("lists/experiment_lists.jsonl")
```

### 6. Deploy Experiment (Stage 5)

Generate jsPsych experiment for JATOS:

```bash
bead deploy \
  --lists lists/experiment_lists.jsonl \
  --experiment-type 2afc \
  --output deployment/experiment.jzip
```

### 7. Train with Active Learning (Stage 6)

Train models with human-in-the-loop:

```python
from bead.active_learning import ActiveLearningLoop

loop = ActiveLearningLoop(
    config="config.yaml",
    strategy="uncertainty_sampling",
    convergence_threshold=0.05
)

# Run until convergence
loop.run()
```

---

## Core Concepts

### Stand-off Annotation

`bead` uses UUID-based references to avoid data duplication:

```python
# Items reference filled templates by UUID
item = Item(
    filled_template_refs=[uuid1, uuid2],  # References, not copies
    judgment_type="forced_choice"
)

# Lists reference items by UUID
experiment_list = ExperimentList(
    item_refs=[item_uuid1, item_uuid2, ...],  # References only
    list_number=0
)
```

This ensures:
- **Consistency**: Single source of truth for each object
- **Efficiency**: Minimal storage and memory usage
- **Provenance**: Complete tracking from resources → deployment

### Metadata Preservation

Every object tracks comprehensive metadata:

```python
class BeadBaseModel(BaseModel):
    """Base model for all bead objects."""

    id: UUID                # UUIDv7 (time-ordered)
    created_at: datetime    # ISO 8601 timestamp
    modified_at: datetime   # Auto-updated
    metadata: dict[str, Any]  # Custom metadata
```

Metadata flows through pipeline:
- **Stage 1**: Lexical features, source information
- **Stage 2**: Slot fillers, constraint satisfaction
- **Stage 3**: Model scores, probabilities, embeddings
- **Stage 4**: List balance metrics, constraint violations
- **Stage 5**: Presentation parameters, UI configuration
- **Stage 6**: Training metrics, model checkpoints

### Configuration-First Design

Single YAML file orchestrates entire pipeline:

```yaml
project:
  name: "verb-acceptability"
  language_code: "en"

resources:
  lexicons:
    - path: "lexicons/verbs.jsonl"
  templates:
    - path: "templates/frames.jsonl"

template:
  filling_strategy: "mlm"
  mlm_model_name: "bert-base-uncased"

items:
  judgment_type: "forced_choice"
  n_alternatives: 2

lists:
  n_lists: 8
  strategy: "quantile_balanced"

  batch_constraints:
    - type: "coverage"
      property_expression: "item['template_id']"
      target_values: [0, 1, 2, 3, 4, 5]

    - type: "balance"
      property_expression: "item['pair_type']"
      target_distribution:
        minimal_pair: 0.5
        control: 0.5

deployment:
  platform: "jatos"
  experiment_type: "2afc"

active_learning:
  strategy: "uncertainty_sampling"
  convergence_threshold: 0.05
```

---

## Pipeline Stages

### Stage 1: Resource Construction

**Purpose**: Define lexical items and templates with rich metadata

**Key Classes**:
- `LexicalItem`: Words/phrases with linguistic features
- `Template`: Sentence patterns with typed slots
- `Lexicon`: Collections of lexical items
- `TemplateCollection`: Collections of templates

**Features**:
- Extensional constraints: Finite sets of valid fillers
- Intensional constraints: Rule-based constraints (DSL)
- Relational constraints: Cross-slot dependencies
- External resource integration: VerbNet, FrameNet, UniMorph via adapters

**Example**:
```python
from bead.resources import LexicalItem, Template
from bead.resources.constraints import IntensionalConstraint

# Lexical item with features
verb = LexicalItem(
    lemma="run",
    pos="VERB",
    language_code="en",
    features={"transitive": False, "frame": "motion"}
)

# Template with constraints
template = Template(
    text="{det} {noun} {verb} {prep} {det} {noun}",
    slots=["det", "noun", "verb", "prep", "det", "noun"],
    constraints=[
        IntensionalConstraint(
            expression="verb.features.transitive == True",
            applies_to=["verb"]
        )
    ]
)
```

### Stage 2: Template Filling

**Purpose**: Generate sentences by filling template slots

**Strategies**:
1. **Exhaustive**: Generate all valid combinations
2. **Random**: Sample random valid combinations
3. **Stratified**: Sample with distribution control
4. **MLM**: Use masked language models (BERT, RoBERTa) for contextually appropriate fillers
5. **Mixed**: Different strategies per slot

**Features**:
- Constraint resolution with external resources
- Lazy evaluation for large combinatorial spaces
- Streaming output for memory efficiency
- Content-addressable caching for MLM predictions

**Example**:
```python
from bead.templates import TemplateFiller

# MLM-based filling
filler = TemplateFiller(
    strategy="mlm",
    mlm_model_name="bert-base-uncased",
    mlm_beam_size=5,
    mlm_fill_direction="left_to_right"
)

filled = filler.fill(templates, lexicons)
```

### Stage 3: Item Construction

**Purpose**: Create experimental items from filled templates

**Judgment Types**:
- `acceptability`: Linguistic acceptability/grammaticality
- `inference`: Semantic relationships (NLI)
- `similarity`: Semantic similarity/distance
- `plausibility`: Event/statement likelihood
- `comprehension`: Understanding/recall
- `preference`: Subjective preference

**Task Types**:
- `forced_choice`: Pick one option (2AFC, 3AFC, etc.)
- `multi_select`: Pick multiple options
- `ordinal_scale`: Likert scales, sliders
- `magnitude`: Unbounded numeric values
- `binary`: Yes/no, true/false
- `categorical`: Unordered categories
- `free_text`: Open-ended responses
- `cloze`: Fill-in-the-blank

**Model Integration**:
```python
from bead.items import ItemConstructor

constructor = ItemConstructor(
    models=[
        {"name": "gpt2", "type": "huggingface_lm"},
        {"name": "roberta-large-mnli", "type": "huggingface_nli"}
    ],
    cache_enabled=True
)

# Create items with model-based constraints
items = constructor.construct(
    item_templates=templates,
    filled_templates=filled,
    constraints=[
        {"type": "lm_probability", "threshold": 0.001},
        {"type": "minimal_pair_distance", "min_diff": 0.5}
    ]
)
```

### Stage 4: List Construction

**Purpose**: Partition items into balanced experimental lists

**Constraint Types**:

**List Constraints** (per-list):
- `UniquenessConstraint`: No duplicate values
- `CountConstraint`: Exact count requirements
- `ProportionConstraint`: Distribution requirements
- `DiversityConstraint`: Value spread requirements
- `GroupedQuantileConstraint`: Stratified sampling by group

**Batch Constraints** (across all lists):
- `BatchCoverageConstraint`: Ensure all values appear
- `BatchBalanceConstraint`: Maintain global distributions
- `BatchDiversityConstraint`: Spread values across lists
- `BatchMinOccurrenceConstraint`: Minimum occurrence guarantees

**Features**:
- Iterative refinement algorithm
- Quantile-based balancing
- Multi-constraint satisfaction
- Stand-off annotation with metadata dictionaries

**Example**:
```python
from bead.lists import ListPartitioner
from bead.lists.constraints import (
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    UniquenessConstraint
)

partitioner = ListPartitioner(random_seed=42)

lists = partitioner.partition_with_batch_constraints(
    items=item_uuids,
    n_lists=8,
    list_constraints=[
        UniquenessConstraint(
            property_expression="item['verb_lemma']"
        )
    ],
    batch_constraints=[
        BatchCoverageConstraint(
            property_expression="item['template_id']",
            target_values=list(range(26)),
            min_coverage=1.0
        ),
        BatchDiversityConstraint(
            property_expression="item['verb_lemma']",
            max_lists_per_value=4
        )
    ],
    metadata=item_metadata
)
```

### Stage 5: Deployment

**Purpose**: Generate web-based experiments for participant data collection

**Features**:
- jsPsych 7.x integration with TypeScript
- Material Design UI components
- JATOS `.jzip` export for easy deployment
- Attention checks and practice trials
- Response time tracking
- Mobile-responsive design

**Supported Platforms**:
- JATOS (primary)
- Prolific (metadata integration)
- Custom web servers

**Example**:
```bash
# Generate JATOS experiment
bead deploy \
  --lists lists/experiment_lists.jsonl \
  --experiment-type likert_scale \
  --attention-checks 3 \
  --output experiments/acceptability.jzip

# Upload to JATOS
bead jatos upload \
  --url https://jatos.example.com \
  --study-id 123 \
  --file experiments/acceptability.jzip
```

### Stage 6: Training & Active Learning

**Purpose**: Train models with human-in-the-loop until convergence

**Features**:
- Automatic data download from JATOS/Prolific APIs
- Participant metadata merging
- Multiple active learning strategies:
  - Uncertainty sampling (entropy, least confidence, margin)
  - Query by committee
  - Expected model change
- Convergence detection (compare to human inter-annotator agreement)
- Model training frameworks:
  - HuggingFace Trainer
  - PyTorch Lightning
- Visualization:
  - TensorBoard
  - Weights & Biases

**Active Learning Loop**:
```python
from bead.active_learning import ActiveLearningLoop

loop = ActiveLearningLoop(
    config="config.yaml",
    initial_training_size=500,
    budget_per_iteration=200,
    max_iterations=20
)

# Run until convergence
results = loop.run()

# Results include:
# - Model checkpoints
# - Performance metrics per iteration
# - Convergence status
# - Selected items for next round
```

**Convergence Detection**:
```yaml
training:
  convergence:
    metric: "krippendorff_alpha"  # Human agreement baseline
    threshold: 0.05               # |model_acc - human_agree| < 0.05
    min_iterations: 3
```

---

## Architecture

### Package Structure

```
bead/
├── resources/           # Stage 1: Lexical items, templates
│   ├── models.py
│   ├── lexicon.py
│   ├── template.py
│   └── constraints.py
│
├── templates/           # Stage 2: Template filling
│   ├── filler.py
│   ├── strategies.py    # exhaustive, random, stratified, mlm
│   ├── resolver.py
│   └── streaming.py
│
├── items/               # Stage 3: Item construction
│   ├── models.py
│   ├── constructor.py
│   ├── adapters/        # Model integrations
│   │   ├── huggingface.py
│   │   ├── openai.py
│   │   └── anthropic.py
│   └── cache.py
│
├── lists/               # Stage 4: List partitioning
│   ├── models.py
│   ├── constraints.py   # List & batch constraints
│   ├── partitioner.py
│   └── balancer.py
│
├── deployment/          # Stage 5: Experiment generation
│   ├── jspsych/
│   │   ├── generator.py
│   │   └── templates/
│   └── jatos/
│       ├── exporter.py
│       └── client.py
│
├── active_learning/     # Stage 6: Training & selection
│   ├── loop.py
│   ├── strategies.py
│   ├── models/
│   │   ├── binary.py
│   │   ├── categorical.py
│   │   └── forced_choice.py
│   └── trainers/
│       ├── huggingface.py
│       └── lightning.py
│
├── data_collection/     # Data retrieval
│   ├── jatos.py
│   ├── prolific.py
│   └── merger.py
│
├── evaluation/          # Metrics & reporting
│   ├── metrics.py
│   ├── agreement.py
│   └── reports.py
│
├── simulation/          # Simulation framework
│   ├── runner.py
│   └── generators.py
│
├── dsl/                 # Constraint DSL
│   ├── parser.py
│   ├── evaluator.py
│   └── stdlib.py
│
├── config/              # Configuration system
│   ├── models.py
│   ├── loader.py
│   └── validation.py
│
├── data/                # Data management
│   ├── base.py
│   ├── identifiers.py   # UUIDv7 generation
│   ├── language_codes.py
│   └── serialization.py
│
├── adapters/            # External resources
│   ├── glazing.py       # VerbNet, PropBank, FrameNet
│   ├── unimorph.py      # Morphological features
│   └── cache.py
│
└── cli/                 # Command-line interface
    ├── main.py
    ├── resources.py
    ├── templates.py
    ├── items.py
    ├── lists.py
    ├── deployment.py
    └── training.py
```

### Design Principles

1. **Stand-off Annotation**: UUID-based references minimize duplication
2. **Metadata Preservation**: Complete provenance tracking throughout
3. **Type Safety**: Full Python 3.13 type hints, no `Any` or `object`
4. **Configuration-First**: YAML orchestration for reproducibility
5. **Modularity**: Each stage is independently usable
6. **Extensibility**: Plugin architecture for models and resources
7. **Language-Agnostic**: ISO 639 language codes, multi-language support

### Data Flow

```
Stage 1: Resources
├── lexicons/verbs.jsonl (UUIDs: v1, v2, ...)
└── templates/frames.jsonl (UUIDs: t1, t2, ...)

Stage 2: Filled Templates
└── filled_templates.jsonl (UUIDs: f1, f2, ...)
    ├── references: [v1, t1]  # Stand-off annotation
    └── metadata: {slot_fillers: {...}}

Stage 3: Items
└── items.jsonl (UUIDs: i1, i2, ...)
    ├── references: [f1, f2]
    └── metadata: {lm_scores: {...}}

Stage 4: Lists
└── lists.jsonl (UUIDs: l1, l2, ...)
    ├── item_refs: [i1, i2, i3, ...]
    └── metadata: {balance_metrics: {...}}

Stage 5: Deployment
└── experiment.jzip
    ├── Contains: Lists + resolved items + resources
    └── Format: jsPsych timeline JSON

Stage 6: Results
└── results.jsonl
    ├── Contains: Participant responses
    └── Provenance: [l1, i1, f1, v1, t1]
```

---

## Configuration

### Complete Configuration Example

```yaml
# ============================================================================
# Project Configuration
# ============================================================================
project:
  name: "argument_structure"
  language_code: "eng"
  description: "VerbNet argument structure alternations"
  version: "1.0.0"

# ============================================================================
# Paths
# ============================================================================
paths:
  data_dir: "."
  output_dir: "."
  cache_dir: ".cache"
  lexicons_dir: "lexicons"
  templates_dir: "templates"
  items_dir: "items"
  lists_dir: "lists"

# ============================================================================
# Resources (Stage 1)
# ============================================================================
resources:
  lexicons:
    - path: "lexicons/verbnet_verbs.jsonl"
      name: "verbnet_verbs"
    - path: "lexicons/bleached_nouns.jsonl"
      name: "bleached_nouns"

  templates:
    - path: "templates/generic_frames.jsonl"
      name: "generic_frames"

# ============================================================================
# Template Filling (Stage 2)
# ============================================================================
template:
  filling_strategy: "mixed"

  # MLM settings
  mlm:
    model_name: "bert-base-uncased"
    beam_size: 5
    top_k: 10
    device: "cpu"
    cache_enabled: true

  # Per-slot strategies
  slot_strategies:
    verb:
      strategy: "exhaustive"
    noun:
      strategy: "exhaustive"
    adjective:
      strategy: "mlm"
      beam_size: 3

# ============================================================================
# Items (Stage 3)
# ============================================================================
items:
  judgment_type: "forced_choice"
  n_alternatives: 2

  models:
    - name: "gpt2"
      provider: "huggingface"
      device: "cpu"
      use_for_scoring: true

  construction:
    create_minimal_pairs: true
    pair_types:
      - "same_verb"
      - "different_verb"

    score_filtering:
      enabled: true
      min_score_diff: 0.5

# ============================================================================
# Lists (Stage 4)
# ============================================================================
lists:
  strategy: "quantile_balanced_minimal_pairs"
  n_lists: 8
  items_per_list: 100
  quantile_bins: 10

  # Per-list constraints
  constraints:
    - type: "uniqueness"
      property_expression: "item.metadata.verb_lemma"

    - type: "grouped_quantile"
      property_expression: "item.metadata.lm_score_diff"
      group_by_expression: "item.metadata.pair_type"
      n_quantiles: 10
      items_per_quantile: 5

  # Batch-level constraints (across all lists)
  batch_constraints:
    - type: "coverage"
      property_expression: "item['template_id']"
      target_values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      min_coverage: 1.0

    - type: "balance"
      property_expression: "item['pair_type']"
      target_distribution:
        same_verb: 0.5
        different_verb: 0.5
      tolerance: 0.05

    - type: "diversity"
      property_expression: "item['verb_lemma']"
      max_lists_per_value: 4

# ============================================================================
# Deployment (Stage 5)
# ============================================================================
deployment:
  platform: "jatos"

  experiment:
    title: "Sentence Acceptability Judgments"
    description: "Rate which sentence sounds more natural"
    estimated_duration_minutes: 15

  jspsych:
    version: "7.3.0"
    trial:
      type: "html-button-response"
    choices:
      - "Sentence A"
      - "Sentence B"
    randomize_order: true
    randomize_choices: true

  participants:
    n_per_list: 30
    payment_usd: 2.50

# ============================================================================
# Training & Active Learning (Stage 6)
# ============================================================================
training:
  convergence:
    metric: "krippendorff_alpha"
    threshold: 0.05
    min_iterations: 3

  model:
    architecture: "transformer"
    model_name: "bert-base-uncased"
    learning_rate: 2e-5
    batch_size: 16
    epochs_per_iteration: 3
    device: "cpu"

  data:
    validation_split: 0.2
    random_seed: 42
    balance_classes: true

active_learning:
  strategy: "uncertainty_sampling"
  method: "entropy"
  budget_per_iteration: 200
  max_iterations: 20
  stopping_criterion: "convergence"
  initial_training_size: 500

  promote_diversity: true
  diversity_lambda: 0.1

# ============================================================================
# Evaluation
# ============================================================================
evaluation:
  cross_validation:
    k_folds: 5
    stratify_by: "metadata.pair_type"
    shuffle: true
    random_seed: 42

  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"

  interannotator:
    metrics:
      - "krippendorff_alpha"
      - "fleiss_kappa"
    min_raters_per_item: 3

# ============================================================================
# Logging
# ============================================================================
logging:
  level: "INFO"
  file:
    enabled: true
    path: "pipeline.log"
  console:
    enabled: true
    colored: true
```

---

## Usage Examples

### Example 1: Verb Acceptability Study

```python
from bead.resources import LexicalItem, Template, Lexicon, TemplateCollection
from bead.templates import TemplateFiller
from bead.items import ItemConstructor
from bead.lists import ListPartitioner

# 1. Define resources
verbs = [
    LexicalItem(lemma="walk", pos="VERB", features={"transitive": False}),
    LexicalItem(lemma="eat", pos="VERB", features={"transitive": True}),
    LexicalItem(lemma="sleep", pos="VERB", features={"transitive": False})
]

template = Template(
    text="The person {verb} the thing",
    slots=["verb"],
    language_code="en"
)

# 2. Fill templates
filler = TemplateFiller(strategy="exhaustive")
filled = filler.fill(
    templates=[template],
    lexicons={"verbs": verbs}
)

# 3. Construct 2AFC items
constructor = ItemConstructor(models=["gpt2"])
items = constructor.construct_forced_choice_items(
    filled_templates=filled,
    n_alternatives=2,
    pair_type="minimal_pair"
)

# 4. Partition into lists
partitioner = ListPartitioner()
lists = partitioner.partition(
    items=items.get_uuids(),
    n_lists=4,
    metadata=items.get_metadata()
)

# 5. Deploy
lists.save("lists/experiment.jsonl")
```

### Example 2: Inference Judgment Study

```python
from bead.items import ItemConstructor, InferenceItemTemplate

# Create inference items
constructor = ItemConstructor(
    models=["roberta-large-mnli"]
)

template = InferenceItemTemplate(
    premise_template="{det} {noun} {verb} {det} {noun}",
    hypothesis_template="{det} {noun} {verb}",
    judgment_type="inference",
    task_type="ordinal_scale",
    scale_points=5,
    scale_labels=["Definitely not", "Probably not", "Maybe", "Probably", "Definitely"]
)

items = constructor.construct_from_template(
    template=template,
    filled_templates=filled
)
```

### Example 3: MLM-Based Template Filling

```python
from bead.templates import TemplateFiller

# Use BERT for contextually appropriate fillers
filler = TemplateFiller(
    strategy="mlm",
    mlm_model_name="bert-base-uncased",
    mlm_beam_size=5,
    mlm_fill_direction="left_to_right",
    mlm_top_k=20,
    mlm_cache_enabled=True
)

filled = filler.fill(
    templates=templates,
    lexicons=lexicons,
    max_combinations=1000
)
```

### Example 4: Batch Constraint Optimization

```python
from bead.lists import ListPartitioner
from bead.lists.constraints import (
    BatchCoverageConstraint,
    BatchBalanceConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint
)

partitioner = ListPartitioner(random_seed=42)

lists = partitioner.partition_with_batch_constraints(
    items=item_uuids,
    n_lists=8,
    batch_constraints=[
        # Ensure all 26 templates appear
        BatchCoverageConstraint(
            property_expression="item['template_id']",
            target_values=list(range(26)),
            min_coverage=1.0
        ),
        # Maintain 50/50 balance across all lists
        BatchBalanceConstraint(
            property_expression="item['pair_type']",
            target_distribution={"same_verb": 0.5, "different_verb": 0.5},
            tolerance=0.05
        ),
        # Spread verbs across participants
        BatchDiversityConstraint(
            property_expression="item['verb_lemma']",
            max_lists_per_value=4
        ),
        # Ensure each quantile appears ≥50 times
        BatchMinOccurrenceConstraint(
            property_expression="item['quantile']",
            min_occurrences=50
        )
    ],
    metadata=metadata,
    max_iterations=500,
    tolerance=0.05
)
```

### Example 5: Active Learning Loop

```python
from bead.active_learning import ActiveLearningLoop
from bead.data_collection import JATOSClient, ProlificClient

# Set up active learning
loop = ActiveLearningLoop(
    config="config.yaml",
    strategy="uncertainty_sampling",
    method="entropy",
    budget_per_iteration=200
)

# Connect to data sources
loop.connect_jatos(
    url="https://jatos.example.com",
    study_id=123
)
loop.connect_prolific(
    api_key=os.environ["PROLIFIC_API_KEY"]
)

# Run until convergence
results = loop.run()

print(f"Converged in {results['iterations']} iterations")
print(f"Final accuracy: {results['final_accuracy']:.3f}")
print(f"Human agreement: {results['human_agreement']:.3f}")
```

---

## API Reference

### Core Modules

#### `bead.resources`

**Classes**:
- `LexicalItem`: Lexical item with features and metadata
- `Lexicon`: Collection of lexical items
- `Template`: Sentence template with slots and constraints
- `TemplateCollection`: Collection of templates
- `FilledTemplate`: Template with filled slots

**Constraint Types**:
- `ExtensionalConstraint`: Finite set constraints
- `IntensionalConstraint`: Rule-based constraints (DSL)
- `RelationalConstraint`: Cross-slot constraints

#### `bead.templates`

**Classes**:
- `TemplateFiller`: Main template filling engine
- `ConstraintResolver`: Constraint satisfaction resolver

**Strategies**:
- `ExhaustiveStrategy`: Generate all valid combinations
- `RandomStrategy`: Random sampling
- `StratifiedStrategy`: Stratified sampling
- `MLMStrategy`: Masked language model generation
- `MixedStrategy`: Per-slot strategy configuration

#### `bead.items`

**Classes**:
- `Item`: Experimental item
- `ItemTemplate`: Item construction specification
- `ItemConstructor`: Item generation engine
- `ItemElement`: Item component (text, audio, etc.)

**Judgment Types**: `acceptability`, `inference`, `similarity`, `plausibility`, `comprehension`, `preference`

**Task Types**: `forced_choice`, `multi_select`, `ordinal_scale`, `magnitude`, `binary`, `categorical`, `free_text`, `cloze`

#### `bead.lists`

**Classes**:
- `ExperimentList`: List of items for presentation
- `ListPartitioner`: List partitioning engine
- `ListBalancer`: Quantile-based balancing

**List Constraints**:
- `UniquenessConstraint`: No duplicate values in list
- `CountConstraint`: Exact count requirements
- `ProportionConstraint`: Distribution requirements
- `DiversityConstraint`: Value spread requirements
- `GroupedQuantileConstraint`: Stratified sampling by group

**Batch Constraints** (apply across all lists):
- `BatchCoverageConstraint`: Ensure all target values appear
- `BatchBalanceConstraint`: Maintain global distributions
- `BatchDiversityConstraint`: Spread values across lists
- `BatchMinOccurrenceConstraint`: Minimum occurrence guarantees

#### `bead.deployment`

**Classes**:
- `JSPsychGenerator`: Generate jsPsych experiments
- `JATOSExporter`: Export to JATOS `.jzip` format
- `JATOSClient`: JATOS API client

#### `bead.active_learning`

**Classes**:
- `ActiveLearningLoop`: Main active learning loop
- `UncertaintySampler`: Uncertainty-based sampling
- `QueryByCommittee`: Committee-based sampling

**Models**:
- `BinaryClassifier`: Binary judgment models
- `CategoricalClassifier`: Multi-class judgment models
- `ForcedChoiceModel`: Forced choice models

#### `bead.data_collection`

**Classes**:
- `JATOSClient`: Download data from JATOS
- `ProlificClient`: Download metadata from Prolific
- `DataMerger`: Merge JATOS + Prolific data

#### `bead.config`

**Classes**:
- `BeadConfig`: Complete pipeline configuration
- `ConfigLoader`: Load and validate YAML config
- `PathsConfig`: File system paths
- `ResourceConfig`: Resource configuration
- `TemplateConfig`: Template filling configuration
- `ItemsConfig`: Item construction configuration
- `ListConfig`: List partitioning configuration
- `DeploymentConfig`: Deployment configuration
- `TrainingConfig`: Training configuration
- `ActiveLearningConfig`: Active learning configuration

For complete API documentation, see the [API Reference](docs/api-reference/).

---

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/aaronstevenwhite/bead.git
cd bead

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev,api,training]"
```

### Run Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=bead --cov-report=html

# Run specific test module
pytest tests/lists/test_partitioner_batch.py -v

# Run with parallel execution
pytest tests/ -n auto
```

### Code Quality

```bash
# Lint with ruff
ruff check bead/

# Format with ruff
ruff format bead/

# Type check with pyright
pyright bead/

# All checks
ruff check bead/ && ruff format bead/ && pyright bead/
```

### Project Structure

- **`bead/`**: Main package code
- **`tests/`**: Test suite (pytest)
- **`docs/`**: Documentation (MkDocs)
- **`gallery/`**: Language-specific research examples
- **`design-notes/`**: Design documentation and proposals

### Testing Philosophy

- **Unit Tests**: Test each module independently with mocks
- **Integration Tests**: Test stage interactions
- **Property Tests**: Generative testing with Hypothesis
- **Target Coverage**: >90%

### Code Standards

- **Python Version**: 3.13+
- **Type Hints**: Full type hints, no `Any` or `object`
- **Docstrings**: NumPy style with comprehensive examples
- **Linting**: Ruff (zero errors, zero warnings)
- **Type Checking**: Pyright in strict mode
- **Formatting**: Ruff (88 character line length)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

1. **Report Bugs**: Open an issue with reproducible example
2. **Request Features**: Propose new features with use cases
3. **Submit Pull Requests**: Fix bugs or implement features
4. **Improve Documentation**: Fix typos, add examples, clarify explanations
5. **Add Language Examples**: Contribute research examples for new languages

### Language-Specific Examples

The `gallery/` directory contains complete research project replication packages organized by language:

```
gallery/
├── eng/                   # English projects
│   └── argument_structure/
├── kor/                   # Korean projects (future)
├── igb/                   # Igbo projects (future)
└── mar/                   # Marathi projects (future)
```

Each language directory should contain:
- Complete config.yaml
- Lexicons and templates
- Analysis notebooks (optional)
- README with linguistic background

See [gallery/README.md](gallery/README.md) for contribution guidelines.

---

## Citation

If you use `bead` in your research, please cite:

```bibtex
@software{white2025bead,
  author = {White, Aaron Steven},
  title = {bead: A framework for linguistic judgment experiments},
  year = {2025},
  url = {https://github.com/aaronstevenwhite/bead},
  version = {0.1.0}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **glazing**: VerbNet, PropBank, FrameNet access
- **unimorph**: Morphological features
- **Pydantic**: Data validation
- **HuggingFace**: Model infrastructure
- **jsPsych**: Experiment framework
- **JATOS**: Deployment platform

---

## Support

- **Documentation**: [https://bead.readthedocs.io](https://bead.readthedocs.io)
- **Issues**: [https://github.com/aaronstevenwhite/bead/issues](https://github.com/aaronstevenwhite/bead/issues)
- **Discussions**: [https://github.com/aaronstevenwhite/bead/discussions](https://github.com/aaronstevenwhite/bead/discussions)
- **Email**: aaron.white@rochester.edu

---

**Built with ❤️ for linguistic research**
