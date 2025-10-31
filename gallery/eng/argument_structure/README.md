# Argument Structure Active Learning Pipeline

A comprehensive framework for collecting human judgments on argument structure alternations using active learning with convergence detection to human-level inter-annotator agreement.

## Overview

This project implements a human-in-the-loop active learning pipeline for studying **argument structure alternations** in English. The pipeline:

1. **Extracts verb-specific templates** from VerbNet with detailed frame information
2. **Generates generic frame structures** by abstracting across verb-specific patterns
3. **Tests all verbs in all frame structures** using a full cross-product design
4. **Generates 2AFC (two-alternative forced choice) pairs** stratified by language model scores
5. **Iteratively collects human judgments** through web-based experiments
6. **Trains predictive models** that converge to human inter-annotator agreement
7. **Detects convergence** using Krippendorff's alpha and other reliability metrics

**Key Innovation:** Rather than testing only "known good" verb-frame combinations from VerbNet (~21,453 attested patterns), this approach systematically tests **every verb in every frame structure**, enabling discovery of both grammatical and ungrammatical patterns.

## Linguistic Background

### Argument Structure Alternations

Argument structure alternations describe systematic variations in how verbs express their semantic arguments syntactically. For example:

```
Transitive:     John broke the window.
Intransitive:   The window broke.
                (Causative/Inchoative Alternation)

Active:         Mary loaded hay onto the wagon.
Passive:        Hay was loaded onto the wagon.
                (Active/Passive Alternation)

Dative:         John gave Mary a book.
                John gave a book to Mary.
                (Dative Alternation)
```

Not all verbs participate in all alternations:

```
✓ John broke the window.  ✓ The window broke.
✓ John cut the bread.      ✗ The bread cut.
✓ The ice melted.          ✗ John melted the ice.
```

### VerbNet

This project uses **VerbNet** (Kipper et al., 2008), a lexical resource accessed through the **Glazing** interface. VerbNet provides:

- **~3,000 unique verb lemmas** organized into semantic classes
- **~21,453 verb-specific frame templates** with syntactic patterns
- **~26 unique generic frame structures** (extracted by this pipeline)
- **Detailed frame information** including syntax, thematic roles, and examples

### MegaAttitude Frame System

For clausal complement constructions, the project uses the **MegaAttitude** frame inventory, which provides comprehensive coverage of:

1. **Finite complements:** that/whether + indicative/subjunctive/conditional
2. **Non-finite complements:** to-infinitive, gerund, perfect infinitive, bare infinitive
3. **Wh-complements:** finite and infinitival
4. **PP complements** with clausal objects
5. **Null/pro-clausal** complements

### Research Questions

1. **Acceptability Judgments:** How acceptable is verb V in frame F?
2. **Generalization:** Can models predict human judgments for unseen verb-frame combinations?
3. **Convergence:** How many annotations are needed for models to reach human-level agreement?
4. **Active Learning:** Which items should be annotated to maximize learning efficiency?

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Generation Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. generate_lexicons.py                                        │
│     ├─ VerbNet verbs (via GlazingAdapter)                      │
│     ├─ Morphological forms (via UniMorphAdapter)               │
│     └─ Controlled lexicons (from resources/ CSVs)              │
│                                                                  │
│  2. generate_templates.py                                       │
│     ├─ Extract all verb-specific VerbNet frames               │
│     ├─ Map to MegaAttitude clausal structures                 │
│     └─ Generate DSL constraints                               │
│                                                                  │
│  3. extract_generic_templates.py                               │
│     └─ Abstract verb-specific → generic structures            │
│                                                                  │
│  4. generate_cross_product.py                                  │
│     └─ Cross all verbs × all generic frames                   │
│                                                                  │
│  5. create_2afc_pairs.py                                       │
│     ├─ Fill templates using MixedFillingStrategy              │
│     ├─ Score with language model (GPT-2)                      │
│     ├─ Create minimal pairs                                   │
│     └─ Stratify by quantile                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Active Learning & Orchestration                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  run_pipeline.py                                               │
│     ├─ Load configuration (config.yaml)                       │
│     ├─ Initialize convergence detector                        │
│     ├─ Run active learning loop                               │
│     ├─ Monitor human-model agreement                          │
│     └─ Stop when converged                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Utility Modules

The `utils/` package provides specialized extraction and generation tools:

**1. verbnet_parser.py** - VerbNet data extraction
- Wraps `GlazingAdapter` for VerbNet access
- Extracts verbs with frame information
- Filters by patterns (clausal, PP, particle verbs)
- Methods: `extract_all_verbs()`, `extract_all_verbs_with_frames()`, `get_clausal_frames()`, `get_frames_with_pp()`

**2. morphology.py** - Morphological paradigm extraction
- Wraps `UniMorphAdapter` for verb inflections
- Extracts: 3sg present, past, present participle, past participle
- Handles particle verbs ("turn off", "cross-examine")
- Creates progressive forms (is/was + V-ing)
- Methods: `get_verb_forms()`, `handle_particle_verb()`, `get_all_required_forms()`

**3. template_generator.py** - Template object generation
- Generates `Template` objects from VerbNet frames
- Maps to MegaAttitude clausal structures
- Applies DSL constraints
- Handles both simple (NP V NP) and complex (NP V that S) frames
- Methods: `generate_from_frame()`, `_generate_clausal_templates()`, `_generate_simple_templates()`

**4. constraint_builder.py** - DSL constraint helpers
- Builds programmatic constraints for slots
- Determiner-noun agreement (a/the/some)
- Bleached lexicon restrictions
- Preposition inventories
- Methods: `build_determiner_constraint()`, `build_bleached_noun_constraint()`, `build_combined_constraint()`

**5. clausal_frames.py** - MegaAttitude frame mapping
- Maps VerbNet patterns to MegaAttitude frames
- 13 frame types (that/whether/to/wh/gerund/etc.)
- Mood tracking (indicative/subjunctive/conditional)
- Methods: `map_verbnet_to_clausal_templates()`, `is_clausal_frame()`, `get_all_clausal_frame_types()`

## Dataset Design

### Full Cross-Product Approach

The pipeline implements a three-stage data generation process:

**Stage 1: Verb-Specific Templates (21,453)**
```bash
python generate_templates.py
```
Extracts verb-specific templates from VerbNet with full frame details, thematic roles, and syntactic patterns.

**Stage 2: Generic Frame Templates (26)**
```bash
python extract_generic_templates.py
```
Abstracts 26 unique structural patterns from the verb-specific templates:
- `{subj} {verb}.` (4,408 verbs use this)
- `{subj} {verb} {obj}.` (3,892 verbs)
- `{subj} {verb} {prep} {obj}.` (2,147 verbs)
- ... 23 more generic patterns

**Stage 3: Cross-Product (124,514 combinations)**
```bash
python generate_cross_product.py
```
Tests **all verbs in all frames**:
```
~4,789 verb lemmas × 26 generic frames = 124,514 total combinations
```

This enables:
- Testing **both grammatical and ungrammatical combinations**
- Discovering **novel acceptable patterns** not in VerbNet
- Training models on **full distributional information**
- Measuring **fine-grained acceptability gradients**

### Controlled Lexicons

To minimize semantic confounds, the pipeline uses **bleached lexicons** for argument filling:

**Bleached Nouns (42 entries):**
- person, people, thing, stuff, place, situation, idea, fact, reason, way, time, day, year, moment, life, world, part, side, point, end, hand, body, head, face, eye, voice, word, name, number, work, money, food, water, air, house, room, door, table, chair, car, book, paper

**Bleached Verbs (8 entries):**
- do, be, have, go, get, make, take, give

**Bleached Adjectives (11 entries):**
- good, bad, big, small, right, wrong, easy, hard, ready, sure, certain

These are loaded from CSV files in `resources/` and converted to Lexicon format.

### 2AFC Judgment Format

Rather than Likert scales, we use **two-alternative forced choice (2AFC)**:

```
Which sentence sounds more natural?

A. The child giggled the toy.
B. The politician contributed the charity.

[ Select A or B ]
```

**Pair Types:**
1. **same_verb:** Same verb, different frames (tests alternations)
2. **different_verb:** Different verbs, same frame (tests verb licensing)

**Advantages:**
- Eliminates scale bias (inter-rater reliability increases)
- Forces relative judgments (more sensitive to gradients)
- Faster annotation (no multi-point scale deliberation)
- Natural task (aligns with linguistic intuitions)

### Stratified Sampling

Items are stratified into **10 quantiles** based on language model score differences:

- **Quantile 1-3:** Easy pairs (large LM score differences)
- **Quantile 4-7:** Medium pairs (moderate differences)
- **Quantile 8-10:** Hard pairs (small differences, near ceiling)

This ensures the model learns across the full difficulty spectrum.

## Quick Start

### Prerequisites

```bash
# Clone repository and navigate to project
cd gallery/eng/argument_structure

# Install dependencies (from repository root)
pip install -e ".[dev,api,training]"
```

### Generate Data

```bash
# See all available commands
make help

# Generate all data files (lexicons, templates, items, pairs)
make data

# Or step-by-step:
make lexicons          # Generate lexicon files from VerbNet + resources
make templates         # Generate verb-specific VerbNet templates
make generic-templates # Extract generic frame structures
make cross-product     # Generate verb × frame cross-product
make 2afc-pairs        # Create 2AFC pairs with LM scoring
```

### Test Pipeline

```bash
# Dry run with test data (no actual training)
make pipeline-dry-run

# View statistics
make show-stats

# View configuration
make show-config
```

### Run Production Pipeline

```bash
# Generate full dataset
make prod-data

# Run complete active learning loop
make prod-pipeline
```

## Detailed Usage

### 1. Generate Lexicons

```bash
python generate_lexicons.py
```

**Output:**
- `lexicons/verbnet_verbs.jsonl` (19,160 entries with morphological features)
- `lexicons/bleached_nouns.jsonl` (42 generic nouns)
- `lexicons/bleached_verbs.jsonl` (8 generic verbs)
- `lexicons/bleached_adjectives.jsonl` (11 generic adjectives)
- `lexicons/determiners.jsonl` (3 determiners)
- `lexicons/prepositions.jsonl` (53 prepositions)

**How it works:**
1. **VerbNet extraction:** Uses `VerbNetExtractor` (wraps GlazingAdapter) to fetch all verbs
2. **Morphology expansion:** Uses `MorphologyExtractor` (wraps UniMorphAdapter) to get inflected forms
3. **CSV loading:** Reads bleached lexicons from `resources/*.csv`
4. **Lexicon creation:** Constructs `Lexicon` objects and saves to JSONL

**Purpose:** Provides lexical items for filling frame templates.

**Test with limited data:**
```bash
python generate_lexicons.py --limit 100
```

### 2. Generate Verb-Specific Templates

```bash
python generate_templates.py
```

**Input:** VerbNet (via GlazingAdapter)
**Output:** `templates/verbnet_frames.jsonl` (21,453 verb-specific templates, 52MB)

**How it works:**
1. **Extract verbs with frames:** `VerbNetExtractor.extract_all_verbs_with_frames()`
2. **For each verb-frame pair:**
   - Map to MegaAttitude clausal structures (if clausal)
   - Generate slot definitions with POS constraints
   - Build DSL constraints for slot fillers
   - Create Template object with metadata
3. **Save to JSONL:** One template per line with UUID

**Example output:**
```json
{
  "name": "think_29.9_that_indicative_past",
  "template_string": "{subj} {verb} that {comp_subj} {comp_verb} {comp_obj}",
  "slots": {
    "subj": {"slot_type": "noun", "constraints": []},
    "verb": {"slot_type": "verb", "constraints": []},
    "comp_subj": {"slot_type": "noun", "constraints": []},
    "comp_verb": {"slot_type": "verb_past", "constraints": []},
    "comp_obj": {"slot_type": "noun", "constraints": []}
  },
  "metadata": {
    "verb_lemma": "think",
    "verbnet_class": "29.9",
    "frame_primary": "NP V that S",
    "frame_type": "finite_that_indicative_past",
    "complementizer": "that",
    "mood": "indicative"
  }
}
```

**Test with limited data:**
```bash
python generate_templates.py --limit 100
```

### 3. Extract Generic Templates

```bash
python extract_generic_templates.py
```

**Input:** `templates/verbnet_frames.jsonl` (21,453 verb-specific templates)
**Output:** `templates/generic_frames.jsonl` (26 unique frame structures)

**How it works:**
1. **Group by template_string:** Collect all templates with same structural pattern
2. **Count verb coverage:** How many verbs use each pattern
3. **Extract metadata:** VerbNet frames, example verbs
4. **Create generic Template:** Remove verb-specific constraints

**Example template:**
```json
{
  "name": "subj_verb_obj",
  "template_string": "{subj} {verb} {obj}.",
  "slots": {
    "subj": {"slot_type": "noun", "constraints": []},
    "verb": {"slot_type": "verb", "constraints": []},
    "obj": {"slot_type": "noun", "constraints": []}
  },
  "metadata": {
    "template_structure": "{subj} {verb} {obj}.",
    "verb_count": 3892,
    "frame_primaries": ["NP V NP", "NP V NP-ATTR", "NP V NP.destination", ...],
    "verbnet_class_count": 187,
    "example_verbs": ["break", "cut", "eat", "hit", "kill", ...]
  }
}
```

### 4. Generate Cross-Product Items

```bash
# Full dataset (all ~4,789 verbs × 26 frames = ~124,514 items)
python generate_cross_product.py

# Test with limited data
python generate_cross_product.py --limit 1000
```

**Output:** `items/cross_product_items.jsonl`

**How it works:**
1. **Load generic templates:** Read 26 frame structures
2. **Load verb lexicon:** Read ~4,789 unique verb lemmas
3. **Generate cross-product:** For each (verb, template) pair:
   - Create Item with verb_lemma + template_id
   - Store metadata for pairing and filling
4. **Save to JSONL:** One item per line

**Example item:**
```json
{
  "item_id": "019a2c04-09c5-71b2-9861-abe1765f1c1a",
  "item_template_id": "019a2bbc-4c41-7b33-befc-248335924f3f",
  "rendered_elements": {
    "template_name": "subj_verb",
    "template_string": "{subj} {verb}.",
    "verb_lemma": "giggle"
  },
  "item_metadata": {
    "verb_lemma": "giggle",
    "template_id": "019a2bbc-4c41-7b33-befc-248335924f3f",
    "template_name": "subj_verb",
    "template_structure": "{subj} {verb}.",
    "combination_type": "verb_frame_cross_product"
  }
}
```

### 5. Create 2AFC Pairs

```bash
# Full dataset (processes all cross-product items)
python create_2afc_pairs.py

# Test with limited data
python create_2afc_pairs.py --limit 200
```

**Output:** `items/2afc_pairs.jsonl`

**Pipeline (8 steps):**

1. **Load cross-product items**
2. **Load templates and lexicons**
3. **Fill templates** using `MixedFillingStrategy`:
   - Verb: Exhaustive (test specific target verb)
   - Nouns: Exhaustive (use all bleached nouns)
   - Adjectives: MLM (BERT contextual filling)
4. **Score with GPT-2:**
   - Compute log probability for each filled sentence
   - Use `HuggingFaceLanguageModel` adapter
   - Cache scores for efficiency
5. **Create minimal pairs:**
   - **same_verb:** Same verb, different frames
   - **different_verb:** Different verbs, same frame
6. **Compute score differences:** `|log_prob1 - log_prob2|`
7. **Assign quantiles:** Bin pairs into 10 difficulty levels
8. **Save as Items:** Convert pairs to Item objects with comparison metadata

**Example pair:**
```json
{
  "item_id": "019a2c05-43ef-7ba0-99c0-8ee3a2eb7a89",
  "item_template_id": "a921ebfd-9650-4f4a-a3c9-7aada5393287",
  "rendered_elements": {
    "option_a": "person giggle.",
    "option_b": "person abash."
  },
  "item_metadata": {
    "pair_type": "different_verb",
    "item1_id": "019a2c04-09c5-71b2-9861-abe1765f1c1a",
    "item2_id": "019a2c04-09c5-71b2-9861-abf8fbe5aad4",
    "lm_score1": -27.59,
    "lm_score2": -30.18,
    "lm_score_diff": 2.59,
    "quantile": 3,
    "template_id": "019a2bbc-4c41-7b33-befc-248335924f3f",
    "template_structure": "{subj} {verb}.",
    "verb1": "giggle",
    "verb2": "abash"
  }
}
```

### 6. Configure Pipeline

Edit `config.yaml` to customize:

```yaml
active_learning:
  strategy: "uncertainty_sampling"    # or "random", "query_by_committee"
  method: "entropy"                   # for uncertainty_sampling
  budget_per_iteration: 200           # items to annotate per round
  max_iterations: 20                  # safety limit

training:
  convergence:
    metric: "krippendorff_alpha"      # or "fleiss_kappa", "cohens_kappa"
    threshold: 0.05                   # stop when |model - human| < 0.05
    min_iterations: 3                 # minimum rounds before stopping

template:
  filling_strategy: "mixed"           # use both MLM and exhaustive
  mlm_model_name: "bert-base-uncased"
  slot_strategies:
    verb: {strategy: "exhaustive"}    # test all verbs
    noun: {strategy: "exhaustive"}    # use all bleached nouns
    adjective: {strategy: "mlm"}      # context-sensitive filling

lists:
  n_lists: 8                          # number of experimental lists
  items_per_list: 100                 # items per list
  quantile_bins: 10                   # stratification bins
  constraints:
    - type: "balance"
      property_expression: "item.metadata.pair_type"
      target_counts: {same_verb: 50, different_verb: 50}
    - type: "uniqueness"
      property_expression: "item.metadata.verb_lemma"
    - type: "grouped_quantile"
      property_expression: "item.metadata.lm_score_diff"
      group_by_expression: "item.metadata.pair_type"
      n_quantiles: 10
```

### 7. Run Pipeline

```bash
# Dry run (test configuration without training)
python run_pipeline.py --dry-run --initial-size 500 --unlabeled-size 1000

# Full run with human ratings
python run_pipeline.py --initial-size 500 --unlabeled-size 2000 \
  --human-ratings data/human_ratings.jsonl

# Custom configuration
python run_pipeline.py --config custom_config.yaml
```

**Pipeline phases (7 steps):**
1. **Load configuration** from YAML
2. **Set up convergence detection** (Krippendorff's alpha tracker)
3. **Set up active learning strategy** (uncertainty sampling)
4. **Load 2AFC pairs** from JSONL
5. **Load human ratings** (if available)
6. **Run active learning loop:**
   - Train model on labeled data
   - Compute entropy for unlabeled items
   - Select top K uncertain items
   - Wait for human annotations
   - Check convergence (|α_model - α_human| < threshold)
   - Repeat until converged or max_iterations
7. **Report results:** Final metrics, convergence status, iteration count

## Active Learning Methodology

### Uncertainty Sampling

The pipeline uses **uncertainty sampling with entropy** as the default acquisition function:

```
H(y|x) = -Σ p(y|x) log p(y|x)
```

At each iteration:
1. Train model on labeled data
2. Compute entropy for all unlabeled items
3. Select top K highest-entropy items
4. Collect human annotations
5. Add to training set and repeat

### Convergence Detection

The pipeline monitors convergence to **human-level inter-annotator agreement** using:

**Krippendorff's Alpha:**
```
α = 1 - (D_observed / D_expected)
```

Where:
- `D_observed`: Disagreement in actual annotations
- `D_expected`: Disagreement expected by chance

**Convergence criterion:**
```
|α_model - α_human| < threshold
```

The model stops when its agreement level matches human agreement (typically α ≈ 0.75-0.85 for acceptability judgments).

**Alternative metrics:**
- **Fleiss' Kappa:** Multi-rater agreement
- **Cohen's Kappa:** Pairwise agreement
- **Accuracy:** Model vs. majority vote

**Implementation:** See `bead/evaluation/convergence.py` for full details.

### Active Learning Loop

```
INITIALIZE:
  Training Set = Initial labeled items (n=500)
  Unlabeled Pool = Remaining items
  Model = Random initialization

LOOP until convergence or max_iterations:
  1. TRAIN model on Training Set
  2. EVALUATE model on held-out human data
  3. COMPUTE inter-annotator agreement metrics:
       - α_human (human-human agreement)
       - α_model (model-human agreement)
  4. CHECK convergence: |α_model - α_human| < threshold
  5. IF converged: STOP
  6. SELECT next batch using uncertainty sampling
  7. COLLECT human annotations for batch
  8. UPDATE Training Set with new annotations

RETURN: Trained model + convergence report
```

## Makefile Targets

The project includes a comprehensive Makefile with 30+ targets:

### Main Targets
```bash
make help                # Show all available targets
make all                 # Run complete pipeline with test data
make data                # Generate all data files
make test                # Run all tests (unit, lint, types)
make clean               # Remove generated files
```

### Data Generation
```bash
make lexicons            # Generate lexicon files
make templates           # Generate verb-specific templates
make generic-templates   # Extract generic frame structures
make cross-product       # Generate verb × frame cross-product
make 2afc-pairs          # Create 2AFC pairs with LM scoring
```

### Pipeline Execution
```bash
make pipeline-dry-run    # Test configuration (no training)
make pipeline            # Run with default settings
make pipeline-full       # Run with realistic settings
```

### Testing
```bash
make test-unit           # Run unit tests
make test-lint           # Run linting (ruff)
make test-types          # Run type checking (pyright)
make check               # Run all checks
```

### Data Inspection
```bash
make show-stats          # Show data statistics
make show-config         # Show pipeline configuration
make show-templates      # Show template samples
make show-pairs          # Show 2AFC pair samples
```

### Development
```bash
make dev-test-small      # Quick test (50 items)
make dev-test-medium     # Medium test (500 items)
```

### Production
```bash
make prod-data           # Generate full dataset (124,514 items)
make prod-pipeline       # Run full active learning loop
```

### Cleaning
```bash
make clean-items         # Remove generated items
make clean-cache         # Remove model cache
make clean-all           # Remove everything (including lexicons)
```

## Convergence Detection Details

### Human Agreement Baseline

Human inter-annotator agreement is computed from double-annotated items:

```python
from bead.evaluation.interannotator import compute_interannotator_agreement

human_agreement = compute_interannotator_agreement(
    annotations_1=annotator1_labels,
    annotations_2=annotator2_labels,
    metric="krippendorff_alpha",
    data_type="nominal"
)
```

### Model Agreement

Model-human agreement is computed by treating the model as a "virtual annotator":

```python
from bead.evaluation.convergence import ConvergenceDetector

detector = ConvergenceDetector(
    human_agreement_metric="krippendorff_alpha",
    convergence_threshold=0.05,
    min_iterations=3,
    alpha=0.05
)

# Each iteration
result = detector.check_convergence(
    model_metadata=model,
    human_annotations=annotations,
    predicted_labels=predictions,
    human_agreement_scores=[0.78, 0.81, 0.79]  # from previous rounds
)

if result.has_converged:
    print(f"Converged at iteration {result.iteration}")
    print(f"Model agreement: {result.model_agreement:.3f}")
    print(f"Human agreement: {result.human_agreement:.3f}")
```

### Stopping Criteria

The pipeline stops when **any** of these conditions is met:

1. **Convergence:** `|α_model - α_human| < 0.05` for ≥3 consecutive iterations
2. **Max iterations:** Reached `max_iterations` (default: 20)
3. **Performance threshold:** Model accuracy exceeds threshold (optional)
4. **Budget exhausted:** No more unlabeled items available

## Replication Instructions

### Full Experiment Replication

**1. Generate complete dataset:**

```bash
# Generate all lexicons and templates
make lexicons templates generic-templates

# Generate full cross-product (124,514 items)
python generate_cross_product.py

# Create 2AFC pairs (stratified sampling)
python create_2afc_pairs.py
```

**Expected output:**
- `items/cross_product_items.jsonl` (~76 MB, 124,514 items)
- `items/2afc_pairs.jsonl` (~200 MB, variable based on pairing strategy)

**2. Partition into experimental lists:**

Lists are automatically created by `run_pipeline.py` using constraints from `config.yaml`.

**3. Deploy to JATOS:**

```bash
# (Future phase - deployment integration)
# Will generate jsPsych experiments for each list
# Upload to JATOS server
# Distribute links to participants
```

**4. Collect human ratings:**

Participants see 2AFC trials:

```
Trial 1:
  Which sentence sounds more natural?
  A. The child giggled the toy.
  B. The politician contributed the charity.
  [Select A or B]

Trial 2:
  ...
```

**5. Run active learning pipeline:**

```bash
python run_pipeline.py \
  --initial-size 500 \
  --unlabeled-size 124014 \
  --human-ratings data/human_ratings_batch1.jsonl
```

**6. Iterate until convergence:**

The pipeline will:
- Train model on initial 500 items
- Select 200 highest-uncertainty items
- Wait for human annotations (deploy new batch)
- Add to training set and retrain
- Repeat until `|α_model - α_human| < 0.05`

**Expected convergence:** 5-10 iterations (~1,500-2,500 annotations total)

### Computational Requirements

**Data generation:**
- Time: ~2-4 hours (full cross-product + 2AFC pairs)
- Memory: ~8 GB RAM
- Storage: ~350 MB (all data files)

**Active learning loop (per iteration):**
- Time: ~10-30 minutes (model training + evaluation)
- Memory: ~16 GB RAM (with BERT-based models)
- GPU: Optional but recommended (10x speedup)

**Human data collection:**
- Participants: ~50-100 (depending on list design)
- Time per participant: ~30-45 minutes
- Total annotations needed: ~1,500-2,500 for convergence

## Data Format Documentation

### Cross-Product Items

```jsonl
{"item_id": "uuid", "item_template_id": "uuid", "rendered_elements": {...}, "item_metadata": {...}}
```

**Fields:**
- `item_id`: Unique identifier (UUID)
- `item_template_id`: Reference to template UUID
- `rendered_elements`: Human-readable data
  - `template_name`: Frame name (e.g., "subj_verb_obj")
  - `template_string`: Slot structure
  - `verb_lemma`: Target verb
- `item_metadata`: Machine-readable data
  - `verb_lemma`: Target verb
  - `template_id`: Template UUID
  - `template_structure`: Slot structure
  - `combination_type`: Always "verb_frame_cross_product"

### 2AFC Pairs

```jsonl
{"item_id": "uuid", "item_template_id": "comparison_2afc", "rendered_elements": {...}, "item_metadata": {...}}
```

**Fields:**
- `rendered_elements`:
  - `option_a`: First sentence
  - `option_b`: Second sentence
- `item_metadata`:
  - `pair_type`: "same_verb" or "different_verb"
  - `item1_id`, `item2_id`: Original item UUIDs
  - `verb1`, `verb2`: Verb lemmas
  - `template_id`: Shared template UUID
  - `template_structure`: Slot structure
  - `lm_score1`, `lm_score2`: GPT-2 log probabilities
  - `lm_score_diff`: |score1 - score2|
  - `quantile`: Stratification bin (1-10)

### Human Ratings

```jsonl
{"item_id": "pair_uuid", "participant_id": "string", "response": "a" or "b", "timestamp": "iso8601", ...}
```

**Fields:**
- `item_id`: Reference to 2AFC pair UUID
- `participant_id`: Anonymous participant identifier
- `response`: "a" (chose option_a) or "b" (chose option_b)
- `timestamp`: ISO 8601 datetime
- `reaction_time`: Milliseconds (optional)
- `list_id`: Experimental list identifier

## Project Structure

```
gallery/eng/argument_structure/
├── README.md                      # This file
├── PROGRESS.md                    # Development log (598 lines)
├── Makefile                       # Build automation (380+ lines, 30+ targets)
├── config.yaml                    # Pipeline configuration (399 lines)
│
├── generate_lexicons.py           # Extract VerbNet verbs + bleached lexicons
├── generate_templates.py          # Generate verb-specific VerbNet templates
├── extract_generic_templates.py   # Extract 26 generic frame structures
├── generate_cross_product.py      # Generate verb × frame items
├── create_2afc_pairs.py           # Create 2AFC pairs with LM scoring
├── run_pipeline.py                # Active learning orchestration (498 lines)
│
├── utils/                         # Utility modules (63KB total)
│   ├── __init__.py                # Package initialization
│   ├── verbnet_parser.py          # VerbNet extraction (9KB)
│   ├── morphology.py              # Morphological paradigms (9KB)
│   ├── template_generator.py     # Template generation (21KB)
│   ├── constraint_builder.py     # DSL constraint helpers (9KB)
│   └── clausal_frames.py         # MegaAttitude frame mapping (13KB)
│
├── resources/                     # Reference data
│   ├── README.md                  # Resource documentation
│   ├── bleached_nouns.csv         # 42 controlled nouns
│   ├── bleached_verbs.csv         # 8 controlled verbs
│   └── bleached_adjectives.csv    # 11 controlled adjectives
│
├── lexicons/                      # Generated lexical resources (8.5MB)
│   ├── verbnet_verbs.jsonl        # 19,160 verb forms
│   ├── bleached_nouns.jsonl       # 42 generic nouns
│   ├── bleached_verbs.jsonl       # 8 generic verbs
│   ├── bleached_adjectives.jsonl  # 11 generic adjectives
│   ├── determiners.jsonl          # 3 determiners
│   └── prepositions.jsonl         # 53 prepositions
│
├── templates/                     # Frame templates (52MB)
│   ├── verbnet_frames.jsonl       # 21,453 verb-specific templates
│   └── generic_frames.jsonl       # 26 generic frame structures
│
├── items/                         # Generated experimental items (15MB)
│   ├── cross_product_items.jsonl  # Verb × frame combinations
│   └── 2afc_pairs.jsonl           # Paired comparisons
│
├── lists/                         # Experimental list partitions (TBD)
├── data/                          # Human ratings (TBD)
└── .cache/                        # Model output cache (200+ files)
```

## Dependencies

**Core:**
- Python ≥3.10
- bead (parent library)
  - `sash.resources.adapters.glazing` - VerbNet access
  - `sash.resources.adapters.unimorph` - Morphology access
  - `sash.resources.lexicon` - Lexicon management
  - `sash.resources.structures` - Template/Slot/Constraint models
  - `sash.items.models` - Item models
  - `sash.items.adapters.huggingface` - Language model scoring
  - `sash.evaluation.convergence` - Convergence detection
  - `sash.training.active_learning` - Active learning loop
- transformers (Hugging Face)
- torch (PyTorch)
- pydantic (data validation)
- pyyaml (configuration)

**Optional:**
- anthropic (Claude API)
- google-generativeai (Gemini API)
- openai (GPT-4 API)
- pytorch-lightning (training)
- tensorboard (monitoring)

**Development:**
- pytest (testing)
- ruff (linting)
- pyright (type checking)

## Citation

If you use this pipeline or dataset in your research, please cite:

```bibtex
@misc{argument_structure_active_learning,
  title={Argument Structure Active Learning with Convergence Detection},
  author={White, Aaron Steven},
  year={2025},
  url={https://github.com/aaronstevenwhite/bead/gallery/eng/argument_structure}
}
```

**Related work:**

- **VerbNet:** Kipper, K., Korhonen, A., Ryant, N., & Palmer, M. (2008). A large-scale classification of English verbs. *Language Resources and Evaluation*, 42(1), 21-40.

- **Glazing:** White, A. S., & Rawlins, K. (2024). Glazing: A unified interface for lexical resources. *GitHub repository*.

- **Active Learning:** Settles, B. (2009). Active learning literature survey. *Computer Sciences Technical Report 1648*, University of Wisconsin-Madison.

- **Convergence Detection:** Krippendorff, K. (2004). Reliability in content analysis: Some common misconceptions and recommendations. *Human Communication Research*, 30(3), 411-433.

- **MegaAttitude:** White, A. S., & Rawlins, K. (2018). The role of veridicality and factivity in clause selection. *Proceedings of SALT*, 28, 573-593.

## License

This project is part of the SASH (Structured Acceptability Study Helper) framework.

[Add appropriate license information]

## Contact

For questions, issues, or contributions:

- GitHub Issues: [repository URL]
- Email: aswhite@rochester.edu
- Documentation: [docs URL]

## Acknowledgments

This project builds on:

- **VerbNet** lexical resource (Kipper et al., 2008)
- **Glazing** unified lexical resource interface (White & Rawlins, 2024)
- **UniMorph** morphological paradigms (Kirov et al., 2018)
- **MegaAttitude** clause-embedding frame inventory (White & Rawlins, 2018)
- **SASH** framework for acceptability studies
- **Hugging Face Transformers** for language models
- **PyTorch** for deep learning

---

**Status:** ✅ Ready for production deployment

**Last Updated:** October 28, 2025

**Current Data:** 21,453 verb-specific templates, 26 generic frames, 19,160 verb forms, test dataset with 500 cross-product items and 19,900 2AFC pairs
