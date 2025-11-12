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

The pipeline consists of 10 main scripts organized into 4 stages:

```
┌─────────────────────────────────────────────────────────────────┐
│              Stage 1: Resource Generation                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. generate_lexicons.py                                        │
│     ├─ VerbNet verbs (via GlazingAdapter)                      │
│     ├─ Morphological forms (via UniMorphAdapter)               │
│     └─ Controlled lexicons (from resources/ CSVs)              │
│     → Output: lexicons/*.jsonl (19,160+ entries)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│          Stage 2: Template Generation & Filling                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2. generate_templates.py                                       │
│     ├─ Extract all verb-specific VerbNet frames               │
│     ├─ Map to MegaAttitude clausal structures                 │
│     └─ Generate DSL constraints                               │
│     → Output: templates/verbnet_frames.jsonl (21,453 templates)│
│                                                                  │
│  3. extract_generic_templates.py                               │
│     └─ Abstract verb-specific → generic structures            │
│     → Output: templates/generic_frames.jsonl (26 templates)    │
│                                                                  │
│  4. fill_templates.py                                          │
│     ├─ Fill templates using MixedFillingStrategy              │
│     ├─ Phase 1: Exhaustive filling (det, be, verb slots)      │
│     └─ Phase 2: MLM-based filling (noun, prep, adj slots)     │
│     → Output: filled_templates/generic_frames_filled.jsonl     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│        Stage 3: Item Generation & List Partitioning              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  5. generate_cross_product.py                                  │
│     └─ Cross all verbs × all generic frames                   │
│     → Output: items/cross_product_items.jsonl (~74,880 items)  │
│                                                                  │
│  6. create_2afc_pairs.py                                       │
│     ├─ Load filled templates from previous step               │
│     ├─ Score with language model (GPT-2)                      │
│     ├─ Create minimal pairs (same_verb, different_verb)       │
│     └─ Stratify by LM score quantiles                         │
│     → Output: items/2afc_pairs.jsonl                           │
│                                                                  │
│  7. generate_lists.py                                          │
│     ├─ Partition 2AFC pairs into balanced lists               │
│     ├─ Apply list constraints (balance, uniqueness, etc.)     │
│     └─ Apply batch constraints (coverage, diversity)          │
│     → Output: lists/experiment_lists.jsonl                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│      Stage 4: Deployment & Active Learning                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  8. generate_deployment.py                                     │
│     ├─ Generate jsPsych experiments (local + JATOS versions)  │
│     ├─ Local: Standalone for testing (no server required)     │
│     ├─ JATOS: Production deployment with Prolific support     │
│     └─ Create JATOS .jzip packages                            │
│     → Output: deployment/local/* + deployment/jatos/*          │
│                                                                  │
│  9. simulate_pipeline.py (testing/validation)                  │
│     ├─ Simulate human judgments (LM-based annotator)          │
│     ├─ Test active learning loop                              │
│     └─ Validate convergence detection                         │
│     → Output: simulation_output/simulation_results.json        │
│                                                                  │
│  10. run_pipeline.py (production)                              │
│      ├─ Load configuration (config.yaml)                      │
│      ├─ Initialize convergence detector                       │
│      ├─ Run active learning loop                              │
│      ├─ Monitor human-model agreement                         │
│      └─ Stop when converged to human IAA                      │
│      → Output: results/pipeline_results.json                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Utility Modules

The `utils/` package provides specialized extraction and generation tools. The `verbnet_parser.py` module wraps `GlazingAdapter` to extract verbs with frame information from VerbNet, filtering by patterns like clausal complements, PPs, and particle verbs. For morphology, `morphology.py` wraps `UniMorphAdapter` to get verb inflections (3sg present, past, participles) and handles particle verbs like "turn off" or "cross-examine", including progressive forms.

Template generation happens in `template_generator.py`, which creates `Template` objects from VerbNet frames and maps them to MegaAttitude clausal structures. It handles both simple frames (NP V NP) and complex ones (NP V that S) with DSL constraints. The `constraint_builder.py` module builds programmatic constraints for slots, handling determiner-noun agreement and bleached lexicon restrictions.

Finally, `clausal_frames.py` maps VerbNet patterns to MegaAttitude's 13 frame types (that/whether/to/wh/gerund/etc.) with mood tracking for indicative, subjunctive, and conditional complements.

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

To minimize semantic confounds, the pipeline uses bleached lexicons for argument filling. The noun inventory (41 nouns) includes generic, high-frequency items organized by semantic class: animates (person, people, group, organization), inanimate objects (thing, things, stuff), locations (place, places, area, areas), temporals (time, times), abstracts (information, idea, ideas, reason, reasons, matter, matters, situation, situations, way, ways, level, levels, event, events, activity, activities, amount, amounts, part, parts), plus other-marked variants (other person, other people, other thing, etc.).

For verbs, we use 8 light verbs (do, be, have, go, get, make, happen, come), and for adjectives, 11 basic descriptors (good, bad, right, wrong, okay, certain, ready, done, different, same, other). These are loaded from CSV files in `resources/` and converted to Lexicon format.

### 2AFC Judgment Format

Rather than Likert scales, we use **two-alternative forced choice (2AFC)**:

```
Which sentence sounds more natural?

A. The child giggled the toy.
B. The politician contributed the charity.

[ Select A or B ]
```

We create two types of pairs: same_verb pairs test alternations by using the same verb in different frames, while different_verb pairs test verb licensing by comparing different verbs in the same frame.

This format has several advantages over Likert scales. It eliminates scale bias and increases inter-rater reliability, forces relative rather than absolute judgments (making it more sensitive to acceptability gradients), speeds up annotation by removing multi-point scale deliberation, and feels like a natural task that aligns with linguistic intuitions.

### Stratified Sampling

Items are stratified into 10 quantiles based on language model score differences. Quantiles 1-3 contain easy pairs with large LM score differences, quantiles 4-7 have medium pairs with moderate differences, and quantiles 8-10 include hard pairs with small differences near the ceiling. This ensures the model learns across the full difficulty spectrum.

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
make lexicons           # 1. Generate lexicons from VerbNet + resources
make verbnet-templates  # 2. Generate verb-specific VerbNet templates
make templates          # 3. Extract generic frame structures
make fill-templates     # 4. Fill templates with MLM strategy
make cross-product      # 5. Generate verb × frame cross-product
make 2afc-pairs         # 6. Create 2AFC pairs with LM scoring
make lists              # 7. Partition pairs into experiment lists
make deployment         # 8. Generate jsPsych/JATOS deployment
```

### Test Pipeline

```bash
# Test with simulation (no real participants needed)
make simulate-quick     # Quick test (~2 minutes)
make simulate-medium    # Medium test (~10 minutes)

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

# Deploy to JATOS
make deployment-full

# Run complete active learning loop (with human data)
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

### 4. Fill Templates

```bash
# Dry run (fills 1 template with 5 verbs)
python fill_templates.py --dry-run

# Fill all templates with MLM strategy
python fill_templates.py

# Test with limited data
python fill_templates.py --limit 10
```

**Input:** `templates/generic_frames.jsonl` + lexicons
**Output:** `filled_templates/generic_frames_filled.jsonl`

This script loads generic templates and lexicons, then creates a TemplateFiller with MixedFillingStrategy using slot_strategies from `config.yaml`. The strategy operates in two phases:

**Phase 1 (Exhaustive):** Generates all combinations of determiners (a, the, some), be forms (am, is, are, was, were), and verb forms, applying cross-slot constraints (e.g., subject-verb agreement). This produces a set of partial templates with Phase 1 slots filled.

**Phase 2 (MLM):** For each Phase 1 combination, uses BERT to predict contextually appropriate fillers for noun, preposition, and adjective slots via beam search. The MLM sees correctly inflected forms (e.g., "the [MASK] is [MASK]" not "the [MASK] be [MASK]"), enabling accurate predictions.

Each template gets filled with all valid combinations of slot fillers, producing fully rendered sentences. The output file contains FilledTemplate objects with slot_fillers, rendered_text, and strategy metadata.

The filling strategy is configured per-slot in `config.yaml`:
```yaml
template:
  filling_strategy: "mixed"
  mlm:
    model_name: "bert-base-uncased"
    beam_size: 5
    top_k: 10
  slot_strategies:
    # Phase 1: Exhaustive filling
    det_subj: {strategy: "exhaustive"}
    det_dobj: {strategy: "exhaustive"}
    det_pobj: {strategy: "exhaustive"}
    be: {strategy: "exhaustive"}
    verb: {strategy: "exhaustive"}

    # Phase 2: MLM filling
    noun_subj: {strategy: "mlm", max_fills: 5}
    noun_dobj: {strategy: "mlm", max_fills: 5}
    noun_pobj: {strategy: "mlm", max_fills: 5}
    prep: {strategy: "mlm", max_fills: 5}
    adjective: {strategy: "mlm", max_fills: 3}
```

### 5. Generate Cross-Product Items

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

### 6. Create 2AFC Pairs

```bash
# Full dataset (processes all filled templates)
python create_2afc_pairs.py

# Test with limited data
python create_2afc_pairs.py --limit 200
```

**Input:** `filled_templates/generic_frames_filled.jsonl`
**Output:** `items/2afc_pairs.jsonl`

This script loads filled templates from the previous step, scores each filled sentence with a language model (GPT-2), and creates forced-choice pairs. The scoring uses `LanguageModelScorer` to compute log probabilities with caching for efficiency. Pairs are created using `create_forced_choice_items_from_groups`, generating both same_verb pairs (same verb in different frames, testing alternations) and different_verb pairs (different verbs in same frame, testing verb licensing). After pairing, score differences are computed and items are stratified into 10 quantiles using `assign_quantiles_by_uuid`, ensuring the model learns across the full difficulty spectrum.

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

### 7. Generate Experiment Lists

```bash
# Generate balanced experiment lists
python generate_lists.py
```

**Input:** `items/2afc_pairs.jsonl` + `config.yaml`
**Output:** `lists/experiment_lists.jsonl`

**How it works:**
1. **Load 2AFC pairs:** Read all generated pairs
2. **Load configuration:** Parse list and batch constraints from `config.yaml`
3. **Build constraints:**
   - **List constraints:** Applied to each list individually (balance, uniqueness, diversity)
   - **Batch constraints:** Applied across all lists (coverage, min occurrence)
4. **Partition items:** Use `ListPartitioner` with constraint satisfaction
5. **Create ListCollection:** Package lists with metadata
6. **Save to JSONL:** One ExperimentList per line

**Example constraints (config.yaml):**
```yaml
lists:
  n_lists: 8
  items_per_list: 100
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
  batch_constraints:
    - type: "coverage"
      property_expression: "item.metadata.template_id"
      target_values: [all 26 template UUIDs]
      min_coverage: 1.0
```

**Output format:**
```json
{
  "id": "uuid",
  "name": "list_01",
  "item_refs": ["item_uuid_1", "item_uuid_2", ...],
  "metadata": {
    "n_items": 100,
    "constraints_satisfied": true
  }
}
```

### 8. Generate Deployment

```bash
# Generate deployment for 2 lists (testing)
python generate_deployment.py --n-lists 2

# Generate deployment for all lists
python generate_deployment.py --n-lists 20

# Generate without JATOS export
python generate_deployment.py --n-lists 2 --no-jatos
```

**Input:** `lists/experiment_lists.jsonl` + `items/2afc_pairs.jsonl`
**Output:** Dual deployment versions (local + JATOS) + `.jzip` files

**How it works:**
1. **Load experiment lists:** Read lists and randomly select subset
2. **Load 2AFC pairs:** Index all items by UUID for lookup
3. **Create ItemTemplate:** Minimal template for 2AFC forced choice
4. **Generate TWO versions of jsPsych experiments:**
   - **Local version** (`deployment/local/`) - Standalone for testing
   - **JATOS version** (`deployment/jatos/`) - Production deployment
5. **Export to JATOS:** Create `.jzip` packages from JATOS version
6. **Save output:** HTML/CSS/JS files in deployment/ directory

**Output structure:**
```
deployment/
├── local/                      # Standalone version (open directly in browser)
│   ├── list_01/
│   │   ├── index.html          # jsPsych experiment (no JATOS dependencies)
│   │   ├── css/experiment.css  # Minimal custom styles
│   │   ├── js/experiment.js    # Standalone experiment logic
│   │   └── data/config.json    # Experiment metadata
│   ├── list_02/
│   │   └── ...
│
├── jatos/                      # JATOS-integrated version
│   ├── list_01/
│   │   ├── index.html          # jsPsych experiment (with JATOS integration)
│   │   ├── css/experiment.css  # Minimal custom styles
│   │   ├── js/experiment.js    # JATOS-aware experiment logic
│   │   └── data/config.json    # Experiment metadata
│   ├── list_02/
│   │   └── ...
│   ├── list_01.jzip           # JATOS package for list 1
│   └── list_02.jzip           # JATOS package for list 2
```

**Key Features:**

**Local Version:**
- Runs directly in browser without server
- No JATOS dependencies
- Useful for testing and debugging
- Data stored in browser console (jsPsych.data.get())
- Trial randomization with simple shuffle

**JATOS Version:**
- Full JATOS integration for production deployment
- Automatic participant ID capture from URL parameters
- Prolific integration support (PROLIFIC_PID, STUDY_ID, SESSION_ID)
- Data submission to JATOS server with error handling
- Abort button on each trial (via `jatos.addAbortButton`)
- Worker and study result ID tracking
- Configurable Prolific completion redirect
- Trial randomization with constraint satisfaction

**Deployment Options:**

**Option 1: Testing Locally**
```bash
# Open local version in browser
open deployment/local/list_01/index.html

# No server required, no JATOS errors
```

**Option 2: Production via JATOS**
```bash
# Upload JATOS packages to your JATOS server
# 1. Go to your JATOS server admin panel
# 2. Click "Import Study"
# 3. Upload deployment/jatos/*.jzip files
# 4. Configure Prolific integration (optional)
# 5. Distribute experiment URLs to participants
```

**Prolific Integration:**

To deploy via JATOS on Prolific, configure your completion code in `config.yaml`:

```yaml
deployment:
  jspsych:
    prolific_completion_code: "YOUR_CODE_HERE"  # e.g., "C1A2B3C4"
```

This automatically generates the redirect URL: `https://app.prolific.co/submissions/complete?cc=YOUR_CODE_HERE`

Participant metadata captured:
- `PROLIFIC_PID`: Prolific participant ID
- `STUDY_ID`: Prolific study ID
- `SESSION_ID`: Prolific session ID
- `jatos_worker_id`: JATOS worker ID
- `jatos_study_result_id`: JATOS study result ID
- `jatos_component_result_id`: JATOS component result ID

**Data Collection:**

1. **JATOS deployment:** Results automatically submitted to JATOS server
2. **Export data:** Use JATOS admin panel to export results as JSON
3. **Format:** Results include all metadata + participant responses
4. **Analysis:** Load exported JSON into pipeline for active learning iteration

### 9. Simulate Pipeline (Testing)

```bash
# Quick simulation (50 items, 3 iterations)
python simulate_pipeline.py \
  --initial-size 30 \
  --budget 10 \
  --max-iterations 3 \
  --temperature 1.0 \
  --seed 42 \
  --max-items 50

# Or use Makefile
make simulate-quick    # ~2 minutes
make simulate-medium   # ~10 minutes
make simulate-full     # ~30 minutes
```

**Purpose:** Test the complete active learning pipeline with simulated human judgments before deploying to real participants.

**How it works:**
1. **Load 2AFC pairs:** Sample subset for testing
2. **Create simulated annotator:**
   - Uses LM scores from items to generate probabilistic judgments
   - Adds temperature-based noise to simulate human variability
   - Configurable noise levels (temperature parameter)
3. **Generate initial annotations:** Simulate human ratings for initial set
4. **Run active learning loop:**
   - Train ForcedChoiceModel on labeled data
   - Compute uncertainty (entropy) for unlabeled items
   - Select most uncertain items
   - Simulate human annotations for selected items
   - Repeat until convergence or max iterations
5. **Monitor convergence:** Check if model accuracy approaches simulated human agreement
6. **Save results:** JSON file with iteration metrics

**Example output:**
```json
{
  "config": {
    "initial_size": 50,
    "budget_per_iteration": 20,
    "temperature": 1.0
  },
  "human_agreement": 0.782,
  "iterations": [
    {"iteration": 1, "train_accuracy": 0.652, "test_accuracy": 0.640},
    {"iteration": 2, "train_accuracy": 0.701, "test_accuracy": 0.685},
    ...
  ],
  "converged": true,
  "total_annotations": 150
}
```

The temperature parameter controls noise levels: 0.5 produces low noise with clean judgments and faster convergence, 1.0 gives medium noise with realistic judgments, and 2.0 creates high noise with noisier judgments and slower convergence.

### 10. Run Production Pipeline

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

### 11. Run Production Pipeline

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
make verbnet-templates   # Generate verb-specific VerbNet templates
make templates           # Extract generic frame structures
make fill-templates      # Fill templates with MLM strategy (optional)
make cross-product       # Generate verb × frame cross-product
make 2afc-pairs          # Create 2AFC pairs with LM scoring
make lists               # Partition pairs into experiment lists
make deployment          # Generate jsPsych/JATOS deployment
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

The pipeline stops when any of these conditions is met: convergence (`|α_model - α_human| < 0.05` for ≥3 consecutive iterations), reaching max_iterations (default: 20), exceeding an optional performance threshold, or exhausting the budget of unlabeled items.

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

```bash
# Generate balanced experiment lists
python generate_lists.py
```

This creates `lists/experiment_lists.jsonl` with balanced lists according to constraints in `config.yaml`.

**3. Deploy to JATOS:**

```bash
# Generate jsPsych experiments and JATOS packages
python generate_deployment.py --n-lists 20

# Upload to JATOS server
# 1. Go to your JATOS server admin panel
# 2. Click "Import Study"
# 3. Upload deployment/*.jzip files
# 4. Distribute experiment URLs to participants
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
├── README.md                       # This file
├── PROGRESS.md                     # Development log
├── Makefile                        # Build automation (500+ lines, 40+ targets)
├── config.yaml                     # Pipeline configuration
│
├── generate_lexicons.py            # [1] Extract VerbNet verbs + bleached lexicons
├── generate_templates.py           # [2] Generate verb-specific VerbNet templates
├── extract_generic_templates.py    # [3] Extract 26 generic frame structures
├── fill_templates.py               # [4] Fill templates with MLM strategy (optional)
├── generate_cross_product.py       # [5] Generate verb × frame cross-product
├── create_2afc_pairs.py            # [6] Create 2AFC pairs with LM scoring
├── generate_lists.py               # [7] Partition pairs into experiment lists
├── generate_deployment.py          # [8] Generate jsPsych/JATOS deployment
├── simulate_pipeline.py            # [9] Simulate active learning (testing)
├── run_pipeline.py                 # [10] Run production active learning pipeline
│
├── utils/                          # Utility modules
│   ├── __init__.py                 # Package initialization
│   ├── verbnet_parser.py           # VerbNet extraction via GlazingAdapter
│   ├── morphology.py               # Morphological paradigms via UniMorphAdapter
│   ├── template_generator.py      # Template generation with DSL constraints
│   ├── constraint_builder.py      # DSL constraint helpers
│   └── clausal_frames.py          # MegaAttitude frame mapping
│
├── tests/                          # Test suite
│   ├── __init__.py
│   └── test_simulation.py         # Simulation tests
│
├── resources/                      # Reference data
│   ├── README.md                   # Resource documentation
│   ├── bleached_nouns.csv          # 42 controlled nouns
│   ├── bleached_verbs.csv          # 8 controlled verbs
│   └── bleached_adjectives.csv     # 11 controlled adjectives
│
├── lexicons/                       # Generated lexical resources
│   ├── verbnet_verbs.jsonl         # 19,160 verb forms
│   ├── bleached_nouns.jsonl        # 42 generic nouns
│   ├── bleached_verbs.jsonl        # 8 generic verbs
│   ├── bleached_adjectives.jsonl   # 11 generic adjectives
│   ├── determiners.jsonl           # 3 determiners
│   ├── prepositions.jsonl          # 53 prepositions
│   └── be_forms.jsonl              # Auxiliary "be" forms
│
├── templates/                      # Frame templates
│   ├── verbnet_frames.jsonl        # 21,453 verb-specific templates (gitignored)
│   └── generic_frames.jsonl        # 26 generic frame structures (checked in)
│
├── filled_templates/               # Filled templates
│   └── generic_frames_filled.jsonl # Templates with slot fillers (required for 2AFC)
│
├── items/                          # Generated experimental items
│   ├── cross_product_items.jsonl   # Verb × frame combinations (~74,880 items)
│   └── 2afc_pairs.jsonl            # Paired comparisons for judgments
│
├── lists/                          # Experimental list partitions
│   └── experiment_lists.jsonl      # Balanced lists with constraints
│
├── deployment/                     # jsPsych/JATOS deployment (gitignored)
│   ├── local/                      # Standalone version for testing
│   │   ├── list_01/                # Experiment for list 1
│   │   │   ├── index.html          # jsPsych experiment (no JATOS)
│   │   │   ├── css/experiment.css  # Minimal custom styles
│   │   │   ├── js/experiment.js    # Standalone experiment logic
│   │   │   └── data/config.json    # Metadata
│   │   ├── list_02/                # Experiment for list 2
│   │   │   └── ...
│   │
│   └── jatos/                      # JATOS-integrated version
│       ├── list_01/                # Experiment for list 1
│       │   ├── index.html          # jsPsych experiment (with JATOS)
│       │   ├── css/experiment.css  # Minimal custom styles
│       │   ├── js/experiment.js    # JATOS-aware experiment logic
│       │   └── data/config.json    # Metadata
│       ├── list_02/                # Experiment for list 2
│       │   └── ...
│       ├── list_01.jzip            # JATOS package for list 1
│       └── list_02.jzip            # JATOS package for list 2
│
├── simulation_output/              # Simulation results
│   └── simulation_results.json     # Convergence metrics
│
├── results/                        # Pipeline results
│   └── pipeline_results.json       # Active learning metrics
│
├── data/                           # Human ratings (from JATOS)
│   └── human_ratings.jsonl         # Participant responses
│
└── .cache/                         # Model output cache
    └── ...                         # Cached LM scores
```

