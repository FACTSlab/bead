# Korean Argument Structure

A pipeline for collecting two-alternative forced-choice (2AFC) acceptability judgments on Korean verb × frame combinations, grammatical and ungrammatical alike. It mirrors `gallery/eng/argument_structure`, which serves as the coverage baseline.

## Overview

The pipeline:

1. **Builds lexicons** from UniMorph Korean (verbs) and hand-built resources (bleached nouns, case markers, postpositions)
2. **Generates frames**, abstract sentence schemas with case-marked noun slots and a verb slot
3. **Scores concrete templates** (frames filled except for the verb) with a masked language model, on a sample from each frame class × aspect cell
4. **Selects concrete templates** evenly across score bins, keeping odd ones as well as natural ones
5. **Builds 2AFC pairs** by crossing selected concrete templates with lexicon verbs
6. **Partitions pairs** into balanced experiment lists
7. **Deploys** the lists as Korean-language jsPsych/JATOS experiments

**Key feature:** like the English gallery, every verb can meet every frame, so participants judge unlicensed combinations as well as licensed ones. The full Korean cross-product is about 14.5 billion sentences, so instead of filling it exhaustively the pipeline samples concrete templates and pairs them with verbs.

## Linguistic Background

### Case marking and word order

Korean is verb-final, and arguments carry case particles instead of relying on position:

```
사람이   물건을   옮겼다.          person-NOM thing-ACC moved
사람이   장소에서 지역으로 갔다.   person-NOM place-SRC area-DIR went
```

A frame therefore fixes the particles (`NP-NOM NP-ACC ___`), while English varies a preposition inside one frame. Particles also agree with the noun's final sound: 이/가, 을/를, 과/와, and 으로/로, where ㄹ-final nouns take 로 (물질로).

### Frames and concrete templates

| Term | Meaning | Example |
|------|---------|---------|
| Frame | Abstract variant: noun and particle slots plus a verb slot | `{noun_subj}{nom} {noun_dobj}{acc} {verb}.` (`NP-NOM NP-ACC ___`) |
| Frame class | Frames grouped by argument shape, with obliques abstracted over case | `NP_V_NP_OBL` |
| Concrete template | A frame with everything filled except the verb and its form | `사람이 물건을 ___` |

### Alternations

English gets the causative/inchoative alternation from labile verbs (*John broke the window* / *The window broke*). Korean marks it on the verb (깨다 / 깨지다, 녹다 / 녹이다), so it needs derived verb forms. These aren't covered yet (see [Known Limitations](#known-limitations)).

## Dataset Design

### Sampled Cross-Product

| | English | Korean |
|---|---|---|
| Verbs | 3,475 VerbNet lemmas | 1,988 UniMorph verbs (predicate adjectives dropped) |
| Frames | 78 (26 base + 52 progressive) | 138 (46 base + 92 progressive) |
| Items | Exhaustive fill, then pairs | 6,057 concrete templates scored → 504 selected → 1,008 pairs (504 × 2 pair types) |

### 2AFC Judgment Format

Each item shows two sentences, and participants pick the more natural one (어느 문장이 더 자연스러운가요?). There are two pair types:

- **`different_verb`**: same concrete template, two different verbs — `상황이 문제를 헤맸다.` vs. `상황이 문제를 포함됐다.`
- **`same_verb`**: same verb, two frames of the same aspect — `방법이 문제부터 모으고 있었다.` vs. `방법이 양을 모으고 있었다.`

### Sampling

The full cross-product (1,988 verbs × ~5.5 million concrete templates, in 2–3 verb forms each) is far too large to pair exhaustively, so items are sampled in four steps. Stages 3–6 below give the details.

1. **Concrete templates** (stage 3): 250 are drawn at random from each of the 27 frame class × aspect cells and scored by how strongly the sentence predicts *some* verb in the verb slot.
2. **Bins** (stage 4): within each cell the scored templates are split into 5 equal-count score bins, and 4 are drawn from every bin, giving 504.
3. **Pairs** (stage 5): each selected template yields one `different_verb` pair and one `same_verb` pair, with verbs dealt from a shuffled deck. Every sentence is then scored with its actual verb.
4. **Lists** (stage 6): each pair type is shuffled separately, 400 of its 504 pairs are kept, and they are split into 16 lists of 25.

**What this controls:**
- Every frame class × aspect cell gets the same share of templates, and within a cell, carrier sentences from high- and low-predictability bins are equally represented. Scores are binned within cells because they are only comparable there: complement frames have a predictable main verb, bare intransitives don't.
- Verb use is spread evenly: no verb appears twice.
- Every frame appears in some list.

**What it doesn't control:**
- **Pair difficulty.** Each pair's score difference is recorded and split into 10 quantiles, but nothing selects on it. The template bins measure the carrier sentence, not the gap between a pair's two sentences, and the list-level quantile constraint isn't enforced by bead's partitioner. Its quantile spread is only reported.
- **Repeated comparisons.** Each verb × concrete-template combination appears in exactly one pair, against one competitor. So a ranking of bad combinations has to come from a model that generalizes over verbs, frames and nouns, not from comparing individual sentences.

The [past-tense sampling test](#past-tense-sampling-test-make-data-past) is a simpler alternative that selects pairs on their score gap and aims to cover every verb.

## Quick Start

### Prerequisites

```bash
# From the repository root
uv sync --all-extras
cd gallery/kor/argument_structure
```

### Generate Data

```bash
make help          # Show all available targets
make data          # Quick test run: 40 concrete templates per cell, 2 lists
make deployment    # jsPsych/JATOS deployment for 2 lists (local test)
```

Open `deployment/local/list_01/index.html` in a browser to try the experiment.

### Run Production Pipeline

```bash
make data-full         # 250 concrete templates per cell, 16 lists
make deployment-full   # Deployment for all 16 lists
# or both at once:
make all
```

### Run the Past-Tense Sampling Test

```bash
make data-past         # Simple-aspect frames, past tense, 27 lists
```

See [Past-Tense Sampling Test](#past-tense-sampling-test-make-data-past).

## Pipeline Stages

| Step | Script | Make target |
|------|--------|-------------|
| 1. Lexicons | `generate_lexicons.py` | `make lexicons` |
| 2. Frames | `generate_templates.py` | `make templates-full` |
| 3. Score concrete templates | `score_concrete_templates.py` | `make score-concrete-templates` |
| 4. Bin and select | `bin_concrete_templates.py` | `make bin-concrete-templates` |
| 5. 2AFC pairs | `create_concrete_template_pairs.py` | `make concrete-template-pairs` |
| 6. Lists | `generate_lists.py` | `make lists` / `make lists-full` |
| 7. Deployment | `generate_deployment.py` | `make deployment` / `make deployment-full` |

All parameters are in `config.yaml`, one section per step. Command-line flags on each script override them.

---

### Stage 1 — Lexicons (`make lexicons`)

Generates 9 JSONL files under `lexicons/`:

| File | Entries | Contents |
|------|---------|----------|
| `verbs.jsonl` | 10,012 (2,674 lemmas) | UniMorph Korean predicates (1,988 verbs + 686 adjectives) in present, past, adnominal and `-고` forms |
| `bleached_nouns.jsonl` | 19 | Placeholder nouns in 7 semantic classes (animate, inanimate object, location, temporal, abstract, event, quantity) |
| `bleached_verbs.jsonl` | 7 | Semantically light verbs (하다, 가지다, 가다, 얻다, 만들다, 일어나다, 오다) |
| `bleached_adjectives.jsonl` | 10 | Light adjectives |
| `case_markers.jsonl` | 17 | NOM/ACC/DAT/INST/LOC.SRC/LOC.GOAL + COM/GEN/TERM/INIT/SIM/PRIV particles |
| `auxiliary_verbs.jsonl` | 2 | Progressive auxiliary 있다 (PRS/PST) |
| `spatial_nouns.jsonl` | 19 | Relational nouns (위, 아래, 앞, 뒤, 안, 밖, …) |
| `complex_postpositions.jsonl` | 11 | Multi-morpheme postpositions (DAT-governed: 6, ACC-governed: 5) |
| `comp_verbs.jsonl` | 14 | Complement-clause verb forms (7 adnominal + 7 present declarative) |

Particle allomorphy is handled by `fc_agree` constraints in the frames.

---

### Stage 2 — Frames (`make templates-full`)

Generates `templates/generic_frames.jsonl` with **138 frames** = 46 base + 92 progressive. Every base frame gets a present and a past progressive: 46 × 2 = 92. Word-order variants are treated as equivalent, so each argument set appears in one order only: the dative comes before the accusative (`NP-NOM NP-DAT NP-ACC ___`, the unmarked ditransitive order), and other obliques follow the object (`NP-NOM NP-ACC NP-LOC ___`, `NP-NOM NP-DAT NP-ACC NP-INST ___`).

| Family | Base frames | Example |
|--------|-------------|---------|
| Core (intransitive, transitive, ditransitive) | 3 | `{noun_subj}{nom} {noun_pobj}{dat} {noun_dobj}{acc} {verb}.` |
| Obliques (one or two of dat, loc, goal, inst, com, term, init; includes 4 source–goal frames) | 27 | `{noun_subj}{nom} {noun_pobj}{loc} {noun_pobj2}{inst} {verb}.` |
| Spatial nouns | 6 | `{noun_subj}{nom} {noun_pobj} {spatial_noun}{goal} {verb}.` |
| Complex postpositions | 4 | `{noun_subj}{nom} {noun_pobj}{complex_postp_dat} {verb}.` |
| Source–path–goal | 1 | `{noun_subj}{nom} {noun_pobj}{loc} {noun_pobj2}{pobj_acc} {path_postp} {noun_pobj3}{inst} {verb}.` |
| Complement clauses | 5 | `{noun_subj}{nom} {comp_subj}{comp_nom} {comp_verb_decl}고 {verb}.` |
| **Total base frames** | **46** | |

**Progressive frames (92)**: `-고 있다` / `-고 있었다` variants:
```
{noun_subj}{nom} {noun_dobj}{acc} {verb} {aux}.
```

Every frame carries metadata used downstream:
- **`frame_class`:** one of 9 classes (base frames per class below)
- **`obliques`:** the oblique types in surface order
- **`aspect`:** `simple`, `prog_prs` or `prog_pst`
- **`adjunct_tier`:** `core`, `single_adjunct`, `multi_adjunct` or `clausal`
- **`n_args`**, and **`comp_type`** for complement frames

| Frame class | Base frames | Frame class | Base frames |
|-------------|-------------|-------------|-------------|
| `NP_V` | 1 | `NP_V_NP_OBL_OBL` | 7 |
| `NP_V_NP` | 1 | `NP_V_OBL_OBL_OBL` | 1 |
| `NP_V_OBL` | 12 | `NP_V_S` | 4 |
| `NP_V_NP_OBL` | 12 | `NP_V_OBL_S` | 1 |
| `NP_V_OBL_OBL` | 7 | **Total** | **46** |

To generate a subset:
```bash
uv run python generate_templates.py --include adjuncts --adjuncts dat loc
```

---

### Stage 3 — Score concrete templates (`make score-concrete-templates`)

**Sampling** (`concrete_template_scoring` in `config.yaml`, seed 0):
- **Cells.** Frames are grouped into 27 cells: 9 frame classes × 3 aspects. Each cell gets the same target, 250 concrete templates, however many frames it contains.
- **Within a cell,** the target is split evenly across the cell's frames, with any remainder going to randomly chosen frames.
- **Within a frame,** every non-verb slot is filled independently and uniformly at random from its candidate list. A filling is rejected if it breaks the frame's constraints (e.g. particle allomorphy) or duplicates one already drawn. A frame with at most 4× its share of possible fillings is enumerated in full and sampled without replacement.
- **Exhausted frames.** A frame that runs out of fillings hands its unused share to the cell's other frames over further rounds. The 3 `NP_V` cells have only 19 fillings each (one per subject noun), so they get all 19.
- **Total:** 24 × 250 + 3 × 19 = **6,057** concrete templates.

**Scoring:**
- The verb slot is replaced by `[MASK]` and `klue/bert-base` fills it one word piece at a time, with a beam of 5 and up to 3 pieces. Each step substitutes the chosen piece and masks the next position.
- A completion must end like a real verb form: 다 in finite frames, 고 in progressives. Degenerate completions are rejected, such as a leading `##` piece, a standalone copula, or a finite verb followed by a copula.
- `raw_score` is the log-probability of the best valid completion. It says how strongly the carrier sentence predicts *some* verb; it is not a score for any particular lexicon verb.
- Templates where no beam candidate is valid are flagged (`beam_had_valid_candidate: false`) and left out of binning.

**Variants:**
- `make score-concrete-templates` uses the config sample (250 per cell).
- `make score-concrete-templates-quick` takes 40 per cell: 24 × 40 + 3 × 19 = 1,017. `make data` uses it.
- `uv run python score_concrete_templates.py --exhaustive` scores all ~5.5 million concrete templates. This isn't needed.

Output: `concrete_template_scores/cell_sample_scores.jsonl`

---

### Stage 4 — Bin and select (`make bin-concrete-templates`)

`concrete_template_binning` in `config.yaml`, seed 42.

1. **Bin.** Within each cell, concrete templates are ranked by `raw_score`, with ties broken by a stable template id, and cut into 5 equal-count bins. Bin 1 is the lowest-scoring. A cell gets fewer bins when it can't fill 5 bins of at least 8 (`min_per_bin`), so the 19-template `NP_V` cells get 2.
2. **Select.** 4 concrete templates are drawn from every bin, one at a time. Each draw goes to the frame with the fewest picks so far in the whole cell, with ties broken at random, and takes a random template of that frame from the bin. Low-scoring (anomalous) carriers are therefore kept as deliberately as high-scoring ones, and picks spread across the cell's frames instead of clustering in its largest frame.
3. **Total:** 24 × 5 × 4 + 3 × 2 × 4 = **504** concrete templates.

`bin_summary.json` records each bin's score range, size and number of frames selected.

Output: `concrete_template_scores/selected_concrete_templates.jsonl`, `concrete_template_scores/bin_summary.json`

---

### Stage 5 — 2AFC pairs (`make concrete-template-pairs`)

`concrete_template_pairs` in `config.yaml`, seed 42. Builds one `different_verb` and one `same_verb` pair per selected concrete template.

**Verb deck:**
- The deck holds every lemma with present, past and `-고` forms: the 1,988 verbs. Predicate adjectives are left out by default (`drop_adjectives: true`), because in progressive and object frames they're ungrammatical for reasons unrelated to argument structure.
- The deck is shuffled and dealt without replacement, and reshuffled only when it runs out.
- One run draws 504 × 3 = 1,512 verbs, so no verb appears twice and 476 verbs don't appear at all. Which ones are left out depends on the seed.

**Verb form:**
- Each pair uses one form on both sides.
- Simple-aspect pairs use present or past tense, chosen at random per pair (먹는다 / 먹었다).
- Progressive pairs use the `-고` participle (먹고 있다 / 먹고 있었다).

**Pair types:**
- **`different_verb`:** the selected template with two verbs drawn from the deck.
- **`same_verb`:** one verb from the deck, in the selected template and in a partner template.
  - The partner comes from the same 504, taken from a shuffled rotation of templates with the same aspect.
  - The partner must have a different frame class, and one with the same subject noun is preferred, so the two sentences differ mainly in argument structure.
  - The rotation is used up before it reshuffles, so each template is a partner about once.

**Sentences and scores:**
- Sentences are rendered as participants see them, with a repeated noun shown as 다른 X.
- Each of the 2,016 sentences is scored as the log-probability of its actual verb in that sentence (`lm_raw_*`). Each verb token is masked in turn, with the tokens before it filled in, the same way stage 3 scores.
- That raw score is mostly the verb's own rarity and length, so a per-run least-squares fit is removed from it: `raw ~ baseline + n_tokens`. The `baseline` is the same verb with only the sentence ending (`.`, ` 있다.`, ` 있었다.`) and no arguments.
- The residual is `lm_score_*`. `lm_score_diff` is the absolute difference between the two sides, cut into 10 quantiles (`quantile`) within each pair type.
- Option order is randomized.

Output: `items/2afc_pairs.jsonl` (504 × 2 = **1,008 pairs**, 1,512 distinct verbs), `items/2afc_pairs_scoring.json`

---

### Stages 6–7 — Lists and Deployment

```bash
make lists             # Partition pairs into 2 lists (quick test)
make lists-full        # Partition pairs into 16 lists (config default)
make deployment        # Build jsPsych experiment for 2 lists (local test)
make deployment-full   # Build jsPsych experiment for all 16 lists
```

**Composition:** 16 lists × 50 = 800 of the 1,008 pairs. Each pair type is handled separately (`split_by`, seed 42):
1. Shuffle the type's 504 pairs.
2. Move the first pair of every frame (`frame_id`) to the front, so every frame is guaranteed a place.
3. Keep the first 16 × 25 = 400 pairs. The other 104 are unused.
4. Partition them into 16 lists of 25 with bead's `balanced` strategy, under the list constraint that no verb repeats within a list.
5. Merge list *i* of each pair type, so every list has exactly 25 `same_verb` + 25 `different_verb` pairs.

**Constraints that are recorded but not enforced:**
- The `grouped_quantile` (score-difference) and frame `diversity` list constraints are recorded on the lists, but bead's partitioner doesn't act on them.
- The batch constraints are checked and reported, not enforced by swapping items: every frame appears, pair types split 50/50, each quantile appears at least 50 times, and no verb is in more than 4 lists.
- The quantile spread of the lists is therefore whatever the random draw gives. The 2-list test run can't meet the frame-coverage and quantile constraints and shows ⚠ for them.
- **Language:** all participant-facing text is Korean (title 문장 비교 실험, buttons 이전/다음/실험 시작, labels 문장 1/문장 2/선택, progress 진행률), set in `deployment.experiment`. The question above each pair, 어느 문장이 더 자연스러운가요?, comes from the `protocol:` section (see below).
- **Prolific:** set `deployment.jspsych.prolific_completion_code` to redirect participants on completion.

Outputs: `lists/experiment_lists.jsonl`, `deployment/local/list_XX/`, `deployment/jatos/list_XX.jzip`

---

### Annotation Protocol (`make validate-protocol`)

The 2AFC question is declared once in the `protocol:` section of `config.yaml` and built by `protocol.py` into a bead `AnnotationProtocol`, as in the English gallery. The deployment builds its item template (and prompt) from it, and every pair records the anchor name (`anchor: acceptability`). `make validate-protocol` checks that the protocol builds and that the prompt passes its drift checks (ends with `?`, contains 자연스러, at least 10 characters).

### Active-Learning Simulation (`make simulate`)

`simulate_pipeline.py` runs the active-learning loop with simulated annotators instead of participants, to test it before real data collection. It's a port of the English gallery's script, with fixes noted in its docstring.

1. The pairs are split into a held-out test set (150), an initial labeled set (100) and a pool (the rest).
2. Simulated annotators pick between a pair's two sentences with probability softmax(score / temperature), from the pairs' rarity-corrected verb scores. Their agreement with each other (percentage agreement) is the "human" ceiling.
3. A forced-choice model (`klue/roberta-base`, bead's `ForcedChoiceModel`) is trained on the labeled judgments, and its accuracy is measured on the test set.
4. The 50 pool pairs the model is least sure about are annotated next, and training repeats, for up to 10 rounds or until accuracy is within 0.05 of the annotators' agreement.

```bash
make simulate            # uncertainty sampling (config: simulation)
make simulate-baseline   # random selection, for comparison
make simulate-quick      # 2 rounds, 1 epoch: checks the loop runs
```

Results go to `simulation_output/simulation_<selector>.json`, including each round's accuracy and what the selected pairs look like (pair types, frame classes).

## Past-Tense Sampling Test (`make data-past`)

A simpler replacement for stages 3–5, limited for now to the 46 simple-aspect frames and past-tense verbs (먹었다). It drops beam decoding and template binning. Concrete templates are sampled evenly, pairs are scored with their real verbs, and selection is on the pair's score gap. Parameters are in the `past_test` section of `config.yaml`; the old stages are untouched.

**Coverage target: every verb at least once.**
- **Pair arithmetic:**
  - A `different_verb` pair uses 1 template and 2 verbs; a `same_verb` pair uses 2 templates and 1 verb.
  - So with 25 + 25 pairs per list, each pair averages 1.5 of each.
  - Covering all 1,988 verbs takes about 1,326 pairs. That's 27 lists, giving 1,350 pairs and 2,025 verb slots.
- **Frames and fillers:** the same pairs cover all 46 frames, about 44 templates each, and every noun, spatial noun, complex postposition and complement verb in every slot it can fill.

| Target | Pairs | Lists of 50 |
|---|---|---|
| Every filler in every frame | 583 | 12 |
| **Every verb at least once** (used here) | **1,326** | **27** |
| Every verb in every frame class | 8,946 | 179 |
| Every verb in every frame | 45,724 | 915 |

### Step 1 — Sample and score concrete templates (`make sample-concrete-templates`)

`sample_concrete_templates.py`, seed 42.

**Budget.** 3 candidate pairs per selected pair × 27 lists × (25 + 2 × 25) templates per list = **6,075** concrete templates. They are split evenly across the 46 frames, about 135 each. NP_V has only 19 possible templates, so it gets all of them, and its leftover is shared among the other frames.

**Filler balance.** For each frame:
1. Draw 8× its share uniformly at random from its valid fillings.
2. Pick its share one at a time, each time taking the candidate whose fillers have been used least so far in that frame.

This way every filler appears in every slot it can fill before any repeats. Particle slots (이/가, 을/를, 로/으로) follow their noun, so they aren't balanced.

**Carrier score.**
1. The verb is replaced by `[MASK][MASK]다` in the sentence as participants see it, with a repeated noun shown as 다른 X. The 다 is inserted as the word-internal piece `##다`; typed as text, the tokenizer would make it a separate word.
2. The first mask is the stem. The 5 most probable Hangul word-initial pieces are tried.
3. Each is filled in, and the second mask takes the best past-tense piece: a word-internal syllable ending in ㅆ (었/았/했/됐/였...).
4. `carrier_score` = log p(stem) + log p(tense | stem) for the best of the 5. `carrier_best` is the verb they spell, e.g. 바꾸었다.

klue/bert-base splits 70% of the lexicon's past forms exactly as stem + tense + 다 (먹/##었/##다, 사랑/##했/##다). Filling the masks one after the other keeps the two pieces from belonging to different verbs, which predicting both at once produced (있 + 했 → 있했다).

**What the score is used for.** It says how strongly the carrier predicts *some* past-tense verb, not any lexicon verb. A low score can mean an open carrier (사람이 ___) as much as an odd one. So it is only used to match `same_verb` partners and kept as metadata, never used to bin or filter. `carrier_percentile` is its rank within the frame class, since raw scores differ between classes.

Outputs: `concrete_template_scores/past_templates.jsonl`, `concrete_template_scores/past_templates_summary.json` (per-frame shares and filler coverage).

### Step 2 — Build, score, bin and select pairs (`make select-pairs`)

`select_pairs.py`, seed 42.

1. **Split templates.** Each frame's templates are shuffled. A third go to `different_verb` and the rest to `same_verb`, which uses two per pair. No template is used twice.
2. **Build 3× the needed pairs per type** (2,025 candidates each).
   - **`different_verb`:** one template and two verbs from the shuffled deck, which deals every verb once before any repeats.
   - **`same_verb`:** one deck verb in two templates of different frame classes. The partner is the remaining template with the closest `carrier_percentile`, and a different subject noun costs 0.05. This way the two sentences differ mainly in argument structure, and neither carrier is much odder than the other.
3. **Score** each sentence as the log-probability of its actual verb, minus the part explained by the verb's rarity and length (as in stage 5). `lm_score_diff` is the absolute difference between the two sides.
4. **Bin** `lm_score_diff` into 5 equal-count bins per pair type. `quantile` 1 is the smallest gap, the hardest pair.
5. **Select** 27 × 25 = 675 pairs per type, 135 from every bin.
   - Picks go round robin over (pair type, bin). Each pick takes the candidate whose verbs have been used least so far, then whose frames have, then whose fillers have.
   - A repair pass then swaps in candidates carrying verbs the first pass missed, each replacing a pair from the same bin whose verbs all appear elsewhere.

Outputs: `items/2afc_pairs.jsonl` (the same file and format as stage 5, so stages 6–7 work unchanged), `items/2afc_pairs_selection.json` (bin edges, coverage, rarity fit).

### Step 3 — Lists (`make lists-past`)

Stage 6 with 27 lists of 25 + 25, every selected pair used. `PAST_N_LISTS` in the `Makefile` must match `past_test.n_lists`. For deployment, run `uv run python generate_deployment.py --n-lists 27`, or `make deployment` for 2 lists.

### Results (seed 42)

| | Result |
|---|---|
| Concrete templates sampled | 6,075; all 2,584 filler positions covered |
| Pairs | 1,350 = 675 `different_verb` + 675 `same_verb`, exactly 135 per bin per type |
| Verbs | 1,974 / 1,988 (99.3%); 51 appear twice, 14 are missing |
| Frames | 46 / 46, with 14–48 template slots each |
| Filler positions in the selected pairs | 2,467 / 2,584 (95.5%) |
| `same_verb` partners | Mean carrier-percentile gap 0.0075; 637 / 675 share the subject noun |
| Lists | 27 × 50; all batch constraints met (frame coverage, 50/50 pair types, quantile counts, verb spread) |

**Known gaps:**
- **Missing verbs.** Only 37 verb slots are spare, so the last 14 verbs can't be placed without unbalancing the bins.
- **Filler coverage** is partly lost at selection, where fillers are only the third tie-breaker.
- **Duplicated code.** The old stages 3–5 still exist alongside this path; `select_pairs.py` reuses helpers from `create_concrete_template_pairs.py` and `score_concrete_templates.py`.

## Makefile Targets

### Main Targets

```bash
make help          # Show all targets
make validate-protocol   # Check the protocol section (run first by data targets)
make data          # Quick test run (40 per cell, 2 lists)
make data-full     # Production run (250 per cell, 16 lists)
make all           # data-full + deployment-full
make data-past     # Past-tense sampling test (27 lists)
```

### Pipeline Steps

```bash
make lexicons                        # [1/7]
make templates-full                  # [2/7] all 138 frames (make templates: base frames only)
make score-concrete-templates        # [3/7] (…-quick: 40 per cell)
make bin-concrete-templates          # [4/7]
make concrete-template-pairs         # [5/7]
make lists / lists-full              # [6/7]
make deployment / deployment-full    # [7/7] (deployment-local-only: no JATOS export)
```

### Past-Tense Sampling Test

```bash
make sample-concrete-templates       # Per-frame sample + carrier score
make select-pairs                    # Candidate pairs, score, bin by score gap, select
make lists-past                      # 27 lists of 25 + 25
```

### Legacy Exhaustive-Fill Path

```bash
make fill-templates / fill-templates-full   # Fill frames exhaustively
make 2afc-pairs / 2afc-pairs-full           # Whole-sentence scored pairs (overwrites items/2afc_pairs.jsonl)
```

This path is superseded by concrete-template sampling and kept for reference. The `template`, `resources` and `items` sections of `config.yaml` belong to it.

### Simulation

```bash
make simulate / simulate-baseline / simulate-quick
```

### Inspection and Cleaning

```bash
make show-stats                  # Count entries at each pipeline stage
make show-config                 # Print key config values
make lint                        # Ruff on all pipeline scripts
make clean                       # Remove generated items and model cache
make clean-concrete-templates    # Remove concrete-template scores and selections
make clean-all                   # Remove all generated files
```

## Data Format

### Concrete Template Scores

One row per concrete template (`concrete_template_scores/*.jsonl`):

| Field | Meaning |
|-------|---------|
| `frame_name`, `frame_id` | The frame this concrete template fills |
| `frame_class`, `aspect`, `obliques`, `adjunct_tier`, `n_args` | Frame metadata |
| `prefix`, `suffix` | Text before and after the verb slot |
| `slot_fillers` | Lexicon item (`id`, `lemma`, `form`) for each slot |
| `raw_surface`, `raw_score` | Model's best verb form and its log-probability |
| `beam_mass`, `raw_n_tokens`, `beam_had_valid_candidate` | Diagnostics |
| `concrete_template_id`, `cell`, `bin`, `n_bins` | Added by binning (selected file only) |

### 2AFC Pairs

bead `Item` objects (`items/2afc_pairs.jsonl`) with the two sentences in `options` and in `rendered_elements` (`option_a`, `option_b`, the form bead's forced-choice model reads), and these `item_metadata` fields:

| Field | Meaning |
|-------|---------|
| `anchor` | Protocol anchor the item answers (`acceptability`) |
| `pair_type` | `same_verb` or `different_verb` |
| `verb` / `verb1`, `verb2` | Verb lemma(s), used by the list constraints |
| `verb_form`, `aspect` | `PRS`/`PST`/`PTCP`, and the frames' aspect |
| `frame_id`, `frame_name` / `frame1`, `frame2` | Frame reference(s) |
| `lm_score_a`, `lm_score_b`, `lm_score_diff` | Rarity-corrected verb scores and their difference |
| `lm_raw_*`, `lm_baseline_*`, `lm_n_tokens_*` | Raw scores and the rarity correction's inputs |
| `quantile` | Quantile of `lm_score_diff` within the pair type |
| `concrete_template_id_a/b`, `slot_filler_ids_a/b`, `verb_id_a/b` | References to concrete templates and lexicon items |

## Project Structure

```
argument_structure/
├── config.yaml                          # All pipeline parameters
├── Makefile                             # Pipeline targets
├── protocol.py                          # Builds the 2AFC protocol from config.yaml
├── generate_lexicons.py                 # Stage 1
├── generate_templates.py                # Stage 2 (frames)
├── score_concrete_templates.py          # Stage 3
├── bin_concrete_templates.py            # Stage 4
├── create_concrete_template_pairs.py    # Stage 5
├── sample_concrete_templates.py         # Past-tense test, step 1
├── select_pairs.py                      # Past-tense test, step 2
├── generate_lists.py                    # Stage 6
├── generate_deployment.py               # Stage 7
├── simulate_pipeline.py                 # Active-learning simulation
├── fill_templates.py                    # Legacy exhaustive fill
├── create_2afc_pairs.py                 # Legacy pair construction
├── resources/                           # CSV sources for the hand-built lexicons
├── lexicons/                            # Generated lexicons (JSONL)
├── templates/                           # Generated frames
├── concrete_template_scores/            # Scores, selections, bin summary
├── items/                               # 2AFC pairs and scoring summary
├── lists/                               # Experiment lists
└── deployment/                          # local/ and jatos/ builds
```

## Known Limitations

- **Complement types:** 5 of the English gallery's 8. `-(으)ㄹ 거라고`, `-(으)ㄹ지` and `-기를` need additional UniMorph forms.
- **Voice alternation:** causative/inchoative pairs (깨다 / 깨지다) need derived verb forms and aren't generated yet.
- **Verb coverage:** 1,988 verbs against 3,475 in the English gallery. UniMorph Korean is fully used, so closing the gap needs another lexical resource.
- **Active learning:** the loop can be simulated (`make simulate`), but the real-data loop from the English gallery (training on participants' judgments) isn't ported yet. The `training`, `active_learning` and `evaluation` sections of `config.yaml` are placeholders.
