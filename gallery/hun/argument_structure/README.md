# Hungarian argument structure: Stage 1

This directory adapts the English/Korean argument-structure pipeline to Hungarian. The central question is the same across languages: **which verbs are acceptable in which candidate morphosyntactic frames?**

The Hungarian implementation follows the Korean generic-frame architecture more closely than the English VerbNet-based implementation. It tests UniMorph verb lemmas across a controlled frame inventory, while realizing case, object definiteness, verbal agreement, and temporal context in a Hungarian-specific way.

## What Stage 1 tests

The basic design is:

`verb × frame → generated sentence → acceptability comparison`

The generator does **not** assume that every verb-frame combination is grammatical. A genuine verb-frame mismatch is useful experimental data. Automatic validation instead asks whether the requested Hungarian frame was realized correctly: case, agreement, determiner choice, tense, fillers, punctuation, and word boundaries.

## Relationship to English and Korean

| Component | English | Korean | Hungarian |
| --- | --- | --- | --- |
| Main verb source | VerbNet + morphology | UniMorph | UniMorph |
| Frame source | VerbNet frames | 47 generic frames | 55 generic surface templates |
| Case/frame realization | word order + PPs | NOM/ACC/DAT/LOC/INST particles | case-inflected nouns; NOM/ACC/DAT/SUP/SUB/DEL/ABL/TER/INS |
| Verb-frame cross-product | VerbNet-driven | exhaustive generic-frame design | exhaustive generic-frame design |
| Ongoing/progressive condition | English progressive | Korean progressive auxiliary | `éppen` ongoing-event context |
| Clausal complement | VerbNet clausal frames | generic clausal frame | finite direct-object `hogy` frame |
| Object definiteness split | n/a | n/a | indefinite vs. definite object/conjugation |
| Relational/spatial frames | PPs | spatial nouns + 에/에서/로 | bare postpositions in ESS/LAT/ABL series |
| Complex adpositions | multiword PPs | 에 대해서, 을 통해서 | case-governing postpositions (`képest` ALL, `együtt` INS) |

Korean has 47 generic templates; Hungarian has 55. The extra frames come from
the Hungarian definiteness split: every ACC frame is realized twice, once
indefinite and once definite, and the transitive temporal and postpositional
templates split the same way.

Hungarian covers each of Korean's seven adjunct types. Korean marks them with
particles; Hungarian marks them with case, except for the comitative, which is
syncretic with the instrumental:

| Korean adjunct | Korean particle | Hungarian case | Stage-1 realization |
| --- | --- | --- | --- |
| dat | 에게 | DAT | `egy embernek` |
| loc | 에서 | SUP | `egy helyen` |
| inst | (으)로 | INS | `egy eszközzel` |
| goal | 에 | SUB | `egy helyre` |
| source | 에서 | DEL | `egy helyről` |
| com | 와/과 | INS | `egy emberrel` |
| term | 까지 | TER | `egy helyig` |
| init | 부터 | ABL | `egy helytől` |

The locative/goal/source triple uses Hungarian's *surface* series (SUP `-on`,
SUB `-ra`, DEL `-ról`) rather than mixing series, so those three frames differ
only in direction. The comitative and instrumental frames share a case and
differ only in the semantic class of the noun; that is a fact about Hungarian,
not a gap in the inventory.

The labels *intransitive* and *transitive* are retained where they align with the Korean frame names. In Stage 1, they describe the **surface frame** (no overt ACC object vs. overt ACC object), not a claim about a verb's lexical valency.

## The 55 Hungarian frames

`FRAME_REFERENCE.csv` contains every frame, a grammatical Hungarian example, and an English translation. Each example uses a matrix verb chosen to suit its frame, so the sentence is well formed; that is a presentational choice for the reference table only. The generator itself crosses every verb with every frame and makes no such choice.

`FRAME_REFERENCE.csv` is generated from the template inventory by
`build_frame_reference.py`, so it cannot drift from the frames the pipeline
actually produces. The inventory consists of:

- 12 frames without an overt accusative object;
- 12 accusative-object structures × indefinite/definite realization = 24 templates;
- 1 finite direct-object `hogy` complement frame;
- 6 `éppen` present/past templates, including the definiteness split in transitive frames;
- 9 bare-postposition frames (ESS/LAT/ABL × intransitive/indefinite/definite);
- 3 case-governing postposition frames.

### Hungarian-specific choices

**Object definiteness.** Indefinite objects use the indefinite conjugation; definite objects use `a/az` and the definite conjugation. These changes are treated together as one Hungarian morphosyntactic realization.

**Locative baseline.** Stage 1 uses `egy helyen` (“at/in a place”), with superessive morphology, as a neutral controlled locative.

**Instrument baseline.** Stage 1 uses `egy eszközzel` (“with a tool”) rather than reusing the generic object noun as an instrument.

**Finite `hogy` frame.** The clausal template represents a candidate direct-object clause. The matrix verb therefore uses the objective/definite conjugation. Oblique clausal-complement patterns are left for Stage 2.

**`éppen`.** `éppen` creates an ongoing-event context; it is not analyzed as a Hungarian progressive morpheme and is not claimed to be morphologically equivalent to English `be V-ing` or Korean `-고 있다`.

**Preverbs.** Transparently prefixed verbs remain in nominal and clausal frames but are excluded from Stage-1 `éppen`/tense 2AFC contrasts because Hungarian preverbs interact with aspect and word order.

## 2AFC design

The two broad pair families follow the English/Korean pipeline:

1. **same verb, controlled frame contrast**;
2. **same frame and fillers, different matrix verb**.

Hungarian adds explicit `contrast_type` metadata so each same-verb comparison has a clear interpretation. Current labels are:

- `add_dative`
- `add_locative`
- `add_instrumental`
- `add_goal`
- `add_source`
- `add_comitative`
- `add_terminative`
- `add_initiative`
- `add_spatial_postposition`
- `add_complex_postposition`
- `object_definiteness`
- `clausal_complement`
- `ongoing_context`
- `tense`
- `lexical_verb` (different-verb pairs)

The lative and ablative spatial frames (`mögé`, `mögül`) currently have no
same-verb contrast. Their natural comparison is against the essive series,
which would make `postposition` the manipulated slot, and `shared_slots()` in
`create_2afc_pairs.py` treats `postposition` as shared material that has to
match — so those pairs would all be rejected. Adding that contrast means
excluding `postposition` there first. `tests/test_config.py` pins the current
set, so this cannot drift unnoticed.

For different-verb pairs, the template, non-verb fillers, and relevant inflectional features are held constant. For same-verb pairs, the shared lexical material is held constant and only the licensed contrast-specific change is allowed.

Language-model scores are used **only for stratification**. They are not grammaticality labels and do not filter out verb-frame mismatches.

## Running the pipeline

From `hun/argument_structure`:

```bash
make data          # Quick test run (100 verbs)
make data-full     # Production run (all verbs, all pairs)
make deployment    # Build a jsPsych deployment for 2 lists
make help          # Show all targets
```

| Step | Script | Make target |
| --- | --- | --- |
| 0. Generated resources | `resources/build_bleached_nouns.py` | `make resources` |
| 1. Lexicons | `generate_lexicons.py` | `make lexicons` |
| 2. Templates | `generate_templates.py` | `make templates` |
| 3. Fill templates | `fill_templates.py` | `make fill-templates` / `make fill-templates-full` |
| 4. 2AFC pairs | `create_2afc_pairs.py` | `make 2afc-pairs` / `make 2afc-pairs-full` |
| 5. Lists | `generate_lists.py` | `make lists` |
| 6. Deployment | `generate_deployment.py` | `make deployment` / `make deployment-full` |

Or call the scripts directly:

```bash
python resources/build_bleached_nouns.py
python generate_lexicons.py
python generate_templates.py
python fill_templates.py
python create_2afc_pairs.py
python generate_lists.py
python generate_deployment.py
```

For development or small-scale inspection, `fill_templates.py` accepts `--verb-start`, `--verb-count`, and `--max-per-template`. Omitting `--verb-count` processes the full available verb inventory. `create_2afc_pairs.py --skip-scoring` is available for structural inspection; final experiment preparation should run pair generation with language-model scoring enabled.

`generate_templates.py` takes `--include` to select frame families
(`nominal`, `clausal`, `ongoing`, `spatial`, `complex`), `--adjuncts` to
restrict the nominal frames to particular adjunct types, and `--word-order`
(see below).

## Generated resources

Two files under version control are generated rather than hand-maintained:

- `resources/bleached_nouns.csv`, built by `resources/build_bleached_nouns.py`.
  Hungarian marks 17 productive cases on every noun, so a hand-typed paradigm
  does not scale and a typo in it becomes a silent confound. The generator
  computes vowel harmony and `-v-` assimilation (`tárgy` → `tárggyal`, `hely` →
  `hellyel`); accusatives and any stem alternations are listed explicitly,
  because their linking vowels are not predictable.
- `FRAME_REFERENCE.csv`, built by `build_frame_reference.py`.

Both accept `--check`, and the test suite calls both, so a hand edit that
diverges from the generator fails rather than passing quietly.

## Word order

Hungarian's immediately preverbal position is the focus position, so a frame
that stacks several arguments there is a marked order. `generate_templates.py`
exposes this as `--word-order` rather than leaving it implicit:

```bash
make templates                     # preverbal (default)
make templates WORD_ORDER=neutral  # verb-medial
```

| Order | Example |
| --- | --- |
| `preverbal` | `Egy ember egy tárgyat egy helyen tesz.` |
| `neutral` | `Egy ember tesz egy tárgyat egy helyen.` |

`preverbal` reproduces the original Stage-1 output exactly, and remains the
default so nothing changes silently. `neutral` places the finite verb directly
after the subject, which is the unmarked order for a sentence with no focused
constituent. Both orders are generated from the same frame inventory, so which
one to collect on can be decided empirically. `éppen` stays adjacent to the
subject in both, since it is what creates the ongoing reading.

## Tests

```bash
make test       # run the suite
make samples    # print one rendered example per frame
make check      # generated-resource check + lint + tests
```

`tests/render_samples.py` resolves the real constraint expressions against the
real resource CSVs using a small stub verb lexicon, so frames can be inspected
in Hungarian without downloading UniMorph or loading a language model.

Constraints are evaluated by bead's own DSL (`bead/dsl/grammar.lark`), not by
Python. The two look similar, which makes it easy to write a constraint that
Python accepts and the DSL cannot parse — `is True` rather than `== true`, or
`None`, neither of which the grammar has. `render_samples.py` lints for those
constructs and `tests/test_templates.py` checks every constraint the generator
emits, so the mistake surfaces here rather than as a per-template failure at
fill time.

`tests/test_config.py` ties `config.yaml` to the rest of the pipeline: every
generated lexicon must be declared (an undeclared one is loaded by nothing, so
its frames silently fill zero sentences), every template slot must have a
strategy, and every template named in `create_2afc_pairs.py` must still
exist.

Following section 6 of `LINGUISTIC_DESIGN_NOTES.md`, the tests check that each
frame is *realized* as specified — case, agreement, article allomorphy,
postposition government, controlled fillers, word order, and that every frame
renders at all. They say nothing about whether a verb is acceptable in a frame.

## Human linguistic validation

The `validation/` utilities support manual review of generated stimuli. They distinguish intended low-acceptability outcomes such as `verb_frame_mismatch` from unintended `generator_error` cases. This keeps linguistic acceptability judgments separate from mechanical generation checks and language-model stratification.

## Stage 2

Stage 2 is intentionally separate from this cross-linguistic baseline. Candidate extensions include pro-drop/argument omission, richer local-case alternations, preverbs and event structure, focus/information structure, potential mood, and more detailed complementation patterns.
