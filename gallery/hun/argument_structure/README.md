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
| Frame source | VerbNet frames | 19 generic frames | 28 generic surface templates |
| Case/frame realization | word order + PPs | NOM/ACC/DAT/LOC/INST particles | case-inflected nouns; NOM/ACC/DAT/SUP/INS |
| Verb-frame cross-product | VerbNet-driven | exhaustive generic-frame design | exhaustive generic-frame design |
| Ongoing/progressive condition | English progressive | Korean progressive auxiliary | `éppen` ongoing-event context |
| Clausal complement | VerbNet clausal frames | generic clausal frame | finite direct-object `hogy` frame |
| Object definiteness split | n/a | n/a | indefinite vs. definite object/conjugation |

Korean has 19 generic templates. Hungarian has 28 because the seven Korean-style ACC frames are each split into Hungarian indefinite and definite realizations, and the transitive temporal templates are also split by object definiteness.

The labels *intransitive* and *transitive* are retained where they align with the Korean frame names. In Stage 1, they describe the **surface frame** (no overt ACC object vs. overt ACC object), not a claim about a verb's lexical valency.

## The 28 Hungarian frames

`FRAME_REFERENCE.csv` contains every frame, a grammatical Hungarian example, and an English translation. The examples illustrate the frame shape; they do not claim that every matrix verb is acceptable in that frame.

The inventory consists of:

- 7 frames without an overt accusative object;
- 7 accusative-object structures × Hungarian indefinite/definite realization = 14 templates;
- 1 finite direct-object `hogy` complement frame;
- 6 `éppen` present/past templates, including the Hungarian definiteness split in transitive frames.

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
- `object_definiteness`
- `clausal_complement`
- `ongoing_context`
- `tense`
- `lexical_verb` (different-verb pairs)

For different-verb pairs, the template, non-verb fillers, and relevant inflectional features are held constant. For same-verb pairs, the shared lexical material is held constant and only the licensed contrast-specific change is allowed.

Language-model scores are used **only for stratification**. They are not grammaticality labels and do not filter out verb-frame mismatches.

## Running the pipeline

From `hun/argument_structure`:

```bash
python generate_lexicons.py
python generate_templates.py
python fill_templates.py
python create_2afc_pairs.py
python generate_lists.py
python generate_deployment.py
```

For development or small-scale inspection, `fill_templates.py` accepts `--verb-start`, `--verb-count`, and `--max-per-template`. Omitting `--verb-count` processes the full available verb inventory. `create_2afc_pairs.py --skip-scoring` is available for structural inspection; final experiment preparation should run pair generation with language-model scoring enabled.

## Human linguistic validation

The `validation/` utilities support manual review of generated stimuli. They distinguish intended low-acceptability outcomes such as `verb_frame_mismatch` from unintended `generator_error` cases. This keeps linguistic acceptability judgments separate from mechanical generation checks and language-model stratification.

## Stage 2

Stage 2 is intentionally separate from this cross-linguistic baseline. Candidate extensions include pro-drop/argument omission, richer local-case alternations, preverbs and event structure, focus/information structure, potential mood, and more detailed complementation patterns.
