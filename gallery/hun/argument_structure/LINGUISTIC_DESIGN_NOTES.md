# Linguistic design notes for Hungarian Stage 1

These notes record the choices that are linguistically non-trivial, so the code
is not treated as if every surface difference were language-neutral.

## 1. Definite/indefinite object agreement

Hungarian finite verbs contrast subjective/indefinite and objective/definite
conjugations. The Stage-1 transitive frames therefore come in paired INDF and
DEF realizations. The object noun, case, and other arguments stay controlled;
the determiner and verbal object agreement change together.

Useful background: Coppock, Elizabeth & Stephen Wechsler. 2012. “The objective
conjugation in Hungarian: agreement without phi-features.” *Natural Language &
Linguistic Theory* 30(3): 699–740. DOI 10.1007/s11049-012-9165-5.

## 2. What the `hogy` frame means

The Stage-1 `hogy` template is **not** meant to represent every possible
Hungarian clause introduced by `hogy`. It is a candidate **direct-object clause**
frame.

Direct-object finite clauses trigger objective/definite matrix conjugation. For
example, the literature gives patterns like `mondta, hogy ...` and `akarja, hogy
...`. By contrast, verbs whose clausal argument corresponds to an oblique
argument can remain in the subjective/indefinite conjugation (for example,
verbs with a delative/other oblique valency pattern). Those are a distinct
argument-structure class.

For this reason, the Stage-1 pair

`bare INDF verb` ↔ `DEF verb + hogy clause`

is one structural manipulation: adding a direct-object CP in Hungarian entails
the corresponding object-agreement realization. It is not intended as a
one-token string edit.

A useful Hungarian-specific overview is Tibor Laczkó (2021), “On the inventory
of grammatical functions in LFG from a Hungarian perspective,” *Proceedings of
LFG’21*, especially the discussion of OBJ vs OBL clausal complements.

## 3. Neutral locative baseline

Stage 1 uses `egy helyen` (SUP) as a neutral controlled “at/in a place”
locative realization. `Egy helyben` is avoided because it has a stronger
“in one place / staying in place” reading.

## 4. Instrument baseline

The controlled instrument noun is `eszköz`, yielding `egy eszközzel` (“with a
tool”). Keeping it distinct from the direct-object noun reduces avoidable
semantic oddness without encoding which verbs license an instrumental dependent.

Controlled slots are pinned **by lemma** in `generate_templates.py`
(`CONTROLLED_LEMMAS`). They were previously selected by semantic class alone,
which behaved as a control only because each class happened to contain exactly
one noun; adding a second location or instrument noun would have quietly
un-controlled the baseline without any code change. The test suite asserts the
pinning, and separately asserts that the slots documented as *varying* draw on
more than one lemma, so neither kind of slot can drift into the other.

## 4a. Instrumental and comitative are syncretic

Korean distinguishes an instrumental (으)로 from a comitative 와/과. Hungarian
uses INS `-val/-vel` for both. The Stage-1 inventory therefore contains an
instrumental frame and a comitative frame that share a case and differ only in
the semantic class of the noun (`egy eszközzel` vs `egy emberrel`). This is a
property of Hungarian, not an unfilled cell: the two frames are not
morphologically distinguishable, and any contrast between them is lexical.

## 4b. Postpositions

Korean's spatial relational nouns (위, 앞, 뒤) combined with 에 / 에서 / 로
correspond to Hungarian bare postpositions, which govern a caseless complement
and come in the same three-way series: `alatt` / `alá` / `alól`,
`mögött` / `mögé` / `mögül`. The Hungarian frames use that series directly
rather than reconstructing the Korean particle contrast.

Korean's complex postpositions (에 대해서, 을 통해서) correspond to Hungarian
case-governing postpositions. The governed case varies by postposition
(`képest` takes ALL, `együtt` takes INS, `keresztül` takes SUP, `szerint` takes
a caseless NOM), so the template leaves the complement's case open and a
cross-slot constraint ties it to the postposition's `gov_case` feature. This is
the Hungarian counterpart of the `fc_agree` constraint Korean uses for particle
allomorphy: the requirement lives in the lexicon, not hard-coded per frame, so
adding a postposition does not require editing a template.

## 5. `éppen` is not a progressive morpheme

Hungarian does not have a direct morphological counterpart of Korean `-고 있다`
or English `be V-ing`. The `éppen` templates are temporal/ongoing-event
diagnostics, not a claim that Hungarian has a dedicated progressive tense.

Preverbs can interact with aspect and word order in these environments. Because
preverb behavior is planned as a Hungarian-specific Stage-2 topic, the 2AFC
generator conservatively keeps transparently prefixed verbs out of the
Stage-1 `éppen`/tense contrasts while retaining them in the nominal and clausal
argument-structure frames.

## 6. What automatic tests can and cannot establish

Automatic tests establish that a frame was realized as specified: case,
agreement, determiner choice, tense, word boundaries, controlled fillers,
postposition government, and 2AFC slot invariants. They do **not** establish
that every matrix verb is acceptable in every frame. If all verb-frame outputs
were forced to be natural, the generator would have already encoded the valency
facts that the experiment is supposed to measure.

The suite lives in `tests/` and also guards two failure modes that are
otherwise silent:

- **Column drift.** `generate_lexicons.py` names the CSV columns it wants and
  skips any that are absent without error. Three resource files had drifted:
  `case_markers.csv` supplied `harmony` where the loader read `harmony_pattern`,
  so every case marker reached the lexicon carrying only `pos` and `case`.
  `tests/test_resources.py` now asserts the contract in both directions.
- **Documentation drift.** `resources/bleached_nouns.csv` and
  `FRAME_REFERENCE.csv` are generated, and the tests re-run their generators and
  compare, so a hand edit cannot diverge unnoticed.
- **Wiring drift.** `fill_templates.py` loads only the lexicons named in
  `config.yaml`. A lexicon that is generated but not declared is loaded by
  nothing, and every frame depending on it produces no candidate filler; the
  filler logs "No fills" and continues, so the frame family silently yields zero
  sentences. `tests/test_config.py` asserts the two lists match.
- **Constraint dialect.** Constraints are strings evaluated by bead's DSL, not
  by Python. The grammar has no `is` operator and its boolean literals are
  lowercase `true`/`false`, so `is True` parses in Python and fails in the DSL.
  Every constraint the generator emits is linted for this.

## 7. Word order is a parameter, not an assumption

The Stage-1 frames place arguments between the subject and the verb. In
Hungarian the immediately preverbal position is the focus position, so this is
a marked order, and stacking several constituents there is more marked still.
For an acceptability experiment that risks depressing ratings across the board
and interacting with the verb-frame manipulation being measured.

Rather than silently rewrite the frames, `generate_templates.py` takes
`--word-order`. `preverbal` is the default and reproduces the original Stage-1
strings exactly; `neutral` puts the finite verb directly after the subject.
Both orders are generated from one frame inventory, so the question can be
settled with a pilot rather than by assertion. `éppen` stays adjacent to the
subject under both, because it is the element that creates the ongoing reading.
