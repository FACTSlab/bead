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
agreement, determiner choice, tense, word boundaries, controlled fillers, and
2AFC slot invariants. They do **not** establish that every matrix verb is
acceptable in every frame. If all verb-frame outputs were forced to be natural,
the generator would have already encoded the valency facts that the experiment
is supposed to measure.
