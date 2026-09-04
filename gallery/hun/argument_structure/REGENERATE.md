# Regenerate before use

The generated artefacts (`lexicons/*.jsonl`, `templates/generic_frames.jsonl`,
`filled_templates/*.jsonl`) are **not** included. The copies in the archive were
built from the previous resource CSVs and the old 28-frame inventory, and are
now stale: the CSVs gained columns and lemmas, and the inventory is 55 frames.
Shipping the old ones would be worse than shipping none.

```bash
make check          # generated-resource check + lint + tests   <- start here
make samples        # one rendered example per frame
make resources      # rebuild resources/bleached_nouns.csv
make lexicons       # downloads Hungarian UniMorph on first run
make templates      # 55 frames
make data           # full quick run (100 verbs)
```

`make check` and `make samples` need neither UniMorph nor a language model, so
run those first — they will tell you whether the template layer is behaving
before you spend time on the verb lexicon.

## Things to look at

- **Constraints are not Python.** They are strings evaluated by bead's DSL
  (`bead/dsl/grammar.lark`), which has no `is` operator and uses lowercase
  `true`/`false`. The postposition slots use `== true`. If bead's evaluator
  turns out to reject `.features.get('stage1') == true` for a reason not
  visible from the grammar, drop that clause — every row in both postposition
  CSVs is `stage1=true`, so it is a guard for future rows, not load-bearing.
- **`lexicons/verbs.jsonl` was 88 MB and tracked in git.** Fully regenerable.
  Worth reconsidering separately; `.gitignore` was left alone since changing
  what is tracked is your call.
- **Six frames have no same-verb 2AFC contrast** — the lative and ablative
  spatial series. See the note in README.md; `tests/test_config.py` pins the
  set so it cannot drift.
