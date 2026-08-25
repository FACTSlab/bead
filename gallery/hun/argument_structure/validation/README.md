# Hungarian linguistic validation

The scripts in this directory support manual review of generated Hungarian stimuli.
They distinguish intended low-acceptability verb-frame combinations from unintended
generator errors.

Build a balanced review sheet with:

```bash
python validation/build_human_review.py --per-template 3
```

Judgment labels:

- `natural`
- `acceptable_but_marked`
- `semantically_odd`
- `verb_frame_mismatch`
- `generator_error`
- `uncertain`

`verb_frame_mismatch` is potentially useful experimental signal. `generator_error`
marks unintended problems such as wrong case or conjugation, malformed spacing, a
missing constituent, or an unrelated filler problem.

The English-translation column is for discussion and documentation; Hungarian
acceptability judgments should be based on the Hungarian stimulus.
