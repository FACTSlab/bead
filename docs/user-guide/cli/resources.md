# Resources Module

The resources module handles Stage 1 of the pipeline: creating lexical items, templates, and constraints.

## Creating Lexicons

Lexicons store collections of lexical items (words or phrases) with linguistic features.

### From CSV or JSON

Convert structured data to JSONL format:

```bash
bead resources create-lexicon \
    --input words.csv \
    --name my_lexicon \
    --language-code eng \
    --output lexicons/words.jsonl
```

The CSV file must include columns for `lemma` and `pos` (part of speech). Additional columns become features in the `metadata` dictionary.

Example CSV:

```csv
lemma,pos,frequency,animacy
cat,N,1000,animate
book,N,850,inanimate
run,V,1200,
```

### From VerbNet

Import verbs directly from the VerbNet database via the glazing adapter:

```bash
bead resources import-verbnet \
    --verb-class put-9.1 \
    --output lexicons/put_verbs.jsonl \
    --limit 20
```

Each verb includes VerbNet class information, thematic roles, and frame counts in its metadata. Use `--query` to search for specific verbs:

```bash
bead resources import-verbnet \
    --query "run" \
    --output lexicons/run_verbs.jsonl
```

### From UniMorph

Generate inflected forms with morphological features:

```bash
bead resources import-unimorph \
    --language eng \
    --pos VERB \
    --features "V;PST" \
    --output lexicons/past_tense_verbs.jsonl \
    --limit 100
```

The `--features` parameter uses UniMorph notation. Common feature combinations:

- `V;PST`: past tense verbs
- `V;PRS;3;SG`: present tense, 3rd person singular
- `N;PL`: plural nouns
- `ADJ;CMPR`: comparative adjectives

### From PropBank

Import predicate-argument structures from PropBank:

```bash
bead resources import-propbank \
    --frameset eat.01 \
    --output lexicons/eating_verbs.jsonl
```

PropBank framesets include role labels (ARG0, ARG1, etc.) in metadata.

### From FrameNet

Import frame semantic information:

```bash
bead resources import-framenet \
    --frame Ingestion \
    --output lexicons/ingestion_frame.jsonl
```

FrameNet frames include frame elements and semantic relationships.

## Creating Templates

Templates define sentence patterns with slots for lexical items.

### From Patterns

Generate templates from text patterns with `{placeholder}` syntax:

```bash
bead resources generate-templates \
    --from-pattern "{det} {subj} {verb} {det} {obj}" \
    --slot subj:required \
    --slot verb:required \
    --slot obj:required \
    --slot det:optional \
    --description "Basic transitive sentence frame" \
    --language-code eng \
    --output templates/transitive.jsonl
```

The command auto-detects slots from curly braces. Use `--slot name:required` or `--slot name:optional` to specify requirements explicitly.

Multiple patterns in one command:

```bash
bead resources generate-templates \
    --from-pattern "{subj} {verb} {obj}" \
    --from-pattern "{subj} {aux} {verb} by {agent}" \
    --output templates/multiple_frames.jsonl
```

### Template Variants

Generate systematic variations from base templates:

```bash
bead resources generate-template-variants \
    --base-template templates/transitive.jsonl \
    --name-pattern "{base_name}_variant_{index}" \
    --max-variants 5 \
    --output templates/transitive_variants.jsonl
```

Each variant receives metadata tracking its base template and variant index.

## Creating Constraints

Constraints restrict which lexical items can fill template slots. Three constraint types support different use cases.

### Extensional Constraints

Whitelist specific items for a slot:

```bash
# From comma-separated values
bead resources create-constraint \
    --type extensional \
    --property lemma \
    --values "run,walk,jump" \
    --output constraints/motion_verbs.jsonl

# From file (one value per line)
bead resources create-constraint \
    --type extensional \
    --property lemma \
    --values-file motion_verbs.txt \
    --output constraints/motion_verbs.jsonl
```

Extensional constraints define explicit membership: only listed items can fill the slot.

### Intensional Constraints

Use DSL expressions for feature-based filtering:

```bash
bead resources create-constraint \
    --type intensional \
    --expression "self.features.get('pos') == 'V'" \
    --description "Only verbs" \
    --output constraints/verb_constraint.jsonl
```

The DSL supports:

- Feature access: `self.features.get('pos')`
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Boolean logic: `and`, `or`, `not`
- Membership: `in`, `not in`

Example with multiple conditions:

```bash
bead resources create-constraint \
    --type intensional \
    --expression "self.features.get('pos') == 'N' and self.features.get('animacy') == 'animate'" \
    --description "Animate nouns only" \
    --output constraints/animate_nouns.jsonl
```

### Relational Constraints

Define relationships across multiple slots:

```bash
bead resources create-constraint \
    --type relational \
    --relation "slot1.lemma != slot2.lemma" \
    --description "Subject and object must differ" \
    --output constraints/different_args.jsonl
```

Relational constraints reference slots by name. Use `slot1`, `slot2`, etc., or actual slot names if known.

Agreement constraint example:

```bash
bead resources create-constraint \
    --type relational \
    --relation "subj.features.get('number') == verb.features.get('number')" \
    --description "Subject-verb number agreement" \
    --output constraints/agreement.jsonl
```

## Validation

Verify lexicons and templates before using them:

```bash
# Validate lexicon format
bead resources validate-lexicon \
    --lexicon lexicons/words.jsonl

# Validate template structure
bead resources validate-template \
    --template templates/frames.jsonl
```

Validation checks:

- JSONL format correctness
- Required fields present (id, lemma/pattern, language_code)
- Feature dictionaries well-formed
- Slot names consistent

## Listing Resources

View available lexicons and templates:

```bash
# List all lexicons in directory
bead resources list-lexicons \
    --directory lexicons/

# List all templates
bead resources list-templates \
    --directory templates/
```

Output shows file paths, item counts, and language codes.

## JSONL Format

All resource files use JSONL (JSON Lines) format: one JSON object per line. This enables streaming and partial loading for large datasets.

Example lexicon file:

```jsonl
{"id": "uuid1", "lemma": "run", "pos": "V", "features": {"tense": "base"}, "metadata": {}, "created_at": "2025-01-20T10:00:00Z", "modified_at": "2025-01-20T10:00:00Z"}
{"id": "uuid2", "lemma": "ran", "pos": "V", "features": {"tense": "past"}, "metadata": {"base_form": "run"}, "created_at": "2025-01-20T10:00:01Z", "modified_at": "2025-01-20T10:00:01Z"}
```

Example template file:

```jsonl
{"id": "uuid3", "pattern": "{subj} {verb} {obj}", "slots": {"subj": {"required": true}, "verb": {"required": true}, "obj": {"required": true}}, "description": "Transitive frame", "language_code": "eng", "created_at": "2025-01-20T10:00:00Z", "modified_at": "2025-01-20T10:00:00Z"}
```

## Integration with External Resources

The import commands (VerbNet, UniMorph, PropBank, FrameNet) cache results to avoid repeated network requests. Cache location: `.cache/bead/adapters/`.

To force refresh:

```bash
rm -rf .cache/bead/adapters/
```

All external resources are accessed via the glazing adapter library, which provides unified interfaces to linguistic databases.

## Next Steps

Once you have lexicons and templates:

1. [Fill templates](templates.md) with lexical items
2. [Create experimental items](items.md) from filled templates
3. [Apply constraints](templates.md#constraint-satisfaction) during filling

For complete API documentation, see [bead.resources API reference](../api/resources.md).
