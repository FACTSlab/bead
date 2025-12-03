# Items Module

The items module handles Stage 3 of the pipeline: creating task-specific experimental items from filled templates.

## 8 Task Types

Bead supports eight task types, each with specialized factory functions:

1. **forced_choice**: select one option from N alternatives (2AFC, 3AFC, N-way)
2. **ordinal_scale**: rate on Likert or slider scale
3. **binary**: yes/no, true/false choice
4. **categorical**: select from unordered categories (NLI, semantic relations)
5. **multi_select**: select multiple options (checkboxes)
6. **magnitude**: enter numeric value (reading time, confidence)
7. **free_text**: open-ended text response
8. **cloze**: fill in blanks

## Forced Choice Items

Create items where participants select from alternatives.

### Basic Forced Choice

```bash
bead items create-forced-choice "Option A" "Option B" --output items/2afc.jsonl

bead items create-forced-choice "The cat sleeps" "The cat slept" "The cats sleep" \
    --metadata "source=transitive" \
    --output items/3afc.jsonl
```

Creates a single forced-choice item from the provided options (2AFC, 3AFC, or N-way).

### From Text Files

```bash
bead items create-forced-choice-from-texts \
    --texts-file sentences.txt \
    --n-alternatives 2 \
    --sample 20 \
    --output items/forced_choice.jsonl
```

Creates 2-way forced-choice items from sentences. Randomly samples 20 items from all possible combinations.

### Sampling Combinations

```bash
bead items create-forced-choice-from-texts \
    --texts-file sentences.txt \
    --n-alternatives 3 \
    --sample 10 \
    --output items/sampled.jsonl
```

Randomly samples 10 items from all possible 3-way combinations.

## Ordinal Scale Items

Create rating tasks with Likert or slider scales.

### 7-Point Likert Scale

```bash
bead items create-likert-7 \
    --text "The cat sat on the mat" \
    --output items/likert_item1.jsonl

bead items create-likert-7 \
    --text "Sentence text" \
    --prompt "How natural is this?" \
    --output items/likert_item2.jsonl
```

Creates a single 7-point Likert scale item (1 = Strongly disagree, 7 = Strongly agree).

### Custom Ordinal Scale from Texts

```bash
bead items create-ordinal-scale-from-texts \
    --texts-file sentences.txt \
    --scale-min 1 \
    --scale-max 7 \
    --output items/ordinal.jsonl

bead items create-ordinal-scale-from-texts \
    --texts-file sentences.txt \
    --scale-min 1 \
    --scale-max 5 \
    --prompt "How acceptable?" \
    --output items/likert5.jsonl
```

Creates ordinal scale items from a text file with customizable scale bounds and prompts.

## Categorical Items

Create items with unordered category selection.

### General Categorical

```bash
bead items create-categorical \
    --text "The cat is sleeping" \
    --categories "entailment,contradiction,neutral" \
    --output items/categorical.jsonl
```

Creates a categorical item with specified categories.

### NLI Items

```bash
bead items create-nli \
    --premise "All dogs bark" \
    --hypothesis "Some dogs bark" \
    --output items/nli.jsonl
```

Shorthand for Natural Language Inference with standard categories (entailment, contradiction, neutral).

## Binary Items

Create yes/no or true/false tasks.

```bash
bead items create-binary-from-texts \
    --texts-file sentences.txt \
    --output items/binary.jsonl

bead items create-binary-from-texts \
    --texts-file sentences.txt \
    --prompt "Is this grammatical?" \
    --output items/grammatical.jsonl
```

Creates binary judgment items from a text file. Default prompt is "Is this acceptable?"

## Multi-Select Items

Create checkbox-style items allowing multiple selections.

```bash
bead items create-multi-select-from-texts \
    --texts-file sentences.txt \
    --options "Agent,Patient,Theme,Goal" \
    --output items/multi_select.jsonl

bead items create-multi-select-from-texts \
    --texts-file sentences.txt \
    --options "Semantic,Syntactic,Pragmatic" \
    --min-selections 1 \
    --max-selections 2 \
    --output items/constrained.jsonl
```

Creates multi-select items from a text file. Each text becomes a stimulus with the specified options as checkboxes.

## Magnitude Items

Create numeric input tasks for unbounded measures (reading time, confidence, etc.).

```bash
bead items create-magnitude-from-texts \
    --texts-file sentences.txt \
    --output items/magnitude.jsonl

bead items create-magnitude-from-texts \
    --texts-file sentences.txt \
    --measure "reading_time_ms" \
    --prompt "Reading time (ms):" \
    --output items/reading_times.jsonl
```

Creates magnitude estimation items from a text file. Default measure is "value" with prompt "Enter value:".

## Free Text Items

Create open-ended text response tasks.

```bash
bead items create-free-text-from-texts \
    --texts-file sentences.txt \
    --output items/free_text.jsonl

bead items create-free-text-from-texts \
    --texts-file sentences.txt \
    --prompt "Paraphrase this sentence:" \
    --output items/paraphrase.jsonl
```

Creates free-text response items from a text file. Default prompt is "Provide your response:".

## Cloze Items

Create fill-in-the-blank tasks from plain text.

```bash
bead items create-simple-cloze \
    --text "The quick brown fox" \
    --blank-position 1 \
    --output items/cloze_item1.jsonl

bead items create-simple-cloze \
    --text "The cat sat on the mat" \
    --blank-position 3 \
    --blank-label "preposition" \
    --output items/cloze_item2.jsonl
```

Blanks out the specified position (zero-indexed). First example produces: "The ___ brown fox".

## Validation

Verify items conform to task type requirements:

```bash
# Validate structure
bead items validate-for-task-type items/2afc.jsonl \
    --task-type forced_choice

# Infer task type from structure
bead items infer-task-type items/2afc.jsonl

# Get requirements for task type
bead items get-task-requirements \
    --task-type ordinal_scale
```

Output from `get-task-requirements`:

```json
{
  "required_rendered_keys": ["text"],
  "required_metadata_keys": ["scale_min", "scale_max"],
  "optional_metadata_keys": ["labels", "step"]
}
```

## Metadata

All item creation commands support `--metadata key=value` for adding custom fields:

```bash
bead items create-forced-choice "Option A" "Option B" \
    --metadata "condition=experimental,block=1" \
    --output items/metadata_item.jsonl
```

Metadata flows through partitioning and deployment stages.

## Listing and Statistics

View item statistics:

```bash
# List items in directory
bead items list --directory items/

# Show statistics for an items file
bead items show-stats items/forced_choice.jsonl
```

Output includes:

```
Items: 1,250
Task types: forced_choice (800), ordinal_scale (300), binary (150)
Average alternatives (forced_choice): 2.3
Scale range (ordinal_scale): 1-7
```

## Workflow Example

Complete workflow from text files to experimental items:

```bash
# 1. Create forced-choice items from sentences
bead items create-forced-choice-from-texts \
    --texts-file sentences.txt \
    --n-alternatives 2 \
    --sample 10 \
    --output items/forced_choice.jsonl

# 2. Create Likert-7 items from same sentences
bead items create-ordinal-scale-from-texts \
    --texts-file sentences.txt \
    --scale-min 1 \
    --scale-max 7 \
    --prompt "How natural?" \
    --output items/likert7.jsonl

# 3. Validate items
bead items validate-for-task-type items/forced_choice.jsonl \
    --task-type forced_choice

# 4. View statistics
bead items show-stats items/forced_choice.jsonl
bead items show-stats items/likert7.jsonl
```

## Next Steps

Once items are created:

1. [Partition items into lists](lists.md) with constraints
2. [Deploy to jsPsych/JATOS](deployment.md)
3. [Train models](training.md) on collected responses

For complete API documentation, see [bead.items API reference](../api/items.md).
