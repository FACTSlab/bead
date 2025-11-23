# Active Learning Module

The active learning module handles Stage 6 of the pipeline: training models on collected data with optional active learning loops.

## GLMM Models

Bead uses Generalized Linear Mixed Models (GLMMs) to account for participant and item variability. Three mixed-effects modes balance complexity and data requirements.

### Fixed Effects Only

Train without random effects:

```bash
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --data responses/responses.jsonl \
    --model-name bert-base-uncased \
    --mixed-effects-mode fixed \
    --output models/fixed_model/
```

Use when participant and item variability is minimal or when sample sizes are small.

### Random Intercepts

Model participant and item baseline differences:

```bash
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --data responses/responses.jsonl \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_intercepts \
    --participant-intercept \
    --item-intercept \
    --output models/random_intercepts_model/
```

Random intercepts capture that some participants are consistently stricter/lenient and some items are consistently easier/harder.

### Random Slopes (Full Mixed Effects)

Model interactions between participants and items:

```bash
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --data responses/responses.jsonl \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_slopes \
    --participant-intercept \
    --item-intercept \
    --interaction \
    --output models/random_slopes_model/
```

Random slopes capture that participants respond differently to different items (participant × item interactions).

## Task-Type Models

All 8 task types support GLMM training:

- `forced_choice`: 2AFC, 3AFC, N-way choice
- `ordinal_scale`: Likert scales, sliders
- `binary`: yes/no, true/false
- `categorical`: NLI, semantic relations
- `multi_select`: checkbox tasks
- `magnitude`: reading time, confidence
- `free_text`: open-ended text
- `cloze`: fill-in-the-blank

Specify the task type with `--task-type`:

```bash
bead models train-model \
    --task-type ordinal_scale \
    --items items/likert7.jsonl \
    --data responses/likert_responses.jsonl \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_intercepts \
    --participant-intercept \
    --item-intercept \
    --output models/ordinal_model/
```

## LoRA Training

Use Low-Rank Adaptation for parameter-efficient fine-tuning:

```bash
bead models train-model \
    --task-type free_text \
    --items items/paraphrase.jsonl \
    --data responses/paraphrase_responses.jsonl \
    --model-name gpt2 \
    --mixed-effects-mode fixed \
    --use-lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --output models/lora_model/
```

LoRA reduces trainable parameters, enabling fine-tuning of larger models with limited compute.

## Prediction

Generate predictions from trained models:

```bash
bead models predict \
    --model models/random_intercepts_model/ \
    --items items/new_items.jsonl \
    --participant-ids participant_ids.txt \
    --output predictions/predictions.jsonl
```

The `--participant-ids` file contains one participant ID per line, matching the order of items.

### Probability Predictions

Get class probability distributions:

```bash
bead models predict-proba \
    --model models/random_intercepts_model/ \
    --items items/new_items.jsonl \
    --participant-ids participant_ids.txt \
    --output predictions/probabilities.json
```

Output is a JSON array of probability vectors.

## Convergence Detection

Check if model performance matches human inter-annotator agreement:

```bash
bead active-learning check-convergence \
    --predictions predictions/model_preds.jsonl \
    --human-labels responses/human_responses.jsonl \
    --metric krippendorff_alpha \
    --threshold 0.80
```

Output:

```
Krippendorff's alpha: 0.823
✓ Convergence threshold met (0.80)
Model performance matches human agreement
```

Supported metrics:

- **krippendorff_alpha**: inter-annotator agreement (works with all task types)
- **fleiss_kappa**: multi-rater agreement (categorical data)
- **cohens_kappa**: two-rater agreement (pairwise)

## Evaluation

### Model Evaluation

Compute standard metrics:

```bash
bead training evaluate \
    --model models/random_intercepts_model/ \
    --test-data responses/test_set.jsonl \
    --metrics accuracy,f1,precision,recall \
    --output evaluation/metrics.json
```

### Cross-Validation

k-fold cross-validation with stratification:

```bash
bead training cross-validate \
    --items items/all.jsonl \
    --data responses/all_responses.jsonl \
    --model-config config/model_config.yaml \
    --k-folds 5 \
    --stratify-by participant_id \
    --output evaluation/cv_results.json
```

Stratifying by `participant_id` ensures participants don't appear in both train and test sets.

### Learning Curve

Plot performance vs training set size:

```bash
bead training learning-curve \
    --items items/all.jsonl \
    --data responses/all_responses.jsonl \
    --model-config config/model_config.yaml \
    --train-sizes 0.1,0.2,0.5,0.8,1.0 \
    --output evaluation/learning_curve.png
```

## Inter-Annotator Agreement

Compute agreement among human annotators:

```bash
# Krippendorff's alpha (works with all task types)
bead training compute-agreement \
    --annotations responses/multi_annotator.jsonl \
    --metric krippendorff_alpha \
    --task-type ordinal_scale

# Fleiss' kappa (categorical data, multiple raters)
bead training compute-agreement \
    --annotations responses/multi_annotator.jsonl \
    --metric fleiss_kappa \
    --task-type categorical

# Cohen's kappa (pairwise agreement)
bead training compute-agreement \
    --annotations responses/two_annotators.jsonl \
    --metric cohens_kappa \
    --task-type binary \
    --pairwise
```

## Data Collection

After deployment, collect responses from JATOS:

```bash
bead training collect-data \
    --server https://jatos.example.com \
    --study-id 123 \
    --output responses/raw_responses.jsonl
```

The command downloads all responses and converts them to bead's JSONL format.

## Workflow Example

Complete training and convergence detection workflow:

```bash
# 1. Collect data from JATOS
bead training collect-data \
    --server https://jatos.example.com \
    --study-id 123 \
    --output responses/collected_data.jsonl

# 2. Train GLMM model
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --data responses/collected_data.jsonl \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_intercepts \
    --participant-intercept \
    --item-intercept \
    --output models/trained_model/

# 3. Generate predictions on test set
bead models predict \
    --model models/trained_model/ \
    --items items/test_set.jsonl \
    --participant-ids test_participant_ids.txt \
    --output predictions/model_predictions.jsonl

# 4. Check convergence to human agreement
bead active-learning check-convergence \
    --predictions predictions/model_predictions.jsonl \
    --human-labels responses/gold_standard.jsonl \
    --metric krippendorff_alpha \
    --threshold 0.80

# 5. Evaluate model performance
bead training evaluate \
    --model models/trained_model/ \
    --test-data responses/test_set.jsonl \
    --metrics accuracy,f1,precision,recall \
    --output evaluation/metrics.json
```

## Next Steps

After training:

1. [Evaluate models](active-learning.md#evaluation) with cross-validation
2. Deploy models for real-time prediction
3. Analyze learned representations

For complete API documentation, see [bead.active_learning API reference](../api/active_learning.md).
