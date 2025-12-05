# Active Learning Module

The active learning module handles Stage 6 of the pipeline: training models on collected data with optional active learning loops.

## GLMM Models

Bead uses Generalized Linear Mixed Models (GLMMs) to account for participant and item variability. Three mixed-effects modes balance complexity and data requirements.

### Fixed Effects Only

Train without random effects:

<!--pytest.mark.skip(reason="requires model training infrastructure")-->
```bash
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --labels responses/labels.jsonl \
    --model-name bert-base-uncased \
    --mixed-effects-mode fixed \
    --output-dir models/fixed_model/
```

Use when participant and item variability is minimal or when sample sizes are small.

### Random Intercepts

Model participant and item baseline differences:

<!--pytest.mark.skip(reason="requires model training infrastructure")-->
```bash
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --labels responses/labels.jsonl \
    --participant-ids responses/participant_ids.txt \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_intercepts \
    --output-dir models/random_intercepts_model/
```

Random intercepts capture that some participants are consistently stricter/lenient and some items are consistently easier/harder.

### Random Slopes (Full Mixed Effects)

Model interactions between participants and items:

<!--pytest.mark.skip(reason="requires model training infrastructure")-->
```bash
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --labels responses/labels.jsonl \
    --participant-ids responses/participant_ids.txt \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_slopes \
    --output-dir models/random_slopes_model/
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

<!--pytest.mark.skip(reason="requires model training infrastructure")-->
```bash
bead models train-model \
    --task-type ordinal_scale \
    --items items/likert7.jsonl \
    --labels responses/likert_labels.jsonl \
    --participant-ids responses/participant_ids.txt \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_intercepts \
    --output-dir models/ordinal_model/
```

## LoRA Training

Use Low-Rank Adaptation for parameter-efficient fine-tuning:

<!--pytest.mark.skip(reason="requires model training infrastructure")-->
```bash
bead models train-model \
    --task-type free_text \
    --items items/paraphrase.jsonl \
    --labels responses/paraphrase_labels.jsonl \
    --model-name gpt2 \
    --mixed-effects-mode fixed \
    --use-lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --output-dir models/lora_model/
```

LoRA reduces trainable parameters, enabling fine-tuning of larger models with limited compute.

## Prediction

Generate predictions from trained models:

<!--pytest.mark.skip(reason="requires trained model")-->
```bash
bead models predict \
    --model-dir models/random_intercepts_model/ \
    --items items/new_items.jsonl \
    --participant-ids participant_ids.txt \
    --output predictions/predictions.jsonl
```

The `--participant-ids` file contains one participant ID per line, matching the order of items.

### Probability Predictions

Get class probability distributions:

<!--pytest.mark.skip(reason="requires trained model")-->
```bash
bead models predict-proba \
    --model-dir models/random_intercepts_model/ \
    --items items/new_items.jsonl \
    --participant-ids participant_ids.txt \
    --output predictions/probabilities.json
```

Output is a JSON array of probability vectors.

## Convergence Detection

Check if model performance matches human inter-annotator agreement:

<!--pytest.mark.skip(reason="requires model predictions")-->
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

<!--pytest.mark.skip(reason="requires trained model")-->
```bash
bead training evaluate \
    --model-dir models/random_intercepts_model/ \
    --test-items items/test_set.jsonl \
    --test-labels responses/test_labels.jsonl \
    --metrics accuracy,f1,precision,recall \
    --output evaluation/metrics.json
```

### Cross-Validation

k-fold cross-validation with stratification:

<!--pytest.mark.skip(reason="requires model training infrastructure")-->
```bash
bead training cross-validate \
    --items items/all.jsonl \
    --labels responses/all_labels.jsonl \
    --model-config config/model_config.yaml \
    --k-folds 5 \
    --stratify-by participant_id \
    --output evaluation/cv_results.json
```

Stratifying by `participant_id` ensures participants don't appear in both train and test sets.

### Learning Curve

Plot performance vs training set size:

<!--pytest.mark.skip(reason="requires model training infrastructure")-->
```bash
bead training learning-curve \
    --items items/all.jsonl \
    --labels responses/all_labels.jsonl \
    --model-config config/model_config.yaml \
    --train-sizes 0.1,0.2,0.5,0.8,1.0 \
    --output evaluation/learning_curve.json
```

## Inter-Annotator Agreement

Compute agreement among human annotators:

```bash
# Krippendorff's alpha (works with all task types)
bead training compute-agreement \
    --annotations responses/multi_annotator.jsonl \
    --metric krippendorff_alpha \
    --data-type ordinal
```

```bash
# Fleiss' kappa (categorical data, multiple raters)
bead training compute-agreement \
    --annotations responses/multi_annotator.jsonl \
    --metric fleiss_kappa
```

```bash
# Cohen's kappa (pairwise agreement)
bead training compute-agreement \
    --annotations responses/two_annotators.jsonl \
    --metric cohens_kappa
```

## Data Collection

After deployment, collect responses from JATOS:

<!--pytest.mark.skip(reason="requires external JATOS server")-->
```bash
bead training collect-data responses/raw_responses.jsonl \
    --jatos-url https://jatos.example.com \
    --api-token your-api-token \
    --study-id 123
```

The command downloads all responses and converts them to bead's JSONL format.

## Workflow Example

Complete training and convergence detection workflow:

<!--pytest.mark.skip(reason="requires external JATOS server and model training")-->
```bash
# 1. Collect data from JATOS
bead training collect-data responses/collected_data.jsonl \
    --jatos-url https://jatos.example.com \
    --api-token your-api-token \
    --study-id 123

# 2. Train GLMM model
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --labels responses/labels.jsonl \
    --participant-ids responses/participant_ids.txt \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_intercepts \
    --output-dir models/trained_model/

# 3. Generate predictions on test set
bead models predict \
    --model-dir models/trained_model/ \
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
    --model-dir models/trained_model/ \
    --test-items items/test_set.jsonl \
    --test-labels responses/test_labels.jsonl \
    --metrics accuracy,f1,precision,recall \
    --output evaluation/metrics.json
```

## Next Steps

After training:

1. [Evaluate models](active-learning.md#evaluation) with cross-validation
2. Deploy models for real-time prediction
3. Analyze learned representations

For complete API documentation, see [bead.active_learning API reference](../api/active_learning.md).
