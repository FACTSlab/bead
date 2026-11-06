# bead.active_learning

Stage 6 of the bead pipeline: active learning with GLMM support and convergence detection.

## Active Learning Loop

::: bead.active_learning.loop
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.selection
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.strategies
    options:
      show_root_heading: true
      show_source: false

## Configuration

::: bead.active_learning.config
    options:
      show_root_heading: true
      show_source: false

## Model Registry

Single canonical task-type → model-class and task-type → config-class
mapping used by the CLI training commands and the protocol-encoding
factory.

::: bead.active_learning.models.registry
    options:
      show_root_heading: true
      show_source: false

## Base Model Interface

::: bead.active_learning.models.base
    options:
      show_root_heading: true
      show_source: false

## Task-Specific Models

### Forced-choice scoring

`ForcedChoiceModelConfig` separates encoding from scoring. The default
`scoring_mode="joint"` preserves the existing classifier: it predicts all option
logits from either one encoding of the concatenated texts (`single_encoder`) or
concatenated per-option embeddings (`dual_encoder`).

With `scoring_mode="independent"`, each option is encoded separately and passed
through the same scalar head. For option texts \(s_1,\ldots,s_K\), the model uses

\[
a_j=f_\phi(E_\theta(s_j)),\qquad
P(y=j)=\frac{\exp(a_j)}{\sum_{k=1}^{K}\exp(a_k)}.
\]

The head has a 256-unit ReLU hidden layer, dropout of 0.1 during training, and one
scalar output. For two options, the first option's probability is
\(\sigma(a_1-a_2)\). Scores are defined up to a common additive constant; the
logistic noise scale is fixed to one. Independent scoring requires
`encoder_mode="dual_encoder"`; combining it with `single_encoder` raises a
configuration validation error.

```python
from uuid import uuid4

from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.config.active_learning import ForcedChoiceModelConfig
from bead.items.item import Item

config = ForcedChoiceModelConfig(
    model_name="bert-base-uncased",
    encoder_mode="dual_encoder",
    scoring_mode="independent",
    num_epochs=1,
)
items = [
    Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "The cat sleeps.", "option_b": "Cat the sleeps."},
    ),
    Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "Dogs the bark.", "option_b": "The dogs bark."},
    ),
]
model = ForcedChoiceModel(config)
metrics = model.train(items, ["option_a", "option_b"])
probabilities = model.predict_proba(items)
```

In independent mode, every `rendered_elements` entry is an alternative. The
model infers option names from the first training item, sorts them for probability
column order, and requires the same names on subsequent training, validation,
and prediction items. An option remains in the denominator even when no training
response selects it. Each fitted model has a fixed option vocabulary; the scalar
head's parameter shapes do not depend on the number of alternatives.

At prediction time, swapping option texts swaps their probabilities, identical
texts receive equal scores, and changing a third option leaves the odds between
the first two unchanged. These properties hold for the shared head and for each
participant-specific scalar head. Position-specific intercepts can break the
swap and tie properties.

The existing `mixed_effects.mode` options also apply:

- `fixed`: one scalar head shared across participants.
- `random_intercepts`: a participant-specific bias for each option position is
  added to the scalar scores. A common scalar bias would cancel under softmax.
- `random_slopes`: a participant-specific copy of the scalar head scores every
  option. The existing head initialization and regularization rules apply.

Unseen participants use the shared head, with equal prior-mean biases in
intercept mode. These are fallback predictions, not averages over participant
effects. Dual-encoder training uses the same input preparation as prediction,
includes participant parameters in the optimizer, and supports held-out
cross-entropy early stopping. Checkpoints retain both architecture settings;
older checkpoints without `scoring_mode` load as `joint`.

The argument-structure gallery accepts `encoder_mode: dual_encoder` and
`scoring_mode: independent` in its `acceptability_model` configuration section.

::: bead.active_learning.models.forced_choice
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.ordinal_scale
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.binary
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.categorical
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.multi_select
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.magnitude
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.free_text
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.cloze
    options:
      show_root_heading: true
      show_source: false

## Random Effects and LoRA

::: bead.active_learning.models.random_effects
    options:
      show_root_heading: true
      show_source: false

::: bead.active_learning.models.lora
    options:
      show_root_heading: true
      show_source: false
