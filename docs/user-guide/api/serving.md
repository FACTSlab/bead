# Read-only predictors

`bead.serving` defines the serving contract for externally trained predictors.
Providers implement ordinary Python classes. They do not inherit
`ActiveLearningModel`, implement training or random effects, or supply unrelated
language-model scoring and embedding operations.

## Compose capabilities

The public protocols are independent:

- `ModelIdentity` provides `model_name` and `model_version`.
- `TaskCompatibility` provides `supported_task_types` and
  `validate_item_compatibility(item, item_template)`.
- `ProbabilisticPredictor` provides `response_encoding` and
  `predict_proba(items, participant_ids=None)`.
- `PersistablePredictor` provides `save(path)` and class method `load(path)`.
- `NLICapability` provides `compute_nli(premise, hypothesis)`.

Type a consumer against the smallest protocol it uses. The `ServingPredictor`
protocol composes all five capabilities for consumers that require the complete
read-only surface and for the conformance suite.

The `response_encoding` property defines probability-column semantics. A
predictor returning a matrix `probabilities` must make
`probabilities[:, i]` the probability of
`response_encoding.labels[i]`. The matrix has shape
`(len(items), response_encoding.n_levels)`, contains finite values in the closed
interval `[0, 1]`, and has rows that sum to one.

## Register a loading factory

The factory registry accepts a callable, not a subclass. For a provider class
named `LoFIPredictor`, register its native-artifact class method directly:

```python
from pathlib import Path
from typing import Self

from bead.serving import PredictorRegistry


class LoFIPredictor:
    """Example predictor restored from its native artifact."""

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls()


registry = PredictorRegistry[LoFIPredictor]()
registry.register("lofi", LoFIPredictor.load)

predictor = registry.load("lofi", "artifacts/judgment-model")

assert isinstance(predictor, LoFIPredictor)
```

`LoFIPredictor` only needs to implement the structural protocol. It does not
need to import or subclass a bead implementation class.

## Run the conformance suite

Third-party packages should run the published suite in their own tests with
real bead values:

```python
from pathlib import Path
from typing import Self

import numpy as np

from bead.items import Item, ItemTemplate, PresentationSpec, TaskSpec
from bead.protocol import ResponseEncoding, ScaleType
from bead.serving import NLICase, PredictionCase, run_predictor_conformance


class ExamplePredictor:
    """Minimal complete serving predictor."""

    model_name = "example"
    model_version = "1.0.0"
    supported_task_types = ("binary",)
    response_encoding = ResponseEncoding(
        name="inference",
        n_levels=2,
        scale_type=ScaleType.BINARY,
        labels=("no", "yes"),
    )

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        if item.item_template_id != item_template.id:
            raise ValueError("item and template do not match")

    def predict_proba(
        self,
        items: list[Item],
        participant_ids: list[str] | None = None,
    ) -> np.ndarray:
        return np.tile([0.25, 0.75], (len(items), 1))

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
        return cls()

    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        return {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}


item_template = ItemTemplate(
    name="binary-inference",
    judgment_type="inference",
    task_type="binary",
    task_spec=TaskSpec(
        prompt="Does the hypothesis follow?",
        options=("no", "yes"),
    ),
    presentation_spec=PresentationSpec(),
)
item = Item(item_template_id=item_template.id, rendered_elements={})
predictor = ExamplePredictor()

report = run_predictor_conformance(
    predictor,
    prediction_cases=[
        PredictionCase(item=item, item_template=item_template),
    ],
    response_encoding=predictor.response_encoding,
    nli_cases=[
        NLICase(
            premise="A dog is running.",
            hypothesis="An animal is moving.",
        ),
    ],
)

assert report.checks == (
    "identity",
    "task_compatibility",
    "predict_proba",
    "persistence",
    "nli",
)
```

The suite verifies the item-template link with bead's standard item validator,
checks that the encoding maps to the template task type, calls compatibility
validation, tests prediction with and without participant identifiers, validates
probability shape and normalization, and requires prediction and NLI equality
after a native save-load round trip.
