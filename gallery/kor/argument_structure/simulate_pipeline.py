#!/usr/bin/env python3
"""Simulate the active-learning loop on the Korean 2AFC pairs.

Port of the English gallery's ``simulate_pipeline.py``: simulated annotators
judge pairs from language-model scores, a forced-choice model
(``klue/roberta-base``) is trained on those judgments, and each round the
model's most uncertain pool pairs are "annotated" next, until the model's
accuracy on held-out pairs approaches the simulated annotators' own
agreement. It tests the loop's plumbing and gives a first read on whether
uncertainty sampling beats random selection, before real participants.

Differences from the English script, each fixing something that would not
work there:

- Simulated judgments come from the pairs' rarity-corrected verb scores
  (``lm_score_a`` / ``lm_score_b``). bead's LM-score annotator only reads
  ``model_outputs`` named ``lm_score`` (or metadata ``lm_score1``/``2``), so
  the scores are attached as model outputs on load, and every item is
  checked to carry both: English's pairs store ``lm_score_a``/``_b``, which
  the annotator can't find, so its fallback answers at random.
- The item template's options are ``option_a`` / ``option_b`` for both the
  annotator and the model (English's simulation uses the protocol's
  ``first`` / ``second``, which the model then looks up in
  ``rendered_elements`` and does not find).
- Human agreement is percentage agreement between two simulated annotators,
  the same scale as the model's accuracy (English compares accuracy with
  Cohen's kappa). Kappa is still reported.
- The test set is fixed and held out of the pool (English resamples it from
  the pool each round).
- Pool probabilities are computed in one batch per round and handed to
  bead's ``UncertaintySampler`` as a lookup (it asks per item).
- Temperature is applied by scaling the scores (bead's forced-choice
  simulation ignores its temperature setting).
- A ``random`` selector is available as a baseline.

Parameters live in config.yaml ``simulation``; CLI flags override.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from protocol import acceptability_family, build_protocol
from rich.console import Console
from sklearn.metrics import accuracy_score

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.active_learning.selection import UncertaintySampler
from bead.config.active_learning import (
    ForcedChoiceModelConfig,
    UncertaintySamplerConfig,
)
from bead.config.simulation import NoiseModelConfig, SimulatedAnnotatorConfig
from bead.evaluation.convergence import ConvergenceDetector
from bead.evaluation.interannotator import InterAnnotatorMetrics
from bead.items.item import Item, ModelOutput
from bead.items.item_template import ItemTemplate
from bead.protocol.items import family_to_item_template
from bead.simulation.annotators.base import SimulatedAnnotator

console = Console()

OPTIONS = ("option_a", "option_b")


def load_pairs(path: Path, score: str, temperature: float = 1.0) -> list[Item]:
    """Load pairs and attach their scores as the annotator's model outputs.

    ``score`` is the metadata prefix (``lm_score`` = rarity-corrected,
    ``lm_raw`` = raw); ``<score>_a`` / ``<score>_b`` become two ``lm_score``
    model outputs in option order, which is what bead's LM-score annotator
    reads. Scores are divided by ``temperature`` here: bead's forced-choice
    strategy samples from softmax(scores) with no temperature, and its
    temperature noise model returns forced-choice answers unchanged, so
    temperature would otherwise have no effect. softmax(s / T) is exactly a
    temperature-T softmax.
    """
    items: list[Item] = []
    with open(path) as f:
        for line in f:
            item = Item.model_validate_json(line)
            meta = item.item_metadata
            for option, side in zip(OPTIONS, ("a", "b"), strict=True):
                item = item.with_model_output(ModelOutput(
                    model_name=str(meta.get("score_model", "klue/bert-base")),
                    model_version="concrete-template-pairs",
                    operation="lm_score",
                    inputs={"option": option, "score_field": f"{score}_{side}"},
                    output=float(str(meta[f"{score}_{side}"])) / temperature,
                    cache_key=f"{item.id}:{option}:{score}",
                ))
            items.append(item)
    missing = [
        i for i in items
        if len([o for o in i.model_outputs if o.operation == "lm_score"]) != 2
        or set(i.rendered_elements) != set(OPTIONS)
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} pairs lack two lm_score outputs or option_a/option_b "
            "rendered_elements; regenerate with create_concrete_template_pairs.py"
        )
    return items


def build_item_template(config_path: Path) -> ItemTemplate:
    """Protocol-built 2AFC template with options option_a / option_b.

    Same as the real-data loop: the protocol's response labels are
    first/second, but the model reads option texts from
    ``rendered_elements`` under the template's option names.
    """
    family = acceptability_family(build_protocol(config_path))
    template = family_to_item_template(family, judgment_type="acceptability")
    return template.with_(task_spec=template.task_spec.with_(options=OPTIONS))


def make_annotator(temperature: float, seed: int) -> SimulatedAnnotator:
    """LM-score annotator with temperature noise (higher = noisier)."""
    return SimulatedAnnotator.from_config(SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=temperature),
        random_state=seed,
        fallback_to_random=False,
    ))


def describe(items: list[Item]) -> dict[str, Any]:
    """Composition of a selected batch (to spot selection collapsing)."""
    classes = Counter(str(i.item_metadata["frame_class_a"]) for i in items)
    return {
        "pair_type": dict(Counter(str(i.item_metadata["pair_type"]) for i in items)),
        "top_frame_classes": classes.most_common(3),
        "n_frame_classes": len(classes),
        "mean_abs_lm_score_diff": float(np.mean([
            float(str(i.item_metadata["lm_score_diff"])) for i in items
        ])),
    }


def run_simulation(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Run one simulated active-learning experiment and return its results."""
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    mcfg = cfg["model"]

    items = load_pairs(Path(cfg["pairs"]), cfg["score"], float(cfg["temperature"]))
    random.shuffle(items)
    test_items = items[: cfg["test_size"]]
    labeled = items[cfg["test_size"]: cfg["test_size"] + cfg["initial_size"]]
    pool = items[cfg["test_size"] + cfg["initial_size"]:]
    console.print(
        f"[bold]Pairs[/bold] {len(items)}: test {len(test_items)} (held out), "
        f"initial {len(labeled)}, pool {len(pool)}"
    )

    template = build_item_template(config_path)
    annotator = make_annotator(cfg["temperature"], seed)
    ratings = annotator.annotate_batch(labeled, template)
    test_labels_map = annotator.annotate_batch(test_items, template)
    test_labels = [test_labels_map[str(i.id)] for i in test_items]

    # Simulated "human" agreement: two independent annotators on the test set.
    temperature = float(cfg["temperature"])
    s1 = make_annotator(temperature, seed + 1000).annotate_batch(test_items, template)
    s2 = make_annotator(temperature, seed + 2000).annotate_batch(test_items, template)
    l1 = [s1[str(i.id)] for i in test_items]
    l2 = [s2[str(i.id)] for i in test_items]
    metrics = InterAnnotatorMetrics()
    human_agreement = float(np.mean([a == b for a, b in zip(l1, l2, strict=True)]))
    kappa = float(metrics.cohens_kappa(l1, l2))
    console.print(
        f"[bold]Simulated human agreement[/bold]: {human_agreement:.3f} "
        f"(Cohen's kappa {kappa:.3f}; temperature {cfg['temperature']})"
    )

    model = ForcedChoiceModel(ForcedChoiceModelConfig(
        model_name=mcfg["model_name"],
        device=mcfg["device"],
        num_epochs=int(mcfg["num_epochs"]),
        batch_size=int(mcfg["batch_size"]),
        mixed_effects=MixedEffectsConfig(mode="fixed"),
    ))
    sampler = UncertaintySampler(
        config=UncertaintySamplerConfig(method=cfg["uncertainty_method"])
    )
    detector = ConvergenceDetector(
        human_agreement_metric="percentage_agreement",
        convergence_threshold=float(cfg["convergence_threshold"]),
        min_iterations=int(cfg["min_iterations"]),
        alpha=0.05,
    )
    console.print(
        f"[bold]Model[/bold] {mcfg['model_name']} on {mcfg['device']}; "
        f"selector {cfg['selector']} ({cfg['uncertainty_method']}); "
        f"budget {cfg['budget_per_iteration']} x up to {cfg['max_iterations']} rounds\n"
    )

    rounds: list[dict[str, Any]] = []
    converged = False
    for it in range(1, int(cfg["max_iterations"]) + 1):
        t0 = time.time()
        labels = [ratings[str(i.id)] for i in labeled]
        train = model.train(labeled, labels)
        preds = [p.predicted_class for p in model.predict(test_items)]
        test_acc = float(accuracy_score(test_labels, preds))
        converged = bool(detector.check_convergence(
            model_accuracy=test_acc, iteration=it, human_agreement=human_agreement,
        ))
        row: dict[str, Any] = {
            "iteration": it, "n_labeled": len(labeled), "n_pool": len(pool),
            "train_accuracy": float(train["train_accuracy"]),
            "test_accuracy": test_acc, "gap": abs(test_acc - human_agreement),
            "seconds": 0.0,
        }
        console.print(
            f"  round {it:2d}: labeled {len(labeled):4d}  "
            f"test acc {test_acc:.3f}  gap {row['gap']:.3f}"
            + ("  [green]converged[/green]" if converged else "")
        )
        if converged or not pool:
            row["seconds"] = round(time.time() - t0, 1)
            rounds.append(row)
            break

        n = min(int(cfg["budget_per_iteration"]), len(pool))
        if cfg["selector"] == "random":
            selected = random.sample(pool, n)
        else:
            # One batched pass over the pool; the sampler asks per item.
            probs = model.predict_proba(pool)
            by_id = {str(i.id): np.asarray(p) for i, p in zip(pool, probs, strict=True)}
            selected = sampler.select(
                items=pool, model=model,
                predict_fn=lambda _m, i, by_id=by_id: by_id[str(i.id)],
                budget=n,
            )
        ratings.update(annotator.annotate_batch(selected, template))
        chosen = {str(i.id) for i in selected}
        labeled = labeled + selected
        pool = [i for i in pool if str(i.id) not in chosen]
        row["selected"] = describe(selected)
        row["seconds"] = round(time.time() - t0, 1)
        rounds.append(row)

    return {
        "config": cfg,
        "human_agreement": human_agreement,
        "human_kappa": kappa,
        "converged": converged,
        "rounds": rounds,
        "final_test_accuracy": rounds[-1]["test_accuracy"],
        "total_annotations": len(ratings),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate the Korean active-learning loop with LM-score annotators"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--initial-size", type=int)
    parser.add_argument("--budget", type=int, dest="budget_per_iteration")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--selector", choices=["uncertainty", "random"])
    parser.add_argument("--score", choices=["lm_score", "lm_raw"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["simulation"]
    for key in ("initial_size", "budget_per_iteration", "max_iterations", "test_size",
                "temperature", "selector", "seed", "score"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.epochs is not None:
        cfg["model"]["num_epochs"] = args.epochs
    if args.device is not None:
        cfg["model"]["device"] = args.device

    results = run_simulation(cfg, args.config)
    out = args.output or Path(cfg["output_dir"]) / f"simulation_{cfg['selector']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    console.print(
        f"\n[bold]{'Converged' if results['converged'] else 'Not converged'}[/bold]: "
        f"final test accuracy {results['final_test_accuracy']:.3f} vs simulated "
        f"human agreement {results['human_agreement']:.3f} after "
        f"{len(results['rounds'])} rounds, {results['total_annotations']} annotations "
        f"-> {out}"
    )


if __name__ == "__main__":
    main()
