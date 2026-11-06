"""Behavioral tests for independent forced-choice scoring with a local encoder."""

from __future__ import annotations

import json
from uuid import uuid4

import numpy as np
import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.config.active_learning import ForcedChoiceModelConfig
from bead.items.item import Item
from bead.items.scoring import AcceptabilityScorer


@pytest.fixture(scope="module")
def tiny_encoder(tmp_path_factory):
    """Save an actual small transformer and tokenizer, requiring no downloads."""
    directory = tmp_path_factory.mktemp("forced-choice-encoder")
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "good",
        "bad",
        "other",
        "the",
        "cat",
        "sat",
        "on",
        "mat",
        ".",
    ]
    vocab_path = directory / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary) + "\n")
    # Transformers 5 renamed vocab_file to vocab; both accept the first argument.
    BertTokenizer(str(vocab_path)).save_pretrained(directory)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        BertModel(
            BertConfig(
                vocab_size=len(vocabulary),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=24,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                max_position_embeddings=64,
            )
        ).save_pretrained(directory)
    return str(directory)


@pytest.fixture(autouse=True)
def deterministic_torch():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(29)
        yield
    torch.set_num_threads(threads)


def item(*texts: str) -> Item:
    return Item(
        item_template_id=uuid4(),
        rendered_elements={f"option_{i}": text for i, text in enumerate(texts)},
    )


def make_model(tiny_encoder, mode="fixed", **overrides):
    values = {
        "model_name": tiny_encoder,
        "encoder_mode": "dual_encoder",
        "scoring_mode": "independent",
        "max_length": 32,
        "batch_size": 3,
        "num_epochs": 2,
        "learning_rate": 0.01,
        "device": "cpu",
        "mixed_effects": MixedEffectsConfig(
            mode=mode, prior_variance=0.0, regularization_strength=0.001
        ),
    }
    values.update(overrides)
    return ForcedChoiceModel(ForcedChoiceModelConfig(**values))


def initialize(model, n_options=2):
    model.option_names = [f"option_{i}" for i in range(n_options)]
    model.num_classes = n_options
    model._initialize_classifier(n_options)
    model._initialize_random_effects(n_options)
    model._is_fitted = True
    return model


def test_local_encoder_preserves_alternative_tokens(tiny_encoder):
    """The saved tokenizer keeps content words distinct after model loading."""
    model = make_model(tiny_encoder)
    tokens = model.tokenizer(["good", "bad", "other"], add_special_tokens=False)
    assert tokens["input_ids"] == [[5], [6], [7]]


@pytest.mark.parametrize("n_options", [2, 3, 5])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_probabilities_equal_softmax_of_independent_scores(
    tiny_encoder, monkeypatch, n_options, batch_size
):
    model = initialize(make_model(tiny_encoder, batch_size=batch_size), n_options)
    scores = {"good": 2.0, "bad": -1.0, "other": 0.5}

    def encode(texts):
        return torch.tensor([[scores[text]] + [0.0] * 15 for text in texts])

    monkeypatch.setattr(model, "_encode_single", encode)
    model.classifier_head = nn.Linear(16, 1, bias=False)
    with torch.no_grad():
        model.classifier_head.weight.zero_()
        model.classifier_head.weight[0, 0] = 1.0
    texts = ["good", "bad"] + ["other"] * (n_options - 2)
    items = [item(*texts), item(*reversed(texts)), item(*texts)]
    expected_logits = torch.tensor(
        [
            [scores[text] for text in example.rendered_elements.values()]
            for example in items
        ]
    )
    expected = expected_logits.softmax(dim=1).numpy()
    np.testing.assert_allclose(model.predict_proba(items), expected, atol=1e-7)
    if n_options == 2:
        assert expected[0, 0] == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item())
    predictions = model.predict(items)
    for row, prediction in zip(expected, predictions, strict=True):
        assert prediction.predicted_class == model.option_names[int(row.argmax())]
        assert prediction.confidence == pytest.approx(float(row.max()))
        np.testing.assert_allclose(list(prediction.probabilities.values()), row)


@pytest.mark.parametrize("mode", ["fixed", "random_slopes"])
def test_swap_equivariance_and_identical_option_ties(tiny_encoder, mode):
    model = initialize(make_model(tiny_encoder, mode))
    if mode == "random_slopes":
        model.random_effects.get_slopes("rater", model.classifier_head)
    items = [item("good", "bad"), item("bad", "good"), item("good", "good")]
    ids = None if mode == "fixed" else ["rater"] * len(items)
    proba = model.predict_proba(items, ids)
    np.testing.assert_allclose(proba[0], proba[1, ::-1], atol=1e-7)
    np.testing.assert_allclose(proba[2], [0.5, 0.5], atol=1e-7)
    np.testing.assert_array_equal(proba, model.predict_proba(items, ids))


def test_scores_and_pairwise_odds_do_not_depend_on_other_options(tiny_encoder):
    model = initialize(make_model(tiny_encoder), 3)
    items = [
        item("good", "bad", "other"),
        item("good", "bad", "the cat sat on the mat ."),
    ]
    model.encoder.eval()
    model.classifier_head.eval()
    with torch.no_grad():
        embeddings = model._prepare_inputs(items)
        logits = model._compute_logits(embeddings, ["p", "p"])
    torch.testing.assert_close(logits[0, :2], logits[1, :2], atol=1e-7, rtol=0)
    proba = model.predict_proba(items)
    assert proba[0, 0] / proba[0, 1] == pytest.approx(
        proba[1, 0] / proba[1, 1], abs=1e-7
    )


def test_each_alternative_receives_the_choice_loss_gradient(tiny_encoder):
    model = initialize(make_model(tiny_encoder))
    model.classifier_head = nn.Linear(16, 1, bias=False)
    with torch.no_grad():
        model.classifier_head.weight.fill_(0.1)
    embeddings = torch.arange(32, dtype=torch.float).reshape(1, 2, 16).requires_grad_()
    logits = model._compute_logits(embeddings, ["p"])
    loss = nn.functional.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    expected = (logits.detach().softmax(1) - torch.tensor([[1.0, 0.0]])).unsqueeze(
        -1
    ) * 0.1
    torch.testing.assert_close(embeddings.grad, expected.expand_as(embeddings))
    assert model.classifier_head.weight.grad.abs().sum() > 0


def test_position_intercepts_change_log_odds_and_unknown_raters_use_shared_scores(
    tiny_encoder,
):
    model = initialize(make_model(tiny_encoder, "random_intercepts"))
    bias = model.random_effects.get_intercepts("rater", 2, "mu")
    with torch.no_grad():
        bias.copy_(torch.tensor([1.5, -0.5]))
    items = [item("good", "good")]
    known = model.predict_proba(items, ["rater"])[0]
    unknown = model.predict_proba(items, ["new-rater"])[0]
    assert np.log(known[0] / known[1]) == pytest.approx(2.0)
    np.testing.assert_allclose(unknown, [0.5, 0.5], atol=1e-7)
    assert "new-rater" not in model.random_effects.intercepts["mu"]


@pytest.mark.parametrize("mode", ["fixed", "random_intercepts", "random_slopes"])
@pytest.mark.parametrize("n_options", [2, 3])
def test_training_updates_encoder_head_and_participant_parameters(
    tiny_encoder, monkeypatch, mode, n_options
):
    model = make_model(tiny_encoder, mode)
    texts = ["good", "bad", "other"][:n_options]
    items = [item(*texts) for _ in range(5)]
    # options need not all occur as selected labels
    labels = ["option_0"] * len(items)
    ids = None if mode == "fixed" else ["rater"] * len(items)
    initial_encoder = {
        name: value.clone() for name, value in model.encoder.state_dict().items()
    }
    initial_heads = {}
    initial_biases = {}
    initialize_classifier = model._initialize_classifier
    get_slopes = RandomEffectsManager.get_slopes
    get_intercepts = RandomEffectsManager.get_intercepts

    def capture_classifier(n):
        initialize_classifier(n)
        initial_heads["fixed"] = {
            name: value.clone()
            for name, value in model.classifier_head.state_dict().items()
        }

    def capture_slopes(manager, participant_id, fixed_head, create_if_missing=True):
        head = get_slopes(manager, participant_id, fixed_head, create_if_missing)
        if create_if_missing and participant_id not in initial_heads:
            initial_heads[participant_id] = {
                name: value.clone() for name, value in head.state_dict().items()
            }
        return head

    def capture_intercepts(
        manager, participant_id, n_classes, param_name, create_if_missing=True
    ):
        bias = get_intercepts(
            manager, participant_id, n_classes, param_name, create_if_missing
        )
        if create_if_missing and participant_id not in initial_biases:
            initial_biases[participant_id] = bias.detach().clone()
        return bias

    monkeypatch.setattr(model, "_initialize_classifier", capture_classifier)
    monkeypatch.setattr(RandomEffectsManager, "get_slopes", capture_slopes)
    monkeypatch.setattr(RandomEffectsManager, "get_intercepts", capture_intercepts)
    metrics = model.train(items, labels, ids)
    assert np.isfinite(metrics["train_loss"])
    assert model.num_classes == n_options
    assert any(
        not torch.equal(value, initial_encoder[name])
        for name, value in model.encoder.state_dict().items()
    )
    head = (
        model.random_effects.slopes["rater"]
        if mode == "random_slopes"
        else model.classifier_head
    )
    before = initial_heads["rater" if mode == "random_slopes" else "fixed"]
    assert any(
        not torch.equal(value, before[name])
        for name, value in head.state_dict().items()
    )
    if mode == "random_intercepts":
        assert not torch.equal(
            model.random_effects.intercepts["mu"]["rater"], initial_biases["rater"]
        )
    probabilities = model.predict_proba(items, ids)
    assert probabilities.shape == (5, n_options)
    np.testing.assert_allclose(probabilities.sum(1), 1.0, atol=1e-7)


def test_fixed_model_learns_content_preferences(tiny_encoder, monkeypatch):
    model = make_model(tiny_encoder, num_epochs=20, learning_rate=0.02, batch_size=4)
    items = [item("good", "bad"), item("bad", "good")] * 4
    labels = ["option_0", "option_1"] * 4
    initialize_classifier = model._initialize_classifier

    def no_head_dropout(n):
        initialize_classifier(n)
        model.classifier_head[2].p = 0.0

    monkeypatch.setattr(model, "_initialize_classifier", no_head_dropout)
    metrics = model.train(items, labels)
    assert metrics["train_accuracy"] == 1.0
    proba = model.predict_proba(items)
    assert np.mean(proba[np.arange(8), [0, 1] * 4]) > 0.9


@pytest.mark.parametrize("mode", ["fixed", "random_intercepts", "random_slopes"])
def test_checkpoint_round_trip_preserves_scoring_and_known_and_unknown_predictions(
    tiny_encoder, tmp_path, mode
):
    model = make_model(tiny_encoder, mode)
    items = [item("good", "bad"), item("bad", "good")]
    ids = None if mode == "fixed" else ["rater", "rater"]
    model.train(items, ["option_0", "option_1"], ids)
    checkpoint = tmp_path / "model"
    model.save(str(checkpoint))
    restored = make_model(
        tiny_encoder, scoring_mode="joint", encoder_mode="single_encoder"
    )
    restored.load(str(checkpoint))
    assert restored.config.scoring_mode == "independent"
    for participants in (ids, None if mode == "fixed" else ["unknown"] * 2):
        np.testing.assert_allclose(
            restored.predict_proba(items, participants),
            model.predict_proba(items, participants),
            atol=1e-7,
        )
    assert AcceptabilityScorer(restored).score(items[0]) == pytest.approx(
        AcceptabilityScorer(model).score(items[0])
    )


def test_legacy_checkpoint_without_scoring_mode_loads_as_joint(tiny_encoder, tmp_path):
    model = initialize(
        make_model(tiny_encoder, scoring_mode="joint", encoder_mode="single_encoder")
    )
    items = [item("good", "bad")]
    model.save(str(tmp_path))
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    del config["scoring_mode"]
    config_path.write_text(json.dumps(config))
    restored = make_model(tiny_encoder)
    restored.load(str(tmp_path))
    assert restored.config.scoring_mode == "joint"
    np.testing.assert_allclose(
        restored.predict_proba(items), model.predict_proba(items), atol=1e-7
    )


@pytest.mark.parametrize(
    "bad_items", [[item("good")], [item("good", "bad"), item("good", "bad", "other")]]
)
def test_training_rejects_missing_or_inconsistent_options(tiny_encoder, bad_items):
    model = make_model(tiny_encoder)
    with pytest.raises(ValueError, match="at least two options|identical option names"):
        model.train(bad_items, ["option_0"] * len(bad_items))


def test_validation_and_prediction_reject_inconsistent_options(tiny_encoder):
    model = make_model(tiny_encoder)
    items = [item("good", "bad")]
    wrong = [item("good", "bad", "other")]
    with pytest.raises(ValueError, match="identical option names"):
        model.train(
            items, ["option_0"], validation_items=wrong, validation_labels=["option_0"]
        )
    model.train(items, ["option_0"])
    with pytest.raises(ValueError, match="exactly the fitted option names"):
        model.predict_proba(wrong)


def test_dual_joint_training_still_produces_option_logits(tiny_encoder):
    model = make_model(tiny_encoder, scoring_mode="joint")
    items = [item("good", "bad"), item("bad", "good")]
    metrics = model.train(items, ["option_0", "option_1"])
    assert np.isfinite(metrics["train_loss"])
    assert model.classifier_head[0].in_features == 32
    assert model.classifier_head[-1].out_features == 2
    assert model.predict_proba(items).shape == (2, 2)


@pytest.mark.parametrize("mode", ["fixed", "random_intercepts", "random_slopes"])
def test_early_stopping_restores_best_encoder_head_and_participant_effects(
    tiny_encoder, monkeypatch, mode
):
    model = make_model(tiny_encoder, mode, num_epochs=8, early_stopping_patience=2)
    train = [item("good", "bad"), item("bad", "good")]
    dev = [item("good", "other")]
    ids = None if mode == "fixed" else ["rater"] * 2
    predict_proba = model._do_predict_proba
    snapshots = []

    def score_with_controlled_dev_loss(items, participant_ids):
        actual = predict_proba(items, participant_ids)
        if items is dev:
            snapshots.append(
                {
                    "encoder": {
                        name: value.clone()
                        for name, value in model.encoder.state_dict().items()
                    },
                    "head": {
                        name: value.clone()
                        for name, value in model.classifier_head.state_dict().items()
                    },
                    "participant": predict_proba(train, ["rater"] * 2),
                }
            )
            return np.array(
                [
                    [
                        0.9 if len(snapshots) == 1 else 0.4,
                        0.1 if len(snapshots) == 1 else 0.6,
                    ]
                ]
            )
        return actual

    monkeypatch.setattr(model, "_do_predict_proba", score_with_controlled_dev_loss)
    metrics = model.train(
        train,
        ["option_0", "option_1"],
        ids,
        validation_items=dev,
        validation_labels=["option_0"],
    )
    # three training epochs, followed by final validation metrics after restoration
    assert len(snapshots) == 4
    assert "val_accuracy" in metrics
    for component in ("encoder", "head"):
        for name in snapshots[0][component]:
            torch.testing.assert_close(
                snapshots[-1][component][name],
                snapshots[0][component][name],
                rtol=0,
                atol=0,
            )
    np.testing.assert_array_equal(
        snapshots[-1]["participant"], snapshots[0]["participant"]
    )


def test_common_score_offset_leaves_probabilities_unchanged(tiny_encoder):
    model = initialize(make_model(tiny_encoder), 3)
    items = [item("good", "bad", "other")]
    before = model.predict_proba(items)
    with torch.no_grad():
        model.classifier_head[-1].bias.add_(10.0)
    np.testing.assert_allclose(model.predict_proba(items), before, atol=1e-6)


def test_empty_training_is_rejected_and_empty_prediction_keeps_option_dimension(
    tiny_encoder,
):
    model = make_model(tiny_encoder)
    with pytest.raises(ValueError, match="nonempty training items"):
        model.train([], [])
    initialize(model)
    assert model.predict_proba([]).shape == (0, 2)
    assert model.predict([]) == []


def test_unknown_training_response_is_rejected(tiny_encoder):
    model = make_model(tiny_encoder)
    with pytest.raises(ValueError, match="Invalid label"):
        model.train([item("good", "bad")], ["option_2"])


def test_option_order_is_defined_by_names_not_mapping_insertion_order(tiny_encoder):
    model = initialize(make_model(tiny_encoder))
    original = item("good", "bad")
    reordered = original.with_(
        rendered_elements=dict(reversed(list(original.rendered_elements.items())))
    )
    np.testing.assert_array_equal(
        model.predict_proba([original]), model.predict_proba([reordered])
    )


@pytest.mark.parametrize("mode", ["random_intercepts", "random_slopes"])
def test_participant_parameters_are_distinct_and_unknown_predictions_do_not_create_them(
    tiny_encoder, mode
):
    model = make_model(tiny_encoder, mode, num_epochs=3)
    items = [item("good", "bad")] * 4
    model.train(
        items,
        ["option_0", "option_0", "option_1", "option_1"],
        ["alice", "alice", "bob", "bob"],
    )
    parameters = (
        model.random_effects.intercepts["mu"]
        if mode == "random_intercepts"
        else model.random_effects.slopes
    )
    assert set(parameters) == {"alice", "bob"}
    if mode == "random_intercepts":
        assert parameters["alice"].data_ptr() != parameters["bob"].data_ptr()
        assert not torch.equal(parameters["alice"], parameters["bob"])
    else:
        for alice, bob in zip(
            parameters["alice"].parameters(),
            parameters["bob"].parameters(),
            strict=True,
        ):
            assert alice.data_ptr() != bob.data_ptr()
    predictions = model.predict_proba([items[0]], ["new-rater"])
    assert set(parameters) == {"alice", "bob"}
    np.testing.assert_array_equal(
        predictions, model.predict_proba([items[0]], ["new-rater"])
    )


def test_original_single_encoder_joint_training_remains_available(tiny_encoder):
    model = make_model(
        tiny_encoder, scoring_mode="joint", encoder_mode="single_encoder", num_epochs=1
    )
    items = [item("good", "bad"), item("bad", "good")]
    metrics = model.train(items, ["option_0", "option_1"])
    assert np.isfinite(metrics["train_loss"])
    assert model.classifier_head[0].in_features == 16
    assert model.classifier_head[-1].out_features == 2
    assert model.predict_proba(items).shape == (2, 2)
