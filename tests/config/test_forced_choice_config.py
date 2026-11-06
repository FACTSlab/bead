"""Configuration contracts for forced-choice scoring architectures."""

import json

import pytest
import yaml
from didactic.api import ValidationError

from bead.config.active_learning import ForcedChoiceModelConfig
from gallery.eng.argument_structure.train_acceptability_model import build_model_config


def test_existing_configuration_defaults_to_joint_scoring():
    config = ForcedChoiceModelConfig()
    assert config.encoder_mode == "single_encoder"
    assert config.scoring_mode == "joint"


@pytest.mark.parametrize(
    "encoder_mode,scoring_mode",
    [
        ("single_encoder", "joint"),
        ("dual_encoder", "joint"),
        ("dual_encoder", "independent"),
    ],
)
def test_valid_architectures_round_trip_through_json_and_yaml(
    encoder_mode, scoring_mode
):
    config = ForcedChoiceModelConfig(
        encoder_mode=encoder_mode, scoring_mode=scoring_mode
    )
    json_config = ForcedChoiceModelConfig(**json.loads(config.model_dump_json()))
    yaml_config = ForcedChoiceModelConfig(
        **yaml.safe_load(yaml.safe_dump(config.model_dump()))
    )
    assert json_config == config
    assert yaml_config == config


def test_independent_scoring_requires_separate_encoding():
    with pytest.raises(
        ValidationError, match="independent scoring requires.*dual_encoder"
    ):
        ForcedChoiceModelConfig(scoring_mode="independent")


def test_config_updates_revalidate_architecture():
    config = ForcedChoiceModelConfig(
        encoder_mode="dual_encoder", scoring_mode="independent"
    )
    with pytest.raises(ValidationError, match="independent scoring requires"):
        config.with_(encoder_mode="single_encoder")


def test_unknown_scoring_mode_is_rejected():
    with pytest.raises(ValidationError):
        ForcedChoiceModelConfig(scoring_mode="bradley_terry")


def test_gallery_configuration_exposes_both_architecture_options():
    config = build_model_config(
        {"encoder_mode": "dual_encoder", "scoring_mode": "independent"}
    )
    assert config.encoder_mode == "dual_encoder"
    assert config.scoring_mode == "independent"
    assert config.mixed_effects.mode == "random_intercepts"
    default = build_model_config({})
    assert default.encoder_mode == "single_encoder"
    assert default.scoring_mode == "joint"
    with pytest.raises(ValidationError, match="independent scoring requires"):
        build_model_config({"scoring_mode": "independent"})
