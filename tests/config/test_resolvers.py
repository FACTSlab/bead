"""Tests for bead's interpolation resolver and its wiring into the engine."""

from __future__ import annotations

from pathlib import Path

import pytest
from didactic import settings
from didactic.settings import (
    ConfigError,
    InterpolationError,
    list_resolvers,
    resolve,
)

import bead.config
from bead.config import PROFILES, load_config
from bead.config.config import BeadConfig


def test_bead_path_is_registered_with_the_engine() -> None:
    assert "bead.path" in list_resolvers()


def test_bead_path_joins_onto_data_dir(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "paths:\n  data_dir: /d\n  output_dir: ${bead.path:sub}\n",
    )
    config = load_config(config_path=config_file)
    assert config.paths.data_dir == Path("/d")
    assert config.paths.output_dir == Path("/d/sub")


def test_bead_path_reads_an_interpolated_data_dir(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "paths:\n  data_dir: ${oc.env:BEAD_TEST_ROOT,/root}\n"
        "  cache_dir: ${bead.path:cache}\n",
    )
    config = load_config(config_path=config_file)
    assert config.paths.cache_dir == Path("/root/cache")


def test_bead_path_cycle_is_reported(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("paths:\n  data_dir: ${bead.path:x}\n")
    with pytest.raises(InterpolationError, match="cycle"):
        load_config(config_path=config_file)


def test_bead_path_requires_an_argument() -> None:
    with pytest.raises(InterpolationError, match="requires a relative path"):
        resolve("${bead.path:}", root={"paths": {"data_dir": "/d"}})


def test_bead_path_requires_a_string_data_dir() -> None:
    with pytest.raises(InterpolationError, match="expected str"):
        resolve("${bead.path:x}", root={"paths": {"data_dir": 3}})


def test_bead_path_without_a_data_dir_is_unresolved() -> None:
    with pytest.raises(InterpolationError, match="paths.data_dir unresolved"):
        resolve("${bead.path:x}", root={"paths": {}})


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_profile_composes(name: str) -> None:
    config = load_config(profile=name)
    assert isinstance(config, BeadConfig)
    assert config.profile == name


def test_keyword_overrides_nest_and_keep_their_type() -> None:
    config = load_config(
        lists__num_lists=7,
        resources__cache_external=False,
        logging__level="WARNING",
    )
    assert config.lists.num_lists == 7
    assert config.resources.cache_external is False
    assert config.logging.level == "WARNING"


def test_keyword_override_wrong_type_is_refused() -> None:
    with pytest.raises(ConfigError, match=r"lists\.num_lists"):
        load_config(lists__num_lists="seven")


def test_string_overrides_are_read_by_the_field_type() -> None:
    config = load_config(
        overrides=[
            "lists.num_lists=3",
            "resources.cache_external=yes",
            "lists.balance_by=a,b",
            "templates.mlm_custom_order=[1, 2]",
        ]
    )
    assert config.lists.num_lists == 3
    assert config.resources.cache_external is True
    assert config.lists.balance_by == ("a", "b")
    assert config.templates.mlm_custom_order == (1, 2)


def test_model_map_entries_merge_key_by_key(tmp_path: Path) -> None:
    primary = tmp_path / "config.yaml"
    primary.write_text(
        "templates:\n  slot_strategies:\n    subj: {strategy: exhaustive}\n"
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("templates:\n  slot_strategies:\n    obj: {strategy: random}\n")
    config = load_config(config_path=primary, extra=[overlay])
    assert config.templates.slot_strategies is not None
    assert sorted(config.templates.slot_strategies) == ["obj", "subj"]
    assert config.templates.slot_strategies["subj"].strategy == "exhaustive"
    assert config.templates.slot_strategies["obj"].strategy == "random"
    config = load_config(
        config_path=primary,
        overrides=['templates.slot_strategies.obj={"strategy": "random"}'],
    )
    assert config.templates.slot_strategies is not None
    assert config.templates.slot_strategies["obj"].strategy == "random"


def test_unknown_override_key_is_refused_at_merge_time() -> None:
    with pytest.raises(ConfigError, match="Unknown config key 'lists.no_such'"):
        load_config(overrides=["lists.no_such=1"])


def test_unknown_overlay_key_names_the_file(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("lists:\n  no_such: 1\n")
    with pytest.raises(ConfigError, match="no_such"):
        load_config(extra=[overlay])


def test_config_package_exports_the_engine_names() -> None:
    assert bead.config.compose is settings.compose
    assert bead.config.ConfigError is settings.ConfigError
    assert bead.config.InterpolationError is settings.InterpolationError
    assert bead.config.ConfigValue is settings.ConfigValue
    assert bead.config.register_resolver is settings.register_resolver
    for name in (
        "compose",
        "ConfigError",
        "InterpolationError",
        "ConfigValue",
        "register_resolver",
    ):
        assert name in bead.config.__all__
    assert "ComposeValue" not in bead.config.__all__
