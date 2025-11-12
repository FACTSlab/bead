"""Tests for configuration loading from YAML files."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bead.config.config import BeadConfig
from bead.config.loader import load_config, load_yaml_file, merge_configs


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_merge_flat_dicts(self) -> None:
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self) -> None:
        """Test deep merge of nested dictionaries."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"d": 4, "e": 5}}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": {"c": 2, "d": 4, "e": 5}}

    def test_merge_deeply_nested_dicts(self) -> None:
        """Test deep merge with multiple nesting levels."""
        base = {"a": {"b": {"c": 1}}}
        override = {"a": {"b": {"d": 2}}}
        result = merge_configs(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}

    def test_merge_override_replaces_non_dict(self) -> None:
        """Test that non-dict values are replaced entirely."""
        base = {"a": {"b": 1}}
        override = {"a": 2}
        result = merge_configs(base, override)
        assert result == {"a": 2}

    def test_merge_with_empty_override(self) -> None:
        """Test merge with empty override dictionary."""
        base = {"a": 1, "b": 2}
        override: dict[str, int] = {}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 2}

    def test_merge_with_empty_base(self) -> None:
        """Test merge with empty base dictionary."""
        base: dict[str, int] = {}
        override = {"a": 1, "b": 2}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 2}

    def test_merge_preserves_types(self) -> None:
        """Test that types are preserved during merge."""
        base = {"a": 1, "b": "string", "c": [1, 2, 3]}
        override = {"d": 2.5}
        result = merge_configs(base, override)
        assert result["a"] == 1
        assert result["b"] == "string"
        assert result["c"] == [1, 2, 3]
        assert result["d"] == 2.5

    def test_merge_with_none_values(self) -> None:
        """Test merge with None values in override."""
        base = {"a": 1, "b": 2}
        override = {"b": None, "c": 3}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": None, "c": 3}


class TestLoadYamlFile:
    """Tests for load_yaml_file function."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Test loading a valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("profile: test\nlogging:\n  level: DEBUG\n")
        result = load_yaml_file(config_file)
        assert result == {"profile": "test", "logging": {"level": "DEBUG"}}

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading an empty YAML file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        result = load_yaml_file(config_file)
        assert result == {}

    def test_load_complex_nested_yaml(self, tmp_path: Path) -> None:
        """Test loading complex nested YAML structure."""
        yaml_content = """
profile: dev
paths:
  data_dir: /data
  output_dir: /output
  cache_dir: /cache
logging:
  level: INFO
  console: true
  file: false
"""
        config_file = tmp_path / "complex.yaml"
        config_file.write_text(yaml_content)
        result = load_yaml_file(config_file)
        assert result["profile"] == "dev"
        assert result["paths"]["data_dir"] == "/data"
        assert result["logging"]["level"] == "INFO"
        assert result["logging"]["console"] is True

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading a file that doesn't exist."""
        config_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_yaml_file(config_file)

    def test_load_malformed_yaml(self, tmp_path: Path) -> None:
        """Test loading malformed YAML file."""
        config_file = tmp_path / "malformed.yaml"
        config_file.write_text("profile: test\n  invalid:\n    this is: [not valid")
        with pytest.raises(yaml.YAMLError, match="Failed to parse YAML"):
            load_yaml_file(config_file)

    def test_load_yaml_with_string_path(self, tmp_path: Path) -> None:
        """Test loading YAML with string path instead of Path object."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("profile: test\n")
        result = load_yaml_file(str(config_file))
        assert result == {"profile": "test"}

    def test_load_yaml_with_lists(self, tmp_path: Path) -> None:
        """Test loading YAML containing lists."""
        yaml_content = """
items:
  - name: item1
    value: 1
  - name: item2
    value: 2
"""
        config_file = tmp_path / "lists.yaml"
        config_file.write_text(yaml_content)
        result = load_yaml_file(config_file)
        assert len(result["items"]) == 2
        assert result["items"][0]["name"] == "item1"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_profile_default(self) -> None:
        """Test loading config from default profile."""
        config = load_config(profile="default")
        assert isinstance(config, BeadConfig)
        assert config.profile == "default"

    def test_load_config_from_profile_dev(self) -> None:
        """Test loading config from dev profile."""
        config = load_config(profile="dev")
        assert isinstance(config, BeadConfig)
        assert config.profile == "dev"

    def test_load_config_from_profile_prod(self) -> None:
        """Test loading config from prod profile."""
        config = load_config(profile="prod")
        assert isinstance(config, BeadConfig)
        assert config.profile == "prod"

    def test_load_config_from_profile_test(self) -> None:
        """Test loading config from test profile."""
        config = load_config(profile="test")
        assert isinstance(config, BeadConfig)
        assert config.profile == "test"

    def test_load_config_with_none_path(self) -> None:
        """Test loading config with None path uses profile defaults."""
        config = load_config(config_path=None, profile="dev")
        assert config.profile == "dev"
        assert config.logging.level == "DEBUG"

    def test_load_config_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading config from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
profile: test
logging:
  level: WARNING
"""
        )
        config = load_config(config_path=config_file)
        assert config.logging.level == "WARNING"

    def test_load_config_with_yaml_and_profile(self, tmp_path: Path) -> None:
        """Test loading config with both YAML file and profile."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("logging:\n  level: ERROR\n")
        config = load_config(config_path=config_file, profile="dev")
        # YAML should override profile
        assert config.logging.level == "ERROR"
        # But profile should still be set
        assert config.profile == "dev"

    def test_load_config_with_single_level_override(self) -> None:
        """Test loading config with single-level keyword override."""
        config = load_config(logging__console=False)
        assert config.logging.console is False

    def test_load_config_with_nested_override(self) -> None:
        """Test loading config with nested keyword override."""
        config = load_config(profile="default", logging__level="CRITICAL")
        assert config.logging.level == "CRITICAL"

    def test_load_config_with_deeply_nested_override(self) -> None:
        """Test loading config with deeply nested keyword override."""
        config = load_config(profile="default", paths__data_dir="/custom/data/path")
        assert str(config.paths.data_dir) == "/custom/data/path"

    def test_load_config_precedence_profile_yaml_override(self, tmp_path: Path) -> None:
        """Test configuration precedence: profile < yaml < override."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("logging:\n  level: WARNING\n")
        config = load_config(
            config_path=config_file, profile="dev", logging__level="CRITICAL"
        )
        # Override should win
        assert config.logging.level == "CRITICAL"

    def test_load_config_with_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading config with nonexistent file raises error."""
        config_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(config_path=config_file)

    def test_load_config_with_malformed_yaml(self, tmp_path: Path) -> None:
        """Test loading config with malformed YAML raises error."""
        config_file = tmp_path / "malformed.yaml"
        config_file.write_text("this is: [not: valid yaml")
        with pytest.raises(yaml.YAMLError):
            load_config(config_path=config_file)

    def test_load_config_with_invalid_values(self, tmp_path: Path) -> None:
        """Test loading config with invalid values raises ValidationError."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("logging:\n  level: INVALID_LEVEL\n")
        with pytest.raises(ValidationError):
            load_config(config_path=config_file)

    def test_load_config_multiple_overrides(self) -> None:
        """Test loading config with multiple keyword overrides."""
        config = load_config(
            profile="default",
            logging__level="DEBUG",
            logging__console=False,
            paths__data_dir="/data",
        )
        assert config.logging.level == "DEBUG"
        assert config.logging.console is False
        assert str(config.paths.data_dir) == "/data"

    def test_load_config_with_string_path(self, tmp_path: Path) -> None:
        """Test loading config with string path instead of Path object."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("profile: test\n")
        config = load_config(config_path=str(config_file))
        assert config.profile == "test"
