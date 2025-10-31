"""Tests for default configurations."""

from __future__ import annotations

from pathlib import Path

import pytest

from bead.config.defaults import (
    DEFAULT_CONFIG,
    get_default_config,
    get_default_for_model,
)
from bead.config.models import (
    ActiveLearningConfig,
    DeploymentConfig,
    ItemConfig,
    ListConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    ResourceConfig,
    BeadConfig,
    TemplateConfig,
)


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG constant."""

    def test_default_config_is_valid_bead_config(self) -> None:
        """Test DEFAULT_CONFIG is a valid BeadConfig instance."""
        assert isinstance(DEFAULT_CONFIG, BeadConfig)

    def test_default_config_has_default_profile(self) -> None:
        """Test DEFAULT_CONFIG has 'default' profile."""
        assert DEFAULT_CONFIG.profile == "default"

    def test_default_config_has_all_nested_configs(self) -> None:
        """Test DEFAULT_CONFIG has all required nested configs."""
        assert isinstance(DEFAULT_CONFIG.paths, PathsConfig)
        assert isinstance(DEFAULT_CONFIG.resources, ResourceConfig)
        assert isinstance(DEFAULT_CONFIG.templates, TemplateConfig)
        assert isinstance(DEFAULT_CONFIG.items, ItemConfig)
        assert isinstance(DEFAULT_CONFIG.lists, ListConfig)
        assert isinstance(DEFAULT_CONFIG.deployment, DeploymentConfig)
        assert isinstance(DEFAULT_CONFIG.active_learning, ActiveLearningConfig)
        assert isinstance(DEFAULT_CONFIG.logging, LoggingConfig)

    def test_default_paths_are_relative(self) -> None:
        """Test default paths are relative."""
        assert not DEFAULT_CONFIG.paths.data_dir.is_absolute()
        assert not DEFAULT_CONFIG.paths.output_dir.is_absolute()
        assert not DEFAULT_CONFIG.paths.cache_dir.is_absolute()

    def test_default_batch_sizes_are_reasonable(self) -> None:
        """Test default batch sizes are reasonable."""
        assert DEFAULT_CONFIG.templates.batch_size == 1000
        assert DEFAULT_CONFIG.items.model.batch_size == 8
        assert DEFAULT_CONFIG.active_learning.forced_choice_model.batch_size == 16

    def test_default_logging_level_is_info(self) -> None:
        """Test default logging level is INFO."""
        assert DEFAULT_CONFIG.logging.level == "INFO"

    def test_default_filling_strategy_is_exhaustive(self) -> None:
        """Test default filling strategy is exhaustive."""
        assert DEFAULT_CONFIG.templates.filling_strategy == "exhaustive"

    def test_default_model_provider_is_huggingface(self) -> None:
        """Test default model provider is huggingface."""
        assert DEFAULT_CONFIG.items.model.provider == "huggingface"

    def test_default_device_is_cpu(self) -> None:
        """Test default device is CPU."""
        assert DEFAULT_CONFIG.items.model.device == "cpu"


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_bead_config(self) -> None:
        """Test get_default_config returns a BeadConfig instance."""
        config = get_default_config()
        assert isinstance(config, BeadConfig)

    def test_returns_copy(self) -> None:
        """Test get_default_config returns a copy, not the original."""
        config = get_default_config()
        assert config is not DEFAULT_CONFIG

    def test_returns_independent_copies(self) -> None:
        """Test modifications don't affect original DEFAULT_CONFIG."""
        config1 = get_default_config()
        config2 = get_default_config()

        # Modify config1
        config1.profile = "modified"
        config1.templates.batch_size = 9999

        # Check config2 and DEFAULT_CONFIG are unchanged
        assert config2.profile == "default"
        assert config2.templates.batch_size == 1000
        assert DEFAULT_CONFIG.profile == "default"
        assert DEFAULT_CONFIG.templates.batch_size == 1000

    def test_deep_copy_nested_configs(self) -> None:
        """Test nested configs are also copied."""
        config = get_default_config()

        # Modify nested config
        config.paths.data_dir = Path("/modified/path")

        # Check DEFAULT_CONFIG is unchanged
        assert DEFAULT_CONFIG.paths.data_dir == Path("data")

    def test_multiple_calls_return_different_instances(self) -> None:
        """Test multiple calls return different instances."""
        config1 = get_default_config()
        config2 = get_default_config()
        assert config1 is not config2


class TestGetDefaultForModel:
    """Tests for get_default_for_model function."""

    def test_returns_paths_config_default(self) -> None:
        """Test get_default_for_model returns PathsConfig default."""
        config = get_default_for_model(PathsConfig)
        assert isinstance(config, PathsConfig)
        assert config.data_dir == Path("data")

    def test_returns_resource_config_default(self) -> None:
        """Test get_default_for_model returns ResourceConfig default."""
        config = get_default_for_model(ResourceConfig)
        assert isinstance(config, ResourceConfig)
        assert config.cache_external is True

    def test_returns_template_config_default(self) -> None:
        """Test get_default_for_model returns TemplateConfig default."""
        config = get_default_for_model(TemplateConfig)
        assert isinstance(config, TemplateConfig)
        assert config.filling_strategy == "exhaustive"

    def test_returns_model_config_default(self) -> None:
        """Test get_default_for_model returns ModelConfig default."""
        config = get_default_for_model(ModelConfig)
        assert isinstance(config, ModelConfig)
        assert config.provider == "huggingface"

    def test_returns_item_config_default(self) -> None:
        """Test get_default_for_model returns ItemConfig default."""
        config = get_default_for_model(ItemConfig)
        assert isinstance(config, ItemConfig)
        assert config.apply_constraints is True

    def test_returns_list_config_default(self) -> None:
        """Test get_default_for_model returns ListConfig default."""
        config = get_default_for_model(ListConfig)
        assert isinstance(config, ListConfig)
        assert config.num_lists == 1

    def test_returns_deployment_config_default(self) -> None:
        """Test get_default_for_model returns DeploymentConfig default."""
        config = get_default_for_model(DeploymentConfig)
        assert isinstance(config, DeploymentConfig)
        assert config.platform == "jspsych"

    def test_returns_active_learning_config_default(self) -> None:
        """Test get_default_for_model returns ActiveLearningConfig default."""
        config = get_default_for_model(ActiveLearningConfig)
        assert isinstance(config, ActiveLearningConfig)
        assert config.trainer.epochs == 3

    def test_returns_logging_config_default(self) -> None:
        """Test get_default_for_model returns LoggingConfig default."""
        config = get_default_for_model(LoggingConfig)
        assert isinstance(config, LoggingConfig)
        assert config.level == "INFO"

    def test_returns_bead_config_default(self) -> None:
        """Test get_default_for_model returns BeadConfig default."""
        config = get_default_for_model(BeadConfig)
        assert isinstance(config, BeadConfig)
        assert config.profile == "default"

    def test_raises_type_error_for_non_model(self) -> None:
        """Test get_default_for_model raises TypeError for non-model types."""
        with pytest.raises(TypeError) as exc_info:
            get_default_for_model(str)  # type: ignore[arg-type]
        assert "must be a Pydantic BaseModel class" in str(exc_info.value)

    def test_raises_type_error_for_instance(self) -> None:
        """Test get_default_for_model raises TypeError for instances."""
        with pytest.raises(TypeError) as exc_info:
            get_default_for_model(PathsConfig())  # type: ignore[arg-type]
        assert "must be a Pydantic BaseModel class" in str(exc_info.value)
