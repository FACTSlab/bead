"""Tests for jsPsych JavaScript plugins."""

from pathlib import Path


def test_rating_plugin_exists() -> None:
    """Test that rating.js plugin file exists."""
    plugin_path = Path("sash/deployment/jspsych/plugins/rating.js")
    assert plugin_path.exists()


def test_rating_plugin_syntax() -> None:
    """Test that rating.js plugin has valid JavaScript structure."""
    plugin_path = Path("sash/deployment/jspsych/plugins/rating.js")
    content = plugin_path.read_text()

    # Check for key plugin elements
    assert "jsPsychSashRating" in content
    assert "function (jspsych)" in content
    assert "sash-rating" in content
    assert "scale_min" in content
    assert "scale_max" in content
    assert "metadata" in content


def test_rating_plugin_preserves_metadata() -> None:
    """Test that rating plugin preserves metadata."""
    plugin_path = Path("sash/deployment/jspsych/plugins/rating.js")
    content = plugin_path.read_text()

    # Check that metadata is spread into trial_data
    assert "...trial.metadata" in content
    assert "trial_data" in content


def test_cloze_plugin_exists() -> None:
    """Test that cloze-dropdown.js plugin file exists."""
    plugin_path = Path("sash/deployment/jspsych/plugins/cloze-dropdown.js")
    assert plugin_path.exists()


def test_cloze_plugin_syntax() -> None:
    """Test that cloze-dropdown.js plugin has valid JavaScript structure."""
    plugin_path = Path("sash/deployment/jspsych/plugins/cloze-dropdown.js")
    content = plugin_path.read_text()

    # Check for key plugin elements
    assert "jsPsychSashClozeMulti" in content
    assert "sash-cloze-multi" in content
    assert "unfilled_slots" in content
    assert "dropdown" in content
    assert "text" in content
    assert "metadata" in content


def test_cloze_plugin_preserves_metadata() -> None:
    """Test that cloze plugin preserves metadata."""
    plugin_path = Path("sash/deployment/jspsych/plugins/cloze-dropdown.js")
    content = plugin_path.read_text()

    # Check that metadata is spread into trial_data
    assert "...trial.metadata" in content
    assert "trial_data" in content


def test_forced_choice_plugin_exists() -> None:
    """Test that forced-choice.js plugin file exists."""
    plugin_path = Path("sash/deployment/jspsych/plugins/forced-choice.js")
    assert plugin_path.exists()


def test_forced_choice_plugin_syntax() -> None:
    """Test that forced-choice.js plugin has valid JavaScript structure."""
    plugin_path = Path("sash/deployment/jspsych/plugins/forced-choice.js")
    content = plugin_path.read_text()

    # Check for key plugin elements
    assert "jsPsychSashForcedChoice" in content
    assert "sash-forced-choice" in content
    assert "alternatives" in content
    assert "randomize_position" in content
    assert "metadata" in content


def test_forced_choice_plugin_preserves_metadata() -> None:
    """Test that forced choice plugin preserves metadata."""
    plugin_path = Path("sash/deployment/jspsych/plugins/forced-choice.js")
    content = plugin_path.read_text()

    # Check that metadata is spread into trial_data
    assert "...trial.metadata" in content
    assert "trial_data" in content


def test_all_plugins_have_version() -> None:
    """Test that all plugins have version 0.1.0."""
    plugin_dir = Path("sash/deployment/jspsych/plugins")
    plugins = list(plugin_dir.glob("*.js"))

    assert len(plugins) == 3, "Expected 3 plugins"

    for plugin_path in plugins:
        content = plugin_path.read_text()
        assert "0.1.0" in content, f"Plugin {plugin_path.name} missing version"


def test_all_plugins_have_author() -> None:
    """Test that all plugins have SASH Project author."""
    plugin_dir = Path("sash/deployment/jspsych/plugins")
    plugins = list(plugin_dir.glob("*.js"))

    for plugin_path in plugins:
        content = plugin_path.read_text()
        assert "SASH Project" in content, f"Plugin {plugin_path.name} missing author"
