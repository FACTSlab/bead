"""Tests for jsPsych TypeScript plugins."""

from pathlib import Path


def test_rating_plugin_exists() -> None:
    """Test that rating.ts plugin file exists."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/rating.ts")
    assert plugin_path.exists()


def test_rating_plugin_syntax() -> None:
    """Test that rating.ts plugin has valid TypeScript structure."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/rating.ts")
    content = plugin_path.read_text()

    # check for key plugin elements
    assert "BeadRatingPlugin" in content
    assert "bead-rating" in content
    assert "scale_min" in content
    assert "scale_max" in content
    assert "metadata" in content


def test_rating_plugin_preserves_metadata() -> None:
    """Test that rating plugin preserves metadata."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/rating.ts")
    content = plugin_path.read_text()

    # check that metadata is spread into trial_data
    assert "...trial.metadata" in content
    assert "trial_data" in content


def test_cloze_plugin_exists() -> None:
    """Test that cloze-dropdown.ts plugin file exists."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/cloze-dropdown.ts")
    assert plugin_path.exists()


def test_cloze_plugin_syntax() -> None:
    """Test that cloze-dropdown.ts plugin has valid TypeScript structure."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/cloze-dropdown.ts")
    content = plugin_path.read_text()

    # check for key plugin elements
    assert "BeadClozeMultiPlugin" in content
    assert "bead-cloze-multi" in content
    assert "unfilled_slots" in content
    assert "dropdown" in content
    assert "text" in content
    assert "metadata" in content


def test_cloze_plugin_preserves_metadata() -> None:
    """Test that cloze plugin preserves metadata."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/cloze-dropdown.ts")
    content = plugin_path.read_text()

    # check that metadata is spread into trial_data
    assert "...trial.metadata" in content
    assert "trial_data" in content


def test_forced_choice_plugin_exists() -> None:
    """Test that forced-choice.ts plugin file exists."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/forced-choice.ts")
    assert plugin_path.exists()


def test_forced_choice_plugin_syntax() -> None:
    """Test that forced-choice.ts plugin has valid TypeScript structure."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/forced-choice.ts")
    content = plugin_path.read_text()

    # check for key plugin elements
    assert "BeadForcedChoicePlugin" in content
    assert "bead-forced-choice" in content
    assert "alternatives" in content
    assert "randomize_position" in content
    assert "metadata" in content


def test_forced_choice_plugin_preserves_metadata() -> None:
    """Test that forced choice plugin preserves metadata."""
    plugin_path = Path("bead/deployment/jspsych/src/plugins/forced-choice.ts")
    content = plugin_path.read_text()

    # check that metadata is spread into trial_data
    assert "...trial.metadata" in content
    assert "trial_data" in content


def test_all_plugins_have_version() -> None:
    """Test that all plugins have version 0.2.0."""
    plugin_dir = Path("bead/deployment/jspsych/src/plugins")
    # exclude test files
    plugins = [p for p in plugin_dir.glob("*.ts") if not p.name.endswith(".test.ts")]

    assert len(plugins) == 10, f"Expected 10 plugins, found {len(plugins)}"

    for plugin_path in plugins:
        content = plugin_path.read_text()
        assert "0.2.0" in content, f"Plugin {plugin_path.name} missing version"


def test_all_plugins_have_author() -> None:
    """Test that all plugins have Bead Project author."""
    plugin_dir = Path("bead/deployment/jspsych/src/plugins")
    # exclude test files
    plugins = [p for p in plugin_dir.glob("*.ts") if not p.name.endswith(".test.ts")]

    for plugin_path in plugins:
        content = plugin_path.read_text()
        assert "Bead Project" in content, f"Plugin {plugin_path.name} missing author"


def test_compiled_plugins_exist() -> None:
    """Test that compiled JavaScript plugins exist in dist/."""
    dist_dir = Path("bead/deployment/jspsych/dist/plugins")
    assert dist_dir.exists(), "dist/plugins directory should exist after build"

    expected_plugins = [
        "rating.js",
        "forced-choice.js",
        "cloze-dropdown.js",
        "span-label.js",
    ]
    for plugin in expected_plugins:
        plugin_path = dist_dir / plugin
        assert plugin_path.exists(), f"Compiled plugin {plugin} should exist"


def test_slopit_source_exists() -> None:
    """Test that slopit TypeScript source file exists."""
    slopit_path = Path("bead/deployment/jspsych/src/slopit/index.ts")
    assert slopit_path.exists(), "slopit/index.ts should exist"


def test_slopit_source_exports() -> None:
    """Test that slopit source exports required modules."""
    slopit_path = Path("bead/deployment/jspsych/src/slopit/index.ts")
    content = slopit_path.read_text()

    assert "SlopitExtension" in content, "Should export SlopitExtension"
    assert "@slopit/adapter-jspsych" in content, "Should import from adapter"


def test_slopit_bundle_exists() -> None:
    """Test that compiled slopit bundle exists in dist/."""
    bundle_path = Path("bead/deployment/jspsych/dist/slopit-bundle.js")
    assert bundle_path.exists(), "slopit-bundle.js should exist after build"


def test_compiled_lib_files_exist() -> None:
    """Test that compiled library files exist in dist/lib/."""
    dist_dir = Path("bead/deployment/jspsych/dist/lib")
    assert dist_dir.exists(), "dist/lib directory should exist after build"

    expected_libs = ["list-distributor.js", "randomizer.js"]
    for lib in expected_libs:
        lib_path = dist_dir / lib
        assert lib_path.exists(), f"Compiled library {lib} should exist"
