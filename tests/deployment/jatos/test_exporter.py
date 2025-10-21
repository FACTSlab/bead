"""Tests for JATOS exporter."""

import json
import zipfile
from pathlib import Path

import pytest

from sash.deployment.jatos.exporter import JATOSExporter


def test_jatos_exporter_creation() -> None:
    """Test JATOS exporter can be created."""
    exporter = JATOSExporter(
        study_title="Test Study", study_description="Test Description"
    )
    assert exporter.study_title == "Test Study"
    assert exporter.study_description == "Test Description"


def test_jatos_exporter_export(
    sample_experiment_dir: Path, jzip_output_path: Path
) -> None:
    """Test JATOS exporter creates valid .jzip."""
    exporter = JATOSExporter(
        study_title="Test Study", study_description="Test Description"
    )

    exporter.export(sample_experiment_dir, jzip_output_path)

    # Verify .jzip file was created
    assert jzip_output_path.exists()
    assert jzip_output_path.suffix == ".jzip"


def test_jzip_structure(
    sample_experiment_dir: Path, jzip_output_path: Path
) -> None:
    """Test .jzip has correct structure."""
    exporter = JATOSExporter(
        study_title="Test Study", study_description="Test Description"
    )
    exporter.export(sample_experiment_dir, jzip_output_path)

    with zipfile.ZipFile(jzip_output_path, "r") as zipf:
        namelist = zipf.namelist()

        # Check for study.json
        assert "study.json" in namelist

        # Check for experiment files
        assert "experiment/index.html" in namelist
        assert "experiment/css/experiment.css" in namelist
        assert "experiment/js/experiment.js" in namelist
        assert "experiment/data/config.json" in namelist
        assert "experiment/data/timeline.json" in namelist


def test_study_json_schema(
    sample_experiment_dir: Path, jzip_output_path: Path
) -> None:
    """Test study.json follows JATOS v3 schema."""
    exporter = JATOSExporter(
        study_title="Test Study", study_description="Test Description"
    )
    exporter.export(sample_experiment_dir, jzip_output_path)

    with zipfile.ZipFile(jzip_output_path, "r") as zipf:
        study_json_content = zipf.read("study.json")
        study_json = json.loads(study_json_content)

        # Check top-level structure
        assert "version" in study_json
        assert study_json["version"] == "3"
        assert "data" in study_json

        # Check data structure
        data = study_json["data"]
        assert "uuid" in data
        assert "title" in data
        assert "description" in data
        assert "dirName" in data
        assert "componentList" in data
        assert "batchList" in data

        # Check title and description
        assert data["title"] == "Test Study"
        assert data["description"] == "Test Description"

        # Check UUID format (should be valid UUID)
        import uuid

        uuid.UUID(data["uuid"])  # Will raise if invalid


def test_study_json_component(
    sample_experiment_dir: Path, jzip_output_path: Path
) -> None:
    """Test study.json has correct component configuration."""
    exporter = JATOSExporter(
        study_title="Test Study",
        study_description="Test Description",
    )
    exporter.export(
        sample_experiment_dir,
        jzip_output_path,
        component_title="Main Experiment",
    )

    with zipfile.ZipFile(jzip_output_path, "r") as zipf:
        study_json_content = zipf.read("study.json")
        study_json = json.loads(study_json_content)

        components = study_json["data"]["componentList"]
        assert len(components) == 1

        component = components[0]
        assert "uuid" in component
        assert "title" in component
        assert "htmlFilePath" in component
        assert "active" in component

        assert component["title"] == "Main Experiment"
        assert component["htmlFilePath"] == "experiment/index.html"
        assert component["active"] is True


def test_sanitize_dirname(
    sample_experiment_dir: Path, jzip_output_path: Path
) -> None:
    """Test directory name sanitization."""
    exporter = JATOSExporter(
        study_title="My Study (2024)",
        study_description="",
    )
    exporter.export(sample_experiment_dir, jzip_output_path)

    with zipfile.ZipFile(jzip_output_path, "r") as zipf:
        study_json_content = zipf.read("study.json")
        study_json = json.loads(study_json_content)

        dirname = study_json["data"]["dirName"]
        assert dirname == "my_study_2024"
        # Check no invalid characters
        assert all(c.isalnum() or c == "_" for c in dirname)


def test_export_nonexistent_directory(jzip_output_path: Path) -> None:
    """Test export raises error for nonexistent directory."""
    exporter = JATOSExporter(study_title="Test", study_description="")

    nonexistent_dir = Path("/nonexistent/path")

    with pytest.raises(ValueError, match="does not exist"):
        exporter.export(nonexistent_dir, jzip_output_path)


def test_export_file_instead_of_directory(
    tmp_path: Path, jzip_output_path: Path
) -> None:
    """Test export raises error when given a file instead of directory."""
    exporter = JATOSExporter(study_title="Test", study_description="")

    file_path = tmp_path / "file.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError, match="not a directory"):
        exporter.export(file_path, jzip_output_path)


def test_export_missing_index_html(
    tmp_path: Path, jzip_output_path: Path
) -> None:
    """Test export raises error when index.html is missing."""
    exporter = JATOSExporter(study_title="Test", study_description="")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="index.html"):
        exporter.export(empty_dir, jzip_output_path)


def test_experiment_files_preserved(
    sample_experiment_dir: Path, jzip_output_path: Path
) -> None:
    """Test that all experiment files are preserved in .jzip."""
    exporter = JATOSExporter(study_title="Test Study", study_description="")
    exporter.export(sample_experiment_dir, jzip_output_path)

    with zipfile.ZipFile(jzip_output_path, "r") as zipf:
        # Read HTML file
        html_content = zipf.read("experiment/index.html").decode("utf-8")
        assert "<title>Test Experiment</title>" in html_content

        # Read CSS file
        css_content = zipf.read("experiment/css/experiment.css").decode("utf-8")
        assert "body { margin: 0; }" in css_content

        # Read JS file
        js_content = zipf.read("experiment/js/experiment.js").decode("utf-8")
        assert "console.log('test');" in js_content

        # Read config.json
        config_content = zipf.read("experiment/data/config.json").decode("utf-8")
        config_json = json.loads(config_content)
        assert config_json["title"] == "Test"


def test_custom_component_title(
    sample_experiment_dir: Path, jzip_output_path: Path
) -> None:
    """Test custom component title."""
    exporter = JATOSExporter(study_title="Test Study", study_description="")
    exporter.export(
        sample_experiment_dir,
        jzip_output_path,
        component_title="Custom Component Name",
    )

    with zipfile.ZipFile(jzip_output_path, "r") as zipf:
        study_json_content = zipf.read("study.json")
        study_json = json.loads(study_json_content)

        component = study_json["data"]["componentList"][0]
        assert component["title"] == "Custom Component Name"


def test_zip_file_compression(
    sample_experiment_dir: Path, jzip_output_path: Path
) -> None:
    """Test that .jzip file is compressed."""
    exporter = JATOSExporter(study_title="Test Study", study_description="")
    exporter.export(sample_experiment_dir, jzip_output_path)

    # Check that it's a valid ZIP file with compression
    with zipfile.ZipFile(jzip_output_path, "r") as zipf:
        # Get info about a file
        info = zipf.getinfo("study.json")
        # ZIP_DEFLATED = 8
        assert info.compress_type == zipfile.ZIP_DEFLATED
