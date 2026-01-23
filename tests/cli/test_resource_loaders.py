"""Tests for resource loader CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bead.cli.resource_loaders import (
    import_framenet,
    import_propbank,
    import_unimorph,
    import_verbnet,
)
from bead.resources.lexical_item import LexicalItem


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_verbnet_items() -> list[LexicalItem]:
    """Create mock VerbNet items."""
    return [
        LexicalItem(
            lemma="break",
            language_code="eng",
            features={"pos": "VERB", "verb_class": "break-45.1"},
        ),
        LexicalItem(
            lemma="shatter",
            language_code="eng",
            features={"pos": "VERB", "verb_class": "break-45.1"},
        ),
    ]


@pytest.fixture
def mock_unimorph_items() -> list[LexicalItem]:
    """Create mock UniMorph items."""
    return [
        LexicalItem(
            lemma="walk",
            form="walked",
            language_code="eng",
            features={"pos": "VERB", "tense": "PST"},
        ),
        LexicalItem(
            lemma="walk",
            form="walking",
            language_code="eng",
            features={"pos": "VERB", "aspect": "PROG"},
        ),
    ]


class TestImportVerbNet:
    """Test import-verbnet command."""

    def test_import_verbnet_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_verbnet_items: list[LexicalItem],
    ) -> None:
        """Test basic VerbNet import."""
        output_file = tmp_path / "verbnet_verbs.jsonl"

        with patch("bead.cli.resource_loaders.GlazingAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = mock_verbnet_items
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_verbnet,
                [
                    "--output",
                    str(output_file),
                    "--query",
                    "break",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Imported 2 verbs from VerbNet" in result.output

    def test_import_verbnet_with_limit(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_verbnet_items: list[LexicalItem],
    ) -> None:
        """Test VerbNet import with limit."""
        output_file = tmp_path / "verbnet_verbs.jsonl"

        with patch("bead.cli.resource_loaders.GlazingAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = mock_verbnet_items
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_verbnet,
                [
                    "--output",
                    str(output_file),
                    "--limit",
                    "1",
                ],
            )

            assert result.exit_code == 0
            assert "Limiting results to 1 items" in result.output
            assert "Imported 1 verbs from VerbNet" in result.output

    def test_import_verbnet_with_frames(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_verbnet_items: list[LexicalItem],
    ) -> None:
        """Test VerbNet import with frames."""
        output_file = tmp_path / "verbnet_verbs.jsonl"

        with patch("bead.cli.resource_loaders.GlazingAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = mock_verbnet_items
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_verbnet,
                [
                    "--output",
                    str(output_file),
                    "--include-frames",
                    "--verb-class",
                    "break-45.1",
                ],
            )

            assert result.exit_code == 0
            mock_adapter.fetch_items.assert_called_once()
            call_kwargs = mock_adapter.fetch_items.call_args[1]
            assert call_kwargs["include_frames"] is True
            assert call_kwargs["verb_class"] == "break-45.1"


class TestImportUniMorph:
    """Test import-unimorph command."""

    def test_import_unimorph_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_unimorph_items: list[LexicalItem],
    ) -> None:
        """Test basic UniMorph import."""
        output_file = tmp_path / "unimorph_forms.jsonl"

        with patch("bead.cli.resource_loaders.UniMorphAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = mock_unimorph_items
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_unimorph,
                [
                    "--output",
                    str(output_file),
                    "--language-code",
                    "eng",
                    "--query",
                    "walk",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Imported 2 inflected forms from UniMorph" in result.output

    def test_import_unimorph_with_pos_filter(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_unimorph_items: list[LexicalItem],
    ) -> None:
        """Test UniMorph import with POS filter."""
        output_file = tmp_path / "unimorph_verbs.jsonl"

        with patch("bead.cli.resource_loaders.UniMorphAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = mock_unimorph_items
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_unimorph,
                [
                    "--output",
                    str(output_file),
                    "--language-code",
                    "eng",
                    "--pos",
                    "VERB",
                ],
            )

            assert result.exit_code == 0
            mock_adapter.fetch_items.assert_called_once()
            call_kwargs = mock_adapter.fetch_items.call_args[1]
            assert call_kwargs["pos"] == "VERB"

    def test_import_unimorph_with_features(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_unimorph_items: list[LexicalItem],
    ) -> None:
        """Test UniMorph import with features filter."""
        output_file = tmp_path / "unimorph_past.jsonl"

        with patch("bead.cli.resource_loaders.UniMorphAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = [mock_unimorph_items[0]]
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_unimorph,
                [
                    "--output",
                    str(output_file),
                    "--language-code",
                    "eng",
                    "--features",
                    "V;PST",
                ],
            )

            assert result.exit_code == 0
            mock_adapter.fetch_items.assert_called_once()
            call_kwargs = mock_adapter.fetch_items.call_args[1]
            assert call_kwargs["features"] == "V;PST"


class TestImportPropBank:
    """Test import-propbank command."""

    def test_import_propbank_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test basic PropBank import."""
        output_file = tmp_path / "propbank_predicates.jsonl"

        mock_items = [
            LexicalItem(
                lemma="eat.01",
                language_code="eng",
                features={"frameset": "eat.01"},
            ),
        ]

        with patch("bead.cli.resource_loaders.GlazingAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = mock_items
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_propbank,
                [
                    "--output",
                    str(output_file),
                    "--query",
                    "eat.01",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Imported 1 predicates from PropBank" in result.output


class TestImportFrameNet:
    """Test import-framenet command."""

    def test_import_framenet_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test basic FrameNet import."""
        output_file = tmp_path / "framenet_frames.jsonl"

        mock_items = [
            LexicalItem(
                lemma="Ingestion",
                language_code="eng",
                features={"frame": "Ingestion"},
            ),
        ]

        with patch("bead.cli.resource_loaders.GlazingAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = mock_items
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_framenet,
                [
                    "--output",
                    str(output_file),
                    "--query",
                    "Ingestion",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Imported 1 frames from FrameNet" in result.output

    def test_import_framenet_with_frame_filter(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test FrameNet import with frame filter."""
        output_file = tmp_path / "framenet_motion.jsonl"

        mock_items = [
            LexicalItem(
                lemma="Motion",
                language_code="eng",
                features={"frame": "Motion"},
            ),
        ]

        with patch("bead.cli.resource_loaders.GlazingAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.fetch_items.return_value = mock_items
            mock_adapter_class.return_value = mock_adapter

            result = runner.invoke(
                import_framenet,
                [
                    "--output",
                    str(output_file),
                    "--frame",
                    "Motion",
                    "--include-frames",
                ],
            )

            assert result.exit_code == 0
            mock_adapter.fetch_items.assert_called_once()
            call_kwargs = mock_adapter.fetch_items.call_args[1]
            assert call_kwargs["frame"] == "Motion"
            assert call_kwargs["include_frames"] is True
