"""Tests for UI component generation (Material Design CSS and helpers)."""

from uuid import uuid4

from bead.deployment.jspsych.ui.components import (
    create_cloze_fields,
    create_forced_choice_config,
    create_rating_scale,
    infer_widget_type,
)
from bead.deployment.jspsych.ui.styles import MaterialDesignStylesheet
from bead.items.models import UnfilledSlot
from bead.resources.constraints import Constraint


def test_material_design_stylesheet_creation() -> None:
    """Test Material Design stylesheet can be created."""
    stylesheet = MaterialDesignStylesheet()
    assert isinstance(stylesheet, MaterialDesignStylesheet)


def test_generate_css_light_theme() -> None:
    """Test generating CSS with light theme."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css(theme="light")

    assert isinstance(css, str)
    assert len(css) > 0
    assert "--primary-color" in css
    assert "--background" in css
    assert "Roboto" in css


def test_generate_css_dark_theme() -> None:
    """Test generating CSS with dark theme."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css(theme="dark")

    assert "--background: #121212" in css
    assert "--surface: #1E1E1E" in css


def test_generate_css_custom_colors() -> None:
    """Test generating CSS with custom colors."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css(
        theme="light", primary_color="#1976D2", secondary_color="#FF5722"
    )

    assert "#1976D2" in css
    assert "#FF5722" in css


def test_css_contains_material_design_classes() -> None:
    """Test that CSS contains Material Design classes."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css()

    # Check for key Material Design classes
    assert ".bead-button" in css
    assert ".bead-rating-scale" in css
    assert ".bead-text-field" in css
    assert ".bead-dropdown" in css
    assert ".bead-card" in css
    assert ".bead-progress" in css


def test_css_contains_rating_scale_styles() -> None:
    """Test that CSS contains rating scale styles."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css()

    assert ".bead-rating-container" in css
    assert ".bead-rating-prompt" in css
    assert ".bead-rating-button" in css
    assert ".bead-rating-label" in css


def test_css_contains_cloze_styles() -> None:
    """Test that CSS contains cloze task styles."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css()

    assert ".bead-cloze-container" in css
    assert ".bead-cloze-text" in css
    assert ".bead-cloze-field" in css


def test_css_contains_forced_choice_styles() -> None:
    """Test that CSS contains forced choice styles."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css()

    assert ".bead-forced-choice-container" in css
    assert ".bead-forced-choice-alternatives" in css
    assert ".bead-alternative" in css


def test_create_rating_scale() -> None:
    """Test creating rating scale configuration."""
    config = create_rating_scale(1, 7, {1: "Low", 7: "High"})

    assert config["scale_min"] == 1
    assert config["scale_max"] == 7
    assert config["scale_labels"][1] == "Low"
    assert config["scale_labels"][7] == "High"


def test_create_rating_scale_no_labels() -> None:
    """Test creating rating scale without labels."""
    config = create_rating_scale(1, 5)

    assert config["scale_min"] == 1
    assert config["scale_max"] == 5
    assert config["scale_labels"] == {}


def test_create_cloze_fields() -> None:
    """Test creating cloze field configurations."""
    slots = [
        UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[]),
        UnfilledSlot(slot_name="verb", position=2, constraint_ids=[]),
    ]

    fields = create_cloze_fields(slots, {})

    assert len(fields) == 2
    assert fields[0]["slot_name"] == "determiner"
    assert fields[0]["type"] == "text"  # No constraints → text input
    assert fields[1]["slot_name"] == "verb"
    assert fields[1]["position"] == 2


def test_create_cloze_fields_with_constraint() -> None:
    """Test creating cloze fields with constraints."""
    constraint_id = uuid4()
    constraint = Constraint(expression="self.pos == 'VERB'")

    slots = [
        UnfilledSlot(slot_name="verb", position=0, constraint_ids=[constraint_id]),
    ]

    fields = create_cloze_fields(slots, {constraint_id: constraint})

    assert len(fields) == 1
    assert fields[0]["slot_name"] == "verb"
    assert fields[0]["type"] == "text"
    assert fields[0]["dsl_expression"] == "self.pos == 'VERB'"


def test_create_forced_choice_config() -> None:
    """Test creating forced choice configuration."""
    config = create_forced_choice_config(
        ["Option A", "Option B"], randomize_position=False
    )

    assert len(config["alternatives"]) == 2
    assert config["randomize_position"] is False
    assert config["enable_keyboard"] is True


def test_infer_widget_type_no_constraints() -> None:
    """Test inferring widget type with no constraints."""
    widget_type = infer_widget_type([], {})
    assert widget_type == "text"


def test_infer_widget_type_with_constraint() -> None:
    """Test inferring widget type with constraint."""
    constraint_id = uuid4()
    constraint = Constraint(expression="self.pos == 'VERB'")

    widget_type = infer_widget_type([constraint_id], {constraint_id: constraint})
    assert widget_type == "text"
