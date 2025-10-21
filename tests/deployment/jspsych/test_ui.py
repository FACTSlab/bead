"""Tests for UI component generation (Material Design CSS and helpers)."""

from sash.deployment.jspsych.ui.components import (
    create_cloze_fields,
    create_forced_choice_config,
    create_rating_scale,
    infer_widget_type,
)
from sash.deployment.jspsych.ui.styles import MaterialDesignStylesheet
from sash.items.models import UnfilledSlot


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
    assert ".sash-button" in css
    assert ".sash-rating-scale" in css
    assert ".sash-text-field" in css
    assert ".sash-dropdown" in css
    assert ".sash-card" in css
    assert ".sash-progress" in css


def test_css_contains_rating_scale_styles() -> None:
    """Test that CSS contains rating scale styles."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css()

    assert ".sash-rating-container" in css
    assert ".sash-rating-prompt" in css
    assert ".sash-rating-button" in css
    assert ".sash-rating-label" in css


def test_css_contains_cloze_styles() -> None:
    """Test that CSS contains cloze task styles."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css()

    assert ".sash-cloze-container" in css
    assert ".sash-cloze-text" in css
    assert ".sash-cloze-field" in css


def test_css_contains_forced_choice_styles() -> None:
    """Test that CSS contains forced choice styles."""
    stylesheet = MaterialDesignStylesheet()
    css = stylesheet.generate_css()

    assert ".sash-forced-choice-container" in css
    assert ".sash-forced-choice-alternatives" in css
    assert ".sash-alternative" in css


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


def test_create_cloze_fields_with_extensional_constraint() -> None:
    """Test creating cloze fields with extensional constraints."""
    from uuid import uuid4

    from sash.resources.constraints import ExtensionalConstraint

    constraint_id = uuid4()
    lexical_item_id1 = uuid4()
    lexical_item_id2 = uuid4()

    constraint = ExtensionalConstraint(
        mode="allow", items=[lexical_item_id1, lexical_item_id2]
    )

    slots = [
        UnfilledSlot(
            slot_name="determiner", position=0, constraint_ids=[constraint_id]
        ),
    ]

    fields = create_cloze_fields(slots, {constraint_id: constraint})

    assert len(fields) == 1
    assert fields[0]["slot_name"] == "determiner"
    assert fields[0]["type"] == "dropdown"
    assert "extensional_item_ids" in fields[0]
    assert len(fields[0]["extensional_item_ids"]) == 2


def test_create_cloze_fields_with_intensional_constraint() -> None:
    """Test creating cloze fields with intensional constraints."""
    from uuid import uuid4

    from sash.resources.constraints import IntensionalConstraint

    constraint_id = uuid4()
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")

    slots = [
        UnfilledSlot(slot_name="verb", position=0, constraint_ids=[constraint_id]),
    ]

    fields = create_cloze_fields(slots, {constraint_id: constraint})

    assert len(fields) == 1
    assert fields[0]["slot_name"] == "verb"
    assert fields[0]["type"] == "text"


def test_create_cloze_fields_with_dsl_constraint() -> None:
    """Test creating cloze fields with DSL constraints."""
    from uuid import uuid4

    from sash.resources.constraints import DSLConstraint

    constraint_id = uuid4()
    constraint = DSLConstraint(expression="pos == 'VERB' and len(lemma) > 4")

    slots = [
        UnfilledSlot(slot_name="verb", position=0, constraint_ids=[constraint_id]),
    ]

    fields = create_cloze_fields(slots, {constraint_id: constraint})

    assert len(fields) == 1
    assert fields[0]["slot_name"] == "verb"
    assert fields[0]["type"] == "text"
    assert fields[0]["dsl_expression"] == "pos == 'VERB' and len(lemma) > 4"


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


def test_infer_widget_type_extensional_allow() -> None:
    """Test inferring widget type with extensional allow constraint."""
    from uuid import uuid4

    from sash.resources.constraints import ExtensionalConstraint

    constraint_id = uuid4()
    constraint = ExtensionalConstraint(mode="allow", items=[uuid4(), uuid4()])

    widget_type = infer_widget_type([constraint_id], {constraint_id: constraint})
    assert widget_type == "dropdown"


def test_infer_widget_type_extensional_deny() -> None:
    """Test inferring widget type with extensional deny constraint."""
    from uuid import uuid4

    from sash.resources.constraints import ExtensionalConstraint

    constraint_id = uuid4()
    constraint = ExtensionalConstraint(mode="deny", items=[uuid4()])

    widget_type = infer_widget_type([constraint_id], {constraint_id: constraint})
    assert widget_type == "text"  # Deny mode uses text input


def test_infer_widget_type_intensional() -> None:
    """Test inferring widget type with intensional constraint."""
    from uuid import uuid4

    from sash.resources.constraints import IntensionalConstraint

    constraint_id = uuid4()
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")

    widget_type = infer_widget_type([constraint_id], {constraint_id: constraint})
    assert widget_type == "text"
