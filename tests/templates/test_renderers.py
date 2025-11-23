"""Test template renderers."""

from __future__ import annotations

import pytest

from bead.resources.lexical_item import LexicalItem
from bead.resources.template import Slot
from bead.templates.renderers import DefaultRenderer, TemplateRenderer


def test_default_renderer_is_template_renderer() -> None:
    """Test that DefaultRenderer is a TemplateRenderer."""
    renderer = DefaultRenderer()
    assert isinstance(renderer, TemplateRenderer)


def test_default_renderer_simple_template() -> None:
    """Test DefaultRenderer with simple template."""
    renderer = DefaultRenderer()

    template_string = "{subject} {verb} {object}"
    slot_fillers = {
        "subject": LexicalItem(lemma="cat", language_code="eng"),
        "verb": LexicalItem(lemma="chase", language_code="eng"),
        "object": LexicalItem(lemma="mouse", language_code="eng"),
    }
    template_slots = {
        "subject": Slot(name="subject"),
        "verb": Slot(name="verb"),
        "object": Slot(name="object"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == "cat chase mouse"


def test_default_renderer_with_forms() -> None:
    """Test DefaultRenderer prioritizes form over lemma."""
    renderer = DefaultRenderer()

    template_string = "{subject} {verb} {object}"
    slot_fillers = {
        "subject": LexicalItem(
            lemma="cat", form="cats", language_code="eng"  # Plural form
        ),
        "verb": LexicalItem(
            lemma="chase", form="chased", language_code="eng"  # Past tense
        ),
        "object": LexicalItem(
            lemma="mouse", form="mice", language_code="eng"  # Plural form
        ),
    }
    template_slots = {
        "subject": Slot(name="subject"),
        "verb": Slot(name="verb"),
        "object": Slot(name="object"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    # Should use forms, not lemmas
    assert result == "cats chased mice"


def test_default_renderer_mixed_forms_and_lemmas() -> None:
    """Test DefaultRenderer with mix of forms and lemmas."""
    renderer = DefaultRenderer()

    template_string = "{det} {noun} {verb}"
    slot_fillers = {
        "det": LexicalItem(
            lemma="the", form=None, language_code="eng"  # No form, use lemma
        ),
        "noun": LexicalItem(
            lemma="cat", form="cats", language_code="eng"  # Has form
        ),
        "verb": LexicalItem(
            lemma="run", form=None, language_code="eng"  # No form, use lemma
        ),
    }
    template_slots = {
        "det": Slot(name="det"),
        "noun": Slot(name="noun"),
        "verb": Slot(name="verb"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == "the cats run"


def test_default_renderer_with_punctuation() -> None:
    """Test DefaultRenderer with template containing punctuation."""
    renderer = DefaultRenderer()

    template_string = "{subject} {verb} {object}."
    slot_fillers = {
        "subject": LexicalItem(lemma="John", language_code="eng"),
        "verb": LexicalItem(lemma="like", form="likes", language_code="eng"),
        "object": LexicalItem(lemma="apple", form="apples", language_code="eng"),
    }
    template_slots = {
        "subject": Slot(name="subject"),
        "verb": Slot(name="verb"),
        "object": Slot(name="object"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == "John likes apples."


def test_default_renderer_complex_template() -> None:
    """Test DefaultRenderer with complex template structure."""
    renderer = DefaultRenderer()

    template_string = "{det1} {noun1} {verb} {det2} {noun2} {prep} {det3} {noun3}"
    slot_fillers = {
        "det1": LexicalItem(lemma="the", language_code="eng"),
        "noun1": LexicalItem(lemma="cat", language_code="eng"),
        "verb": LexicalItem(lemma="put", form="puts", language_code="eng"),
        "det2": LexicalItem(lemma="a", language_code="eng"),
        "noun2": LexicalItem(lemma="book", language_code="eng"),
        "prep": LexicalItem(lemma="on", language_code="eng"),
        "det3": LexicalItem(lemma="the", language_code="eng"),
        "noun3": LexicalItem(lemma="table", language_code="eng"),
    }
    template_slots = {name: Slot(name=name) for name in slot_fillers.keys()}

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == "the cat puts a book on the table"


def test_default_renderer_empty_template() -> None:
    """Test DefaultRenderer with empty template string."""
    renderer = DefaultRenderer()

    template_string = ""
    slot_fillers: dict[str, LexicalItem] = {}
    template_slots: dict[str, Slot] = {}

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == ""


def test_default_renderer_no_placeholders() -> None:
    """Test DefaultRenderer with template containing no placeholders."""
    renderer = DefaultRenderer()

    template_string = "This is a plain sentence."
    slot_fillers: dict[str, LexicalItem] = {}
    template_slots: dict[str, Slot] = {}

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == "This is a plain sentence."


def test_default_renderer_extra_slot_fillers() -> None:
    """Test DefaultRenderer when slot_fillers has extra items not in template."""
    renderer = DefaultRenderer()

    template_string = "{subject} {verb}"
    slot_fillers = {
        "subject": LexicalItem(lemma="cat", language_code="eng"),
        "verb": LexicalItem(lemma="run", form="runs", language_code="eng"),
        "extra": LexicalItem(lemma="unused", language_code="eng"),  # Not in template
    }
    template_slots = {
        "subject": Slot(name="subject"),
        "verb": Slot(name="verb"),
        "extra": Slot(name="extra"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    # Should only render slots that appear in template
    assert result == "cat runs"
    assert "unused" not in result


def test_default_renderer_missing_slot_fillers() -> None:
    """Test DefaultRenderer when template has placeholders not in slot_fillers."""
    renderer = DefaultRenderer()

    template_string = "{subject} {verb} {object}"
    slot_fillers = {
        "subject": LexicalItem(lemma="cat", language_code="eng"),
        "verb": LexicalItem(lemma="run", form="runs", language_code="eng"),
        # Missing "object"
    }
    template_slots = {
        "subject": Slot(name="subject"),
        "verb": Slot(name="verb"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    # Placeholder should remain in output
    assert result == "cat runs {object}"


def test_default_renderer_repeated_slots() -> None:
    """Test DefaultRenderer with same slot appearing multiple times."""
    renderer = DefaultRenderer()

    template_string = "{noun} and {noun} and {noun}"
    slot_fillers = {
        "noun": LexicalItem(lemma="cat", language_code="eng"),
    }
    template_slots = {
        "noun": Slot(name="noun"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    # All occurrences should be replaced
    assert result == "cat and cat and cat"


def test_default_renderer_multilingual() -> None:
    """Test DefaultRenderer with multilingual items."""
    renderer = DefaultRenderer()

    template_string = "{korean_verb} {korean_object}"
    slot_fillers = {
        "korean_verb": LexicalItem(lemma="kkakta", language_code="kor"),
        "korean_object": LexicalItem(lemma="yuri", language_code="kor"),
    }
    template_slots = {
        "korean_verb": Slot(name="korean_verb"),
        "korean_object": Slot(name="korean_object"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == "kkakta yuri"


def test_default_renderer_with_special_characters() -> None:
    """Test DefaultRenderer with items containing special characters."""
    renderer = DefaultRenderer()

    template_string = "{subject} {verb} {object}"
    slot_fillers = {
        "subject": LexicalItem(lemma="O'Brien", language_code="eng"),
        "verb": LexicalItem(lemma="say", form="says", language_code="eng"),
        "object": LexicalItem(
            lemma='quote', form='"Hello!"', language_code="eng"
        ),
    }
    template_slots = {
        "subject": Slot(name="subject"),
        "verb": Slot(name="verb"),
        "object": Slot(name="object"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == 'O\'Brien says "Hello!"'


def test_default_renderer_preserves_template_whitespace() -> None:
    """Test that DefaultRenderer preserves whitespace in template."""
    renderer = DefaultRenderer()

    template_string = "{subject}  {verb}   {object}"  # Multiple spaces
    slot_fillers = {
        "subject": LexicalItem(lemma="cat", language_code="eng"),
        "verb": LexicalItem(lemma="chase", language_code="eng"),
        "object": LexicalItem(lemma="mouse", language_code="eng"),
    }
    template_slots = {
        "subject": Slot(name="subject"),
        "verb": Slot(name="verb"),
        "object": Slot(name="object"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == "cat  chase   mouse"


def test_default_renderer_case_sensitive() -> None:
    """Test that DefaultRenderer is case-sensitive for slot names."""
    renderer = DefaultRenderer()

    template_string = "{Subject} {verb}"
    slot_fillers = {
        "Subject": LexicalItem(lemma="Cat", language_code="eng"),  # Capital S
        "verb": LexicalItem(lemma="run", form="runs", language_code="eng"),
    }
    template_slots = {
        "Subject": Slot(name="Subject"),
        "verb": Slot(name="verb"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    assert result == "Cat runs"


class CustomTestRenderer(TemplateRenderer):
    """Custom renderer for testing subclass behavior."""

    def render(
        self,
        template_string: str,
        slot_fillers: dict[str, LexicalItem],
        template_slots: dict[str, Slot],
    ) -> str:
        """Render with custom logic: uppercase all text."""
        result = template_string
        for slot_name, item in slot_fillers.items():
            placeholder = f"{{{slot_name}}}"
            if placeholder in result:
                surface = item.form if item.form else item.lemma
                result = result.replace(placeholder, surface.upper())
        return result


def test_custom_renderer_subclass() -> None:
    """Test that custom renderer can override render method."""
    renderer = CustomTestRenderer()

    template_string = "{subject} {verb}"
    slot_fillers = {
        "subject": LexicalItem(lemma="cat", language_code="eng"),
        "verb": LexicalItem(lemma="run", form="runs", language_code="eng"),
    }
    template_slots = {
        "subject": Slot(name="subject"),
        "verb": Slot(name="verb"),
    }

    result = renderer.render(template_string, slot_fillers, template_slots)

    # Custom renderer uppercases everything
    assert result == "CAT RUNS"


def test_renderer_abstract_base_class() -> None:
    """Test that TemplateRenderer is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        TemplateRenderer()  # type: ignore
