"""Test CSP-based template filler."""

from __future__ import annotations

import pytest

from bead.resources.constraints import Constraint
from bead.resources.lexicon import Lexicon
from bead.resources.models import LexicalItem
from bead.resources.structures import Slot, Template
from bead.templates.filler import (
    ConstraintUnsatisfiableError,
    CSPFiller,
    FilledTemplate,
)


@pytest.fixture
def sample_lexicon() -> Lexicon:
    """Create sample lexicon for testing."""
    lexicon = Lexicon(name="test_lexicon")

    # Add singular nouns
    lexicon.add(
        LexicalItem(
            lemma="cat",
            pos="NOUN",
            language_code="en",
            features={"number": "singular"},
        )
    )
    lexicon.add(
        LexicalItem(
            lemma="dog",
            pos="NOUN",
            language_code="en",
            features={"number": "singular"},
        )
    )

    # Add plural nouns
    lexicon.add(
        LexicalItem(
            lemma="cats",
            pos="NOUN",
            language_code="en",
            features={"number": "plural"},
        )
    )

    # Add singular verbs
    lexicon.add(
        LexicalItem(
            lemma="runs",
            pos="VERB",
            language_code="en",
            features={"number": "singular"},
        )
    )

    # Add plural verbs
    lexicon.add(
        LexicalItem(
            lemma="run",
            pos="VERB",
            language_code="en",
            features={"number": "plural"},
        )
    )

    return lexicon


def test_csp_filler_basic() -> None:
    """Test basic CSP filler creation."""
    lexicon = Lexicon(name="test")
    filler = CSPFiller(lexicon)
    assert filler.lexicon == lexicon
    assert filler.max_attempts == 10000


def test_csp_filler_returns_iterator(sample_lexicon: Lexicon) -> None:
    """Test that fill() returns an iterator."""
    template = Template(
        name="simple",
        template_string="{subject} {verb}",
        slots={
            "subject": Slot(
                name="subject",
                constraints=[Constraint(expression="self.pos == 'NOUN'")],
            ),
            "verb": Slot(
                name="verb", constraints=[Constraint(expression="self.pos == 'VERB'")]
            ),
        },
    )

    filler = CSPFiller(sample_lexicon)
    result = filler.fill(template, count=1)

    # Check it's an iterator
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

    # Consume one item
    filled = next(result)
    assert isinstance(filled, FilledTemplate)
    assert filled.strategy_name == "backtracking"


def test_csp_filler_multi_slot_agreement(sample_lexicon: Lexicon) -> None:
    """Test multi-slot subject-verb agreement constraint."""
    template = Template(
        name="agreement",
        template_string="{subject} {verb}",
        slots={
            "subject": Slot(
                name="subject",
                constraints=[Constraint(expression="self.pos == 'NOUN'")],
            ),
            "verb": Slot(
                name="verb", constraints=[Constraint(expression="self.pos == 'VERB'")]
            ),
        },
        constraints=[
            Constraint(
                expression=(
                    "subject.features.get('number') == verb.features.get('number')"
                )
            )
        ],
    )

    filler = CSPFiller(sample_lexicon)
    filled_list = list(filler.fill(template, count=10))

    # Should find valid combinations
    assert len(filled_list) > 0

    # Check all satisfy agreement
    for filled in filled_list:
        subj_num = filled.slot_fillers["subject"].features.get("number")
        verb_num = filled.slot_fillers["verb"].features.get("number")
        assert subj_num == verb_num


def test_csp_filler_count_parameter(sample_lexicon: Lexicon) -> None:
    """Test count parameter limits results."""
    template = Template(
        name="simple",
        template_string="{noun}",
        slots={
            "noun": Slot(
                name="noun", constraints=[Constraint(expression="self.pos == 'NOUN'")]
            ),
        },
    )

    filler = CSPFiller(sample_lexicon)

    # Request 2 solutions
    filled_list = list(filler.fill(template, count=2))
    assert len(filled_list) == 2


def test_csp_filler_unsatisfiable_constraint(sample_lexicon: Lexicon) -> None:
    """Test ConstraintUnsatisfiableError on impossible constraints."""
    template = Template(
        name="impossible",
        template_string="{word}",
        slots={
            "word": Slot(
                name="word", constraints=[Constraint(expression="self.pos == 'ADV'")]
            ),
        },
    )

    filler = CSPFiller(sample_lexicon)

    with pytest.raises(ConstraintUnsatisfiableError) as exc_info:
        list(filler.fill(template, count=1))

    assert "impossible" in str(exc_info.value).lower()


def test_csp_filler_metadata_preservation(sample_lexicon: Lexicon) -> None:
    """Test that metadata is preserved in filled templates."""
    template = Template(
        name="test_template",
        template_string="{noun}",
        slots={
            "noun": Slot(
                name="noun", constraints=[Constraint(expression="self.pos == 'NOUN'")]
            ),
        },
    )

    filler = CSPFiller(sample_lexicon)
    filled = next(filler.fill(template, count=1))

    assert filled.template_id == str(template.id)
    assert filled.template_name == "test_template"
    assert filled.strategy_name == "backtracking"
    assert "noun" in filled.slot_fillers


def test_csp_filler_language_filtering(sample_lexicon: Lexicon) -> None:
    """Test language code filtering."""
    # Add Spanish items
    sample_lexicon.add(LexicalItem(lemma="gato", pos="NOUN", language_code="es"))
    sample_lexicon.add(LexicalItem(lemma="perro", pos="NOUN", language_code="es"))

    template = Template(
        name="simple",
        template_string="{noun}",
        slots={
            "noun": Slot(
                name="noun", constraints=[Constraint(expression="self.pos == 'NOUN'")]
            ),
        },
    )

    filler = CSPFiller(sample_lexicon)

    # Fill with English only
    filled_en = list(filler.fill(template, language_code="en", count=10))
    rendered_en = {f.rendered_text for f in filled_en}
    # Should only have English nouns
    assert all(text in {"cat", "dog", "cats"} for text in rendered_en)

    # Fill with Spanish only
    filled_es = list(filler.fill(template, language_code="es", count=10))
    rendered_es = {f.rendered_text for f in filled_es}
    assert rendered_es == {"gato", "perro"}


def test_csp_filler_relational_constraint(sample_lexicon: Lexicon) -> None:
    """Test relational constraint (different items in slots)."""
    template = Template(
        name="relational",
        template_string="{noun1} and {noun2}",
        slots={
            "noun1": Slot(
                name="noun1", constraints=[Constraint(expression="self.pos == 'NOUN'")]
            ),
            "noun2": Slot(
                name="noun2", constraints=[Constraint(expression="self.pos == 'NOUN'")]
            ),
        },
        constraints=[Constraint(expression="noun1.lemma != noun2.lemma")],
    )

    filler = CSPFiller(sample_lexicon)
    filled_list = list(filler.fill(template, count=5))

    # Should find valid combinations
    assert len(filled_list) > 0

    # Check all have different nouns
    for filled in filled_list:
        noun1_lemma = filled.slot_fillers["noun1"].lemma
        noun2_lemma = filled.slot_fillers["noun2"].lemma
        assert noun1_lemma != noun2_lemma
