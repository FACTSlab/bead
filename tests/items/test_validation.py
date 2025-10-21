"""Tests for item validation utilities."""

from __future__ import annotations

from uuid import uuid4

import pytest

from sash.items.models import (
    Item,
    ItemElement,
    ItemTemplate,
    ModelOutput,
    PresentationSpec,
    TaskSpec,
)
from sash.items.validation import (
    item_passes_all_constraints,
    validate_constraint_satisfaction,
    validate_item,
    validate_metadata_completeness,
    validate_model_output,
)


@pytest.fixture
def simple_template():
    """Create a simple item template for testing."""
    return ItemTemplate(
        name="test_template",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this natural?"),
        presentation_spec=PresentationSpec(mode="static"),
        elements=[
            ItemElement(
                element_type="text", element_name="sentence", content="Test sentence"
            )
        ],
        constraints=[uuid4(), uuid4()],
    )


@pytest.fixture
def simple_item(simple_template):
    """Create a simple item for testing."""
    return Item(
        item_template_id=simple_template.id,
        rendered_elements={"sentence": "Test sentence"},
        constraint_satisfaction={
            simple_template.constraints[0]: True,
            simple_template.constraints[1]: True,
        },
        model_outputs=[],
    )


class TestValidateItem:
    """Tests for validate_item function."""

    def test_valid_item(self, simple_item, simple_template) -> None:
        """Test validation of a valid item."""
        errors = validate_item(simple_item, simple_template)
        assert errors == []

    def test_template_id_mismatch(self, simple_item, simple_template) -> None:
        """Test detection of template ID mismatch."""
        simple_item.item_template_id = uuid4()
        errors = validate_item(simple_item, simple_template)
        assert len(errors) == 1
        assert "template ID mismatch" in errors[0]

    def test_missing_rendered_elements(self, simple_item, simple_template) -> None:
        """Test detection of missing rendered elements."""
        simple_item.rendered_elements = {}
        errors = validate_item(simple_item, simple_template)
        assert any("Missing rendered elements" in e for e in errors)

    def test_extra_rendered_elements(self, simple_item, simple_template) -> None:
        """Test detection of extra rendered elements."""
        simple_item.rendered_elements["extra"] = "Extra element"
        errors = validate_item(simple_item, simple_template)
        assert any("Extra rendered elements" in e for e in errors)

    def test_missing_constraint_evaluation(self, simple_item, simple_template) -> None:
        """Test detection of missing constraint evaluations."""
        simple_item.constraint_satisfaction = {}
        errors = validate_item(simple_item, simple_template)
        assert any("Missing constraint evaluations" in e for e in errors)

    def test_invalid_model_output(self, simple_item, simple_template) -> None:
        """Test that invalid model outputs are detected."""
        # Create a model output with wrong type for operation
        simple_item.model_outputs = [
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="log_probability",
                inputs={"text": "test"},
                output="not a number",  # Invalid: should be numeric
                cache_key="abc123",
            )
        ]
        errors = validate_item(simple_item, simple_template)
        assert any("should be numeric" in e for e in errors)


class TestValidateModelOutput:
    """Tests for validate_model_output function."""

    def test_valid_log_probability_output(self) -> None:
        """Test validation of valid log probability output."""
        output = ModelOutput(
            model_name="gpt2",
            model_version="1.0",
            operation="log_probability",
            inputs={"text": "test"},
            output=-42.5,
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert errors == []

    def test_valid_nli_output(self) -> None:
        """Test validation of valid NLI output."""
        output = ModelOutput(
            model_name="roberta-nli",
            model_version="1.0",
            operation="nli",
            inputs={"premise": "p", "hypothesis": "h"},
            output={"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
            cache_key="xyz789",
        )
        errors = validate_model_output(output)
        assert errors == []

    def test_empty_model_name(self) -> None:
        """Test Pydantic validation prevents empty model name."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelOutput(
                model_name="",  # Pydantic will reject this
                model_version="1.0",
                operation="log_probability",
                inputs={"text": "test"},
                output=-42.0,
                cache_key="abc123",
            )

    def test_empty_operation(self) -> None:
        """Test Pydantic validation prevents empty operation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelOutput(
                model_name="gpt2",
                model_version="1.0",
                operation="",  # Pydantic will reject this
                inputs={"text": "test"},
                output=-42.0,
                cache_key="abc123",
            )

    def test_empty_cache_key(self) -> None:
        """Test Pydantic validation prevents empty cache key."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelOutput(
                model_name="gpt2",
                model_version="1.0",
                operation="log_probability",
                inputs={"text": "test"},
                output=-42.0,
                cache_key="",  # Pydantic will reject this
            )

    def test_nli_output_not_dict(self) -> None:
        """Test detection of NLI output that's not a dict."""
        output = ModelOutput(
            model_name="roberta-nli",
            model_version="1.0",
            operation="nli",
            inputs={"premise": "p", "hypothesis": "h"},
            output=0.8,  # Should be dict
            cache_key="xyz789",
        )
        errors = validate_model_output(output)
        assert any("should be dict" in e for e in errors)

    def test_nli_output_missing_keys(self) -> None:
        """Test detection of NLI output with missing keys."""
        output = ModelOutput(
            model_name="roberta-nli",
            model_version="1.0",
            operation="nli",
            inputs={"premise": "p", "hypothesis": "h"},
            output={"entailment": 0.8},  # Missing neutral and contradiction
            cache_key="xyz789",
        )
        errors = validate_model_output(output)
        assert any("keys mismatch" in e for e in errors)

    def test_log_probability_non_numeric(self) -> None:
        """Test detection of non-numeric log probability."""
        output = ModelOutput(
            model_name="gpt2",
            model_version="1.0",
            operation="log_probability",
            inputs={"text": "test"},
            output="not a number",
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert any("should be numeric" in e for e in errors)

    def test_perplexity_non_numeric(self) -> None:
        """Test detection of non-numeric perplexity."""
        output = ModelOutput(
            model_name="gpt2",
            model_version="1.0",
            operation="perplexity",
            inputs={"text": "test"},
            output=[1, 2, 3],
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert any("should be numeric" in e for e in errors)

    def test_similarity_non_numeric(self) -> None:
        """Test detection of non-numeric similarity."""
        output = ModelOutput(
            model_name="sentence-transformer",
            model_version="1.0",
            operation="similarity",
            inputs={"text1": "a", "text2": "b"},
            output=None,
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert any("should be numeric" in e for e in errors)

    def test_embedding_non_list(self) -> None:
        """Test detection of embedding that's not a list."""
        output = ModelOutput(
            model_name="sentence-transformer",
            model_version="1.0",
            operation="embedding",
            inputs={"text": "test"},
            output=42,  # Should be list or dict (serialized array)
            cache_key="abc123",
        )
        errors = validate_model_output(output)
        assert any("should be list/array" in e for e in errors)


class TestValidateConstraintSatisfaction:
    """Tests for validate_constraint_satisfaction function."""

    def test_valid_constraint_satisfaction(self, simple_item, simple_template) -> None:
        """Test validation of valid constraint satisfaction."""
        errors = validate_constraint_satisfaction(simple_item, simple_template)
        assert errors == []

    def test_missing_constraint(self, simple_item, simple_template) -> None:
        """Test detection of missing constraint evaluation."""
        simple_item.constraint_satisfaction = {simple_template.constraints[0]: True}
        errors = validate_constraint_satisfaction(simple_item, simple_template)
        assert len(errors) == 1
        assert "not evaluated" in errors[0]

    def test_non_boolean_value(self, simple_item, simple_template) -> None:
        """Test detection of non-boolean satisfaction value."""
        simple_item.constraint_satisfaction[simple_template.constraints[0]] = "true"
        errors = validate_constraint_satisfaction(simple_item, simple_template)
        assert any("should be bool" in e for e in errors)

    def test_all_constraints_missing(self, simple_item, simple_template) -> None:
        """Test when all constraints are missing."""
        simple_item.constraint_satisfaction = {}
        errors = validate_constraint_satisfaction(simple_item, simple_template)
        assert len(errors) == len(simple_template.constraints)


class TestValidateMetadataCompleteness:
    """Tests for validate_metadata_completeness function."""

    def test_valid_metadata(self, simple_item) -> None:
        """Test validation of item with complete metadata."""
        errors = validate_metadata_completeness(simple_item)
        # Should have id, created_at, modified_at from SashBaseModel
        assert errors == []

    def test_item_has_id(self, simple_item) -> None:
        """Test that item has id field."""
        assert hasattr(simple_item, "id")
        assert simple_item.id is not None

    def test_item_has_timestamps(self, simple_item) -> None:
        """Test that item has timestamp fields."""
        assert hasattr(simple_item, "created_at")
        assert hasattr(simple_item, "modified_at")
        assert simple_item.created_at is not None
        assert simple_item.modified_at is not None


class TestItemPassesAllConstraints:
    """Tests for item_passes_all_constraints function."""

    def test_all_constraints_pass(self, simple_item) -> None:
        """Test when all constraints are satisfied."""
        assert item_passes_all_constraints(simple_item) is True

    def test_one_constraint_fails(self, simple_item, simple_template) -> None:
        """Test when one constraint fails."""
        simple_item.constraint_satisfaction[simple_template.constraints[0]] = False
        assert item_passes_all_constraints(simple_item) is False

    def test_all_constraints_fail(self, simple_item, simple_template) -> None:
        """Test when all constraints fail."""
        for constraint_id in simple_template.constraints:
            simple_item.constraint_satisfaction[constraint_id] = False
        assert item_passes_all_constraints(simple_item) is False

    def test_no_constraints(self) -> None:
        """Test item with no constraints."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"test": "text"},
            constraint_satisfaction={},
        )
        assert item_passes_all_constraints(item) is True

    def test_mixed_constraints(self, simple_item, simple_template) -> None:
        """Test with mixed constraint satisfaction."""
        simple_item.constraint_satisfaction[simple_template.constraints[0]] = True
        simple_item.constraint_satisfaction[simple_template.constraints[1]] = False
        assert item_passes_all_constraints(simple_item) is False
