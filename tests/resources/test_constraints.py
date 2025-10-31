"""Tests for constraint models."""

from __future__ import annotations

from bead.resources.constraints import Constraint


class TestConstraint:
    """Test unified Constraint class."""

    def test_create_basic_constraint(self) -> None:
        """Test creating a basic constraint."""
        constraint = Constraint(expression="self.pos == 'VERB'")
        assert constraint.expression == "self.pos == 'VERB'"
        assert constraint.context == {}
        assert constraint.description is None

    def test_create_with_context(self) -> None:
        """Test creating constraint with context."""
        context = {"allowed_verbs": {"break", "shatter"}}
        constraint = Constraint(
            expression="self.lemma in allowed_verbs", context=context
        )
        assert constraint.context == context

    def test_create_with_description(self) -> None:
        """Test creating constraint with description."""
        constraint = Constraint(
            expression="self.pos == 'VERB'", description="Verb constraint"
        )
        assert constraint.description == "Verb constraint"

    def test_serialization(self) -> None:
        """Test constraint serialization."""
        constraint = Constraint(
            expression="self.pos == 'VERB'",
            context={"key": "value"},
            description="Test",
        )
        data = constraint.model_dump()
        assert data["expression"] == "self.pos == 'VERB'"
        assert data["context"] == {"key": "value"}
        assert data["description"] == "Test"

    def test_deserialization(self) -> None:
        """Test constraint deserialization."""
        data = {
            "expression": "self.pos == 'NOUN'",
            "context": {"test": "value"},
        }
        constraint = Constraint.model_validate(data)
        assert constraint.expression == "self.pos == 'NOUN'"
        assert constraint.context == {"test": "value"}

    def test_empty_expression_allowed(self) -> None:
        """Test that empty expression is allowed."""
        constraint = Constraint(expression="")
        assert constraint.expression == ""

    def test_context_serialization(self) -> None:
        """Test context with various types serializes correctly."""
        constraint = Constraint(
            expression="test",
            context={
                "str_val": "test",
                "int_val": 42,
                "float_val": 3.14,
                "bool_val": True,
                "list_val": ["a", "b"],
                "set_val": {"x", "y"},
            },
        )
        data = constraint.model_dump()
        assert data["context"]["str_val"] == "test"
        assert data["context"]["int_val"] == 42
        assert data["context"]["bool_val"] is True
        # Sets may be serialized as lists
        assert "x" in data["context"]["set_val"]
