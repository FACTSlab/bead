"""Tests for constraint models."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from sash.data.identifiers import generate_uuid
from sash.resources import (
    DSLConstraint,
    ExtensionalConstraint,
    IntensionalConstraint,
    RelationalConstraint,
)
from sash.resources.constraints import Constraint


class TestExtensionalConstraint:
    """Test extensional constraint."""

    def test_create_with_allow_mode(self) -> None:
        """Test creating an extensional constraint with allow mode."""
        uuid1 = generate_uuid()
        uuid2 = generate_uuid()
        constraint = ExtensionalConstraint(mode="allow", items=[uuid1, uuid2])
        assert constraint.mode == "allow"
        assert len(constraint.items) == 2
        assert constraint.constraint_type == "extensional"

    def test_create_with_deny_mode(self) -> None:
        """Test creating an extensional constraint with deny mode."""
        uuid1 = generate_uuid()
        constraint = ExtensionalConstraint(mode="deny", items=[uuid1])
        assert constraint.mode == "deny"
        assert len(constraint.items) == 1

    def test_create_with_empty_items_list(self) -> None:
        """Test creating an extensional constraint with empty items list."""
        constraint = ExtensionalConstraint(mode="allow", items=[])
        assert constraint.items == []

    def test_create_with_multiple_items(self) -> None:
        """Test creating an extensional constraint with multiple items."""
        uuids = [generate_uuid() for _ in range(5)]
        constraint = ExtensionalConstraint(mode="allow", items=uuids)
        assert len(constraint.items) == 5

    def test_serialization(self) -> None:
        """Test extensional constraint serialization."""
        uuid1 = generate_uuid()
        constraint = ExtensionalConstraint(mode="allow", items=[uuid1])
        data = constraint.model_dump()
        assert data["mode"] == "allow"
        assert data["constraint_type"] == "extensional"

    def test_deserialization(self) -> None:
        """Test extensional constraint deserialization."""
        uuid1 = generate_uuid()
        data = {
            "mode": "deny",
            "items": [str(uuid1)],
            "constraint_type": "extensional",
        }
        constraint = ExtensionalConstraint.model_validate(data)
        assert constraint.mode == "deny"
        assert len(constraint.items) == 1

    def test_constraint_type_discriminator(self) -> None:
        """Test that constraint_type discriminator is set."""
        constraint = ExtensionalConstraint(mode="allow", items=[])
        assert constraint.constraint_type == "extensional"

    def test_invalid_mode_fails(self) -> None:
        """Test that invalid mode validation fails."""
        with pytest.raises(ValidationError):
            ExtensionalConstraint(mode="invalid", items=[])  # type: ignore


class TestIntensionalConstraint:
    """Test intensional constraint."""

    def test_create_with_equals_operator(self) -> None:
        """Test creating an intensional constraint with == operator."""
        constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")
        assert constraint.property == "pos"
        assert constraint.operator == "=="
        assert constraint.value == "VERB"
        assert constraint.constraint_type == "intensional"

    def test_create_with_not_equals_operator(self) -> None:
        """Test creating an intensional constraint with != operator."""
        constraint = IntensionalConstraint(
            property="lemma", operator="!=", value="test"
        )
        assert constraint.operator == "!="

    def test_create_with_in_operator(self) -> None:
        """Test creating an intensional constraint with in operator."""
        constraint = IntensionalConstraint(
            property="pos", operator="in", value=["VERB", "NOUN"]
        )
        assert constraint.operator == "in"
        assert isinstance(constraint.value, list)

    def test_create_with_less_than_operator(self) -> None:
        """Test creating an intensional constraint with < operator."""
        constraint = IntensionalConstraint(
            property="frequency", operator="<", value=100
        )
        assert constraint.operator == "<"
        assert constraint.value == 100

    def test_create_with_list_value(self) -> None:
        """Test creating an intensional constraint with list value for 'in' operator."""
        constraint = IntensionalConstraint(
            property="category", operator="in", value=["motion", "action"]
        )
        assert isinstance(constraint.value, list)
        assert len(constraint.value) == 2

    def test_with_property_path(self) -> None:
        """Test intensional constraint with property path like 'features.tense'."""
        constraint = IntensionalConstraint(
            property="features.tense", operator="==", value="past"
        )
        assert constraint.property == "features.tense"

    def test_serialization(self) -> None:
        """Test intensional constraint serialization."""
        constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")
        data = constraint.model_dump()
        assert data["property"] == "pos"
        assert data["operator"] == "=="
        assert data["value"] == "VERB"
        assert data["constraint_type"] == "intensional"

    def test_deserialization(self) -> None:
        """Test intensional constraint deserialization."""
        data = {
            "property": "lemma",
            "operator": "!=",
            "value": "test",
            "constraint_type": "intensional",
        }
        constraint = IntensionalConstraint.model_validate(data)
        assert constraint.property == "lemma"
        assert constraint.operator == "!="

    def test_constraint_type_discriminator(self) -> None:
        """Test that constraint_type discriminator is set."""
        constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")
        assert constraint.constraint_type == "intensional"

    def test_invalid_operator_fails(self) -> None:
        """Test that invalid operator validation fails."""
        with pytest.raises(ValidationError):
            IntensionalConstraint(property="pos", operator="invalid", value="VERB")  # type: ignore

    def test_empty_property_fails(self) -> None:
        """Test that empty property validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            IntensionalConstraint(property="", operator="==", value="VERB")
        assert "property must be non-empty" in str(exc_info.value)

    def test_whitespace_property_fails(self) -> None:
        """Test that whitespace-only property validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            IntensionalConstraint(property="   ", operator="==", value="VERB")
        assert "property must be non-empty" in str(exc_info.value)


class TestRelationalConstraint:
    """Test relational constraint."""

    def test_create_with_same_relation(self) -> None:
        """Test creating a relational constraint with 'same' relation."""
        constraint = RelationalConstraint(
            slot_a="subject",
            slot_b="object",
            relation="same",
            property="pos",
        )
        assert constraint.slot_a == "subject"
        assert constraint.slot_b == "object"
        assert constraint.relation == "same"
        assert constraint.property == "pos"
        assert constraint.constraint_type == "relational"

    def test_create_with_different_relation(self) -> None:
        """Test creating a relational constraint with 'different' relation."""
        constraint = RelationalConstraint(
            slot_a="subject",
            slot_b="object",
            relation="different",
            property="lemma",
        )
        assert constraint.relation == "different"

    def test_serialization(self) -> None:
        """Test relational constraint serialization."""
        constraint = RelationalConstraint(
            slot_a="subject",
            slot_b="object",
            relation="same",
            property="pos",
        )
        data = constraint.model_dump()
        assert data["slot_a"] == "subject"
        assert data["slot_b"] == "object"
        assert data["relation"] == "same"
        assert data["constraint_type"] == "relational"

    def test_deserialization(self) -> None:
        """Test relational constraint deserialization."""
        data = {
            "slot_a": "subject",
            "slot_b": "object",
            "relation": "different",
            "property": "lemma",
            "constraint_type": "relational",
        }
        constraint = RelationalConstraint.model_validate(data)
        assert constraint.slot_a == "subject"
        assert constraint.relation == "different"

    def test_constraint_type_discriminator(self) -> None:
        """Test that constraint_type discriminator is set."""
        constraint = RelationalConstraint(
            slot_a="subject",
            slot_b="object",
            relation="same",
            property="pos",
        )
        assert constraint.constraint_type == "relational"

    def test_invalid_relation_fails(self) -> None:
        """Test that invalid relation validation fails."""
        with pytest.raises(ValidationError):
            RelationalConstraint(
                slot_a="subject",
                slot_b="object",
                relation="invalid",  # type: ignore
                property="pos",
            )

    def test_empty_slot_a_fails(self) -> None:
        """Test that empty slot_a validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            RelationalConstraint(
                slot_a="",
                slot_b="object",
                relation="same",
                property="pos",
            )
        assert "field must be non-empty" in str(exc_info.value)

    def test_empty_property_fails(self) -> None:
        """Test that empty property validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            RelationalConstraint(
                slot_a="subject",
                slot_b="object",
                relation="same",
                property="",
            )
        assert "field must be non-empty" in str(exc_info.value)


class TestDSLConstraint:
    """Test DSL constraint."""

    def test_create_with_simple_expression(self) -> None:
        """Test creating a DSL constraint with simple expression."""
        constraint = DSLConstraint(expression="pos == 'VERB'")
        assert constraint.expression == "pos == 'VERB'"
        assert constraint.constraint_type == "dsl"

    def test_create_with_complex_expression(self) -> None:
        """Test creating a DSL constraint with complex expression."""
        constraint = DSLConstraint(
            expression="pos == 'VERB' and len(lemma) > 4 and frequency < 1000"
        )
        assert "and" in constraint.expression
        assert "len(lemma)" in constraint.expression

    def test_serialization(self) -> None:
        """Test DSL constraint serialization."""
        constraint = DSLConstraint(expression="pos == 'VERB'")
        data = constraint.model_dump()
        assert data["expression"] == "pos == 'VERB'"
        assert data["constraint_type"] == "dsl"

    def test_deserialization(self) -> None:
        """Test DSL constraint deserialization."""
        data = {
            "expression": "len(lemma) > 3",
            "constraint_type": "dsl",
        }
        constraint = DSLConstraint.model_validate(data)
        assert constraint.expression == "len(lemma) > 3"

    def test_constraint_type_discriminator(self) -> None:
        """Test that constraint_type discriminator is set."""
        constraint = DSLConstraint(expression="pos == 'VERB'")
        assert constraint.constraint_type == "dsl"

    def test_empty_expression_fails(self) -> None:
        """Test that empty expression validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            DSLConstraint(expression="")
        assert "expression must be non-empty" in str(exc_info.value)

    def test_whitespace_expression_fails(self) -> None:
        """Test that whitespace-only expression validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            DSLConstraint(expression="   ")
        assert "expression must be non-empty" in str(exc_info.value)


class TestConstraintUnionDiscrimination:
    """Test constraint union type discrimination."""

    def test_deserialize_extensional_via_union(self) -> None:
        """Test deserializing extensional constraint via union type."""
        data = {
            "constraint_type": "extensional",
            "mode": "allow",
            "items": [],
        }
        adapter = TypeAdapter(Constraint)
        constraint = adapter.validate_python(data)
        assert isinstance(constraint, ExtensionalConstraint)
        assert constraint.mode == "allow"

    def test_deserialize_intensional_via_union(self) -> None:
        """Test deserializing intensional constraint via union type."""
        data = {
            "constraint_type": "intensional",
            "property": "pos",
            "operator": "==",
            "value": "VERB",
        }
        adapter = TypeAdapter(Constraint)
        constraint = adapter.validate_python(data)
        assert isinstance(constraint, IntensionalConstraint)
        assert constraint.property == "pos"

    def test_deserialize_relational_via_union(self) -> None:
        """Test deserializing relational constraint via union type."""
        data = {
            "constraint_type": "relational",
            "slot_a": "subject",
            "slot_b": "object",
            "relation": "same",
            "property": "pos",
        }
        adapter = TypeAdapter(Constraint)
        constraint = adapter.validate_python(data)
        assert isinstance(constraint, RelationalConstraint)
        assert constraint.slot_a == "subject"

    def test_deserialize_dsl_via_union(self) -> None:
        """Test deserializing DSL constraint via union type."""
        data = {
            "constraint_type": "dsl",
            "expression": "pos == 'VERB'",
        }
        adapter = TypeAdapter(Constraint)
        constraint = adapter.validate_python(data)
        assert isinstance(constraint, DSLConstraint)
        assert constraint.expression == "pos == 'VERB'"

    def test_invalid_constraint_type_fails(self) -> None:
        """Test that invalid constraint type raises error."""
        data = {
            "constraint_type": "invalid",
            "some_field": "value",
        }
        adapter = TypeAdapter(Constraint)
        with pytest.raises(ValidationError):
            adapter.validate_python(data)
