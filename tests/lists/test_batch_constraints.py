"""Tests for batch-level constraint models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bead.lists.constraints import (
    BatchBalanceConstraint,
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint,
)


class TestBatchCoverageConstraint:
    """Tests for BatchCoverageConstraint model."""

    def test_create_basic(self) -> None:
        """Test creating basic coverage constraint."""
        constraint = BatchCoverageConstraint(
            property_expression="item['template_id']",
            target_values=list(range(10)),
        )

        assert constraint.constraint_type == "coverage"
        assert constraint.property_expression == "item['template_id']"
        assert constraint.target_values == list(range(10))
        assert constraint.min_coverage == 1.0

    def test_create_with_min_coverage(self) -> None:
        """Test creating with custom min_coverage."""
        constraint = BatchCoverageConstraint(
            property_expression="item['template_id']",
            target_values=list(range(10)),
            min_coverage=0.9,
        )

        assert constraint.min_coverage == 0.9

    def test_create_without_target_values(self) -> None:
        """Test creating without target_values (auto-detection)."""
        constraint = BatchCoverageConstraint(
            property_expression="item['verb']",
            target_values=None,
        )

        assert constraint.target_values is None

    def test_property_expression_validation_empty(self) -> None:
        """Test empty property_expression raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchCoverageConstraint(
                property_expression="",
                target_values=list(range(10)),
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_property_expression_validation_whitespace(self) -> None:
        """Test whitespace-only property_expression raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchCoverageConstraint(
                property_expression="   ",
                target_values=list(range(10)),
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_property_expression_strips_whitespace(self) -> None:
        """Test property_expression whitespace is stripped."""
        constraint = BatchCoverageConstraint(
            property_expression="  item['test']  ",
            target_values=[1, 2, 3],
        )
        assert constraint.property_expression == "item['test']"

    def test_min_coverage_validation_negative(self) -> None:
        """Test negative min_coverage raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchCoverageConstraint(
                property_expression="item['test']",
                target_values=[1, 2, 3],
                min_coverage=-0.1,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_min_coverage_validation_too_large(self) -> None:
        """Test min_coverage > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchCoverageConstraint(
                property_expression="item['test']",
                target_values=[1, 2, 3],
                min_coverage=1.5,
            )
        assert "less than or equal to 1" in str(exc_info.value)

    def test_min_coverage_validation_zero(self) -> None:
        """Test min_coverage=0.0 is valid."""
        constraint = BatchCoverageConstraint(
            property_expression="item['test']",
            target_values=[1, 2, 3],
            min_coverage=0.0,
        )
        assert constraint.min_coverage == 0.0

    def test_min_coverage_validation_one(self) -> None:
        """Test min_coverage=1.0 is valid."""
        constraint = BatchCoverageConstraint(
            property_expression="item['test']",
            target_values=[1, 2, 3],
            min_coverage=1.0,
        )
        assert constraint.min_coverage == 1.0

    def test_constraint_type_is_coverage(self) -> None:
        """Test discriminator is correct."""
        constraint = BatchCoverageConstraint(
            property_expression="item['test']",
            target_values=[1, 2, 3],
        )
        assert constraint.constraint_type == "coverage"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        constraint = BatchCoverageConstraint(
            property_expression="item['template_id']",
            target_values=list(range(26)),
            min_coverage=0.95,
        )

        data = constraint.model_dump()
        restored = BatchCoverageConstraint(**data)

        assert restored.property_expression == constraint.property_expression
        assert restored.target_values == constraint.target_values
        assert restored.min_coverage == constraint.min_coverage

    def test_inherits_beadbasemodel(self) -> None:
        """Test has BeadBaseModel fields."""
        constraint = BatchCoverageConstraint(
            property_expression="item['test']",
            target_values=[1, 2, 3],
        )

        assert hasattr(constraint, "id")
        assert hasattr(constraint, "created_at")
        assert hasattr(constraint, "modified_at")


class TestBatchBalanceConstraint:
    """Tests for BatchBalanceConstraint model."""

    def test_create_basic(self) -> None:
        """Test creating basic balance constraint."""
        constraint = BatchBalanceConstraint(
            property_expression="item['pair_type']",
            target_distribution={"same_verb": 0.5, "different_verb": 0.5},
        )

        assert constraint.constraint_type == "balance"
        assert constraint.property_expression == "item['pair_type']"
        assert constraint.target_distribution == {
            "same_verb": 0.5,
            "different_verb": 0.5,
        }
        assert constraint.tolerance == 0.1

    def test_create_with_tolerance(self) -> None:
        """Test creating with custom tolerance."""
        constraint = BatchBalanceConstraint(
            property_expression="item['pair_type']",
            target_distribution={"A": 0.5, "B": 0.5},
            tolerance=0.05,
        )

        assert constraint.tolerance == 0.05

    def test_property_expression_validation_empty(self) -> None:
        """Test empty property_expression raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchBalanceConstraint(
                property_expression="",
                target_distribution={"A": 0.5, "B": 0.5},
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_property_expression_validation_whitespace(self) -> None:
        """Test whitespace-only property_expression raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchBalanceConstraint(
                property_expression="   ",
                target_distribution={"A": 0.5, "B": 0.5},
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_tolerance_validation_negative(self) -> None:
        """Test negative tolerance raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchBalanceConstraint(
                property_expression="item['test']",
                target_distribution={"A": 0.5, "B": 0.5},
                tolerance=-0.1,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_tolerance_validation_too_large(self) -> None:
        """Test tolerance > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchBalanceConstraint(
                property_expression="item['test']",
                target_distribution={"A": 0.5, "B": 0.5},
                tolerance=1.5,
            )
        assert "less than or equal to 1" in str(exc_info.value)

    def test_target_distribution_validation_empty(self) -> None:
        """Test empty target_distribution raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchBalanceConstraint(
                property_expression="item['test']",
                target_distribution={},
            )
        assert "must not be empty" in str(exc_info.value)

    def test_target_distribution_validation_negative_values(self) -> None:
        """Test negative target_distribution values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchBalanceConstraint(
                property_expression="item['test']",
                target_distribution={"A": 0.5, "B": -0.5},
            )
        assert "must be in [0, 1]" in str(exc_info.value)

    def test_target_distribution_validation_values_too_large(self) -> None:
        """Test target_distribution values > 1.0 raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchBalanceConstraint(
                property_expression="item['test']",
                target_distribution={"A": 0.5, "B": 1.5},
            )
        assert "must be in [0, 1]" in str(exc_info.value)

    def test_target_distribution_validation_sum_not_one(self) -> None:
        """Test target_distribution not summing to 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchBalanceConstraint(
                property_expression="item['test']",
                target_distribution={"A": 0.3, "B": 0.3},
            )
        assert "must sum to ~1.0" in str(exc_info.value)

    def test_target_distribution_validation_sum_valid_range(self) -> None:
        """Test target_distribution summing to ~1.0 is valid."""
        constraint = BatchBalanceConstraint(
            property_expression="item['test']",
            target_distribution={"A": 0.333, "B": 0.333, "C": 0.334},
        )
        assert constraint.target_distribution == {"A": 0.333, "B": 0.333, "C": 0.334}

    def test_constraint_type_is_balance(self) -> None:
        """Test discriminator is correct."""
        constraint = BatchBalanceConstraint(
            property_expression="item['test']",
            target_distribution={"A": 0.5, "B": 0.5},
        )
        assert constraint.constraint_type == "balance"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        constraint = BatchBalanceConstraint(
            property_expression="item['pair_type']",
            target_distribution={"same": 0.6, "different": 0.4},
            tolerance=0.05,
        )

        data = constraint.model_dump()
        restored = BatchBalanceConstraint(**data)

        assert restored.property_expression == constraint.property_expression
        assert restored.target_distribution == constraint.target_distribution
        assert restored.tolerance == constraint.tolerance


class TestBatchDiversityConstraint:
    """Tests for BatchDiversityConstraint model."""

    def test_create_basic(self) -> None:
        """Test creating basic diversity constraint."""
        constraint = BatchDiversityConstraint(
            property_expression="item['verb_lemma']",
            max_lists_per_value=3,
        )

        assert constraint.constraint_type == "diversity"
        assert constraint.property_expression == "item['verb_lemma']"
        assert constraint.max_lists_per_value == 3

    def test_property_expression_validation_empty(self) -> None:
        """Test empty property_expression raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchDiversityConstraint(
                property_expression="",
                max_lists_per_value=3,
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_property_expression_validation_whitespace(self) -> None:
        """Test whitespace-only property_expression raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchDiversityConstraint(
                property_expression="   ",
                max_lists_per_value=3,
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_max_lists_per_value_validation_zero(self) -> None:
        """Test max_lists_per_value=0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchDiversityConstraint(
                property_expression="item['test']",
                max_lists_per_value=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_max_lists_per_value_validation_negative(self) -> None:
        """Test negative max_lists_per_value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchDiversityConstraint(
                property_expression="item['test']",
                max_lists_per_value=-1,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_max_lists_per_value_validation_one(self) -> None:
        """Test max_lists_per_value=1 is valid."""
        constraint = BatchDiversityConstraint(
            property_expression="item['test']",
            max_lists_per_value=1,
        )
        assert constraint.max_lists_per_value == 1

    def test_constraint_type_is_diversity(self) -> None:
        """Test discriminator is correct."""
        constraint = BatchDiversityConstraint(
            property_expression="item['test']",
            max_lists_per_value=2,
        )
        assert constraint.constraint_type == "diversity"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        constraint = BatchDiversityConstraint(
            property_expression="item['verb']",
            max_lists_per_value=4,
        )

        data = constraint.model_dump()
        restored = BatchDiversityConstraint(**data)

        assert restored.property_expression == constraint.property_expression
        assert restored.max_lists_per_value == constraint.max_lists_per_value

    def test_inherits_beadbasemodel(self) -> None:
        """Test has BeadBaseModel fields."""
        constraint = BatchDiversityConstraint(
            property_expression="item['test']",
            max_lists_per_value=3,
        )

        assert hasattr(constraint, "id")
        assert hasattr(constraint, "created_at")
        assert hasattr(constraint, "modified_at")


class TestBatchMinOccurrenceConstraint:
    """Tests for BatchMinOccurrenceConstraint model."""

    def test_create_basic(self) -> None:
        """Test creating basic min occurrence constraint."""
        constraint = BatchMinOccurrenceConstraint(
            property_expression="item['quantile']",
            min_occurrences=50,
        )

        assert constraint.constraint_type == "min_occurrence"
        assert constraint.property_expression == "item['quantile']"
        assert constraint.min_occurrences == 50

    def test_property_expression_validation_empty(self) -> None:
        """Test empty property_expression raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchMinOccurrenceConstraint(
                property_expression="",
                min_occurrences=10,
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_property_expression_validation_whitespace(self) -> None:
        """Test whitespace-only property_expression raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchMinOccurrenceConstraint(
                property_expression="   ",
                min_occurrences=10,
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_min_occurrences_validation_zero(self) -> None:
        """Test min_occurrences=0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchMinOccurrenceConstraint(
                property_expression="item['test']",
                min_occurrences=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_min_occurrences_validation_negative(self) -> None:
        """Test negative min_occurrences raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BatchMinOccurrenceConstraint(
                property_expression="item['test']",
                min_occurrences=-5,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_min_occurrences_validation_one(self) -> None:
        """Test min_occurrences=1 is valid."""
        constraint = BatchMinOccurrenceConstraint(
            property_expression="item['test']",
            min_occurrences=1,
        )
        assert constraint.min_occurrences == 1

    def test_constraint_type_is_min_occurrence(self) -> None:
        """Test discriminator is correct."""
        constraint = BatchMinOccurrenceConstraint(
            property_expression="item['test']",
            min_occurrences=10,
        )
        assert constraint.constraint_type == "min_occurrence"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        constraint = BatchMinOccurrenceConstraint(
            property_expression="item['quantile']",
            min_occurrences=100,
        )

        data = constraint.model_dump()
        restored = BatchMinOccurrenceConstraint(**data)

        assert restored.property_expression == constraint.property_expression
        assert restored.min_occurrences == constraint.min_occurrences

    def test_inherits_beadbasemodel(self) -> None:
        """Test has BeadBaseModel fields."""
        constraint = BatchMinOccurrenceConstraint(
            property_expression="item['test']",
            min_occurrences=20,
        )

        assert hasattr(constraint, "id")
        assert hasattr(constraint, "created_at")
        assert hasattr(constraint, "modified_at")
