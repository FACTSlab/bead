"""Shared pytest fixtures for list model tests."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from sash.lists.constraints import (
    BalanceConstraint,
    QuantileConstraint,
    SizeConstraint,
    UniquenessConstraint,
)
from sash.lists.models import ExperimentList, ListCollection


@pytest.fixture
def sample_uuid() -> UUID:
    """Provide a fixed UUID for testing.

    Returns
    -------
    UUID
        Sample UUID.
    """
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_item_uuids() -> list[UUID]:
    """Generate list of 100 UUIDs for testing.

    Returns
    -------
    list[UUID]
        List of 100 UUIDs.
    """
    return [uuid4() for _ in range(100)]


@pytest.fixture
def empty_experiment_list() -> ExperimentList:
    """Create an empty experiment list.

    Returns
    -------
    ExperimentList
        Empty list with default values.
    """
    return ExperimentList(name="empty_list", list_number=0)


@pytest.fixture
def experiment_list_with_items(sample_item_uuids: list[UUID]) -> ExperimentList:
    """Create an experiment list with 20 items.

    Parameters
    ----------
    sample_item_uuids : list[UUID]
        Pool of UUIDs to draw from.

    Returns
    -------
    ExperimentList
        List with 20 items.
    """
    exp_list = ExperimentList(name="list_with_items", list_number=1)
    for item_id in sample_item_uuids[:20]:
        exp_list.add_item(item_id)
    return exp_list


@pytest.fixture
def uniqueness_constraint() -> UniquenessConstraint:
    """Create a uniqueness constraint.

    Returns
    -------
    UniquenessConstraint
        Constraint requiring unique target verbs.
    """
    return UniquenessConstraint(
        property_path="item_metadata.target_verb", allow_null=False
    )


@pytest.fixture
def balance_constraint() -> BalanceConstraint:
    """Create a balance constraint.

    Returns
    -------
    BalanceConstraint
        Constraint for balanced transitivity.
    """
    return BalanceConstraint(property_path="item_metadata.transitivity", tolerance=0.1)


@pytest.fixture
def quantile_constraint() -> QuantileConstraint:
    """Create a quantile constraint.

    Returns
    -------
    QuantileConstraint
        Constraint for LM probability quantiles.
    """
    return QuantileConstraint(
        property_path="item_metadata.lm_prob",
        n_quantiles=5,
        items_per_quantile=2,
    )


@pytest.fixture
def size_constraint_exact() -> SizeConstraint:
    """Create a size constraint with exact size.

    Returns
    -------
    SizeConstraint
        Constraint requiring exactly 40 items.
    """
    return SizeConstraint(exact_size=40)


@pytest.fixture
def size_constraint_range() -> SizeConstraint:
    """Create a size constraint with range.

    Returns
    -------
    SizeConstraint
        Constraint requiring 30-50 items.
    """
    return SizeConstraint(min_size=30, max_size=50)


@pytest.fixture
def experiment_list_with_constraints(
    uniqueness_constraint: UniquenessConstraint,
    balance_constraint: BalanceConstraint,
) -> ExperimentList:
    """Create an experiment list with constraints.

    Parameters
    ----------
    uniqueness_constraint : UniquenessConstraint
        Uniqueness constraint to add.
    balance_constraint : BalanceConstraint
        Balance constraint to add.

    Returns
    -------
    ExperimentList
        List with constraints.
    """
    return ExperimentList(
        name="list_with_constraints",
        list_number=2,
        list_constraints=[uniqueness_constraint, balance_constraint],
    )


@pytest.fixture
def sample_list_collection(
    sample_uuid: UUID, experiment_list_with_items: ExperimentList
) -> ListCollection:
    """Create a list collection with one list.

    Parameters
    ----------
    sample_uuid : UUID
        UUID for source items.
    experiment_list_with_items : ExperimentList
        List to add to collection.

    Returns
    -------
    ListCollection
        Collection with one list.
    """
    collection = ListCollection(
        name="sample_collection",
        source_items_id=sample_uuid,
        partitioning_strategy="balanced",
        partitioning_config={"n_lists": 1, "seed": 42},
    )
    collection.add_list(experiment_list_with_items)
    return collection
