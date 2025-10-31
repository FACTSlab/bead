"""Shared fixtures for evaluation tests."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from bead.items.models import Item


@pytest.fixture
def sample_items():
    """Create 100 sample items for testing.

    Returns
    -------
    list[Item]
        100 items with 3 classes (labels 0, 1, 2).
    """
    return [
        Item(
            item_template_id=uuid4(),
            rendered_elements={},
            item_metadata={"label": i % 3},
        )
        for i in range(100)
    ]


@pytest.fixture
def binary_predictions():
    """Binary classification predictions.

    Returns
    -------
    dict
        Dictionary with y_true and y_pred lists.
    """
    return {
        "y_true": [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
        "y_pred": [0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
    }


@pytest.fixture
def multi_class_predictions():
    """Multi-class classification predictions.

    Returns
    -------
    dict
        Dictionary with y_true and y_pred lists (3 classes).
    """
    return {
        "y_true": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        "y_pred": [0, 1, 2, 0, 1, 1, 0, 2, 2, 0],
    }


@pytest.fixture
def perfect_predictions():
    """Perfect predictions (100% accuracy).

    Returns
    -------
    dict
        Dictionary with identical y_true and y_pred lists.
    """
    return {
        "y_true": [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
        "y_pred": [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    }


@pytest.fixture
def multi_rater_data():
    """Multiple rater ratings for inter-annotator tests.

    Returns
    -------
    dict
        Dictionary with ratings from 3 raters.
    """
    return {
        "rater1": [1, 2, 3, 1, 2, 3, 1, 2],
        "rater2": [1, 2, 3, 1, 2, 2, 1, 2],
        "rater3": [1, 2, 3, 2, 2, 3, 1, 3],
    }


@pytest.fixture
def perfect_agreement_data():
    """Perfect agreement between two raters.

    Returns
    -------
    dict
        Dictionary with identical ratings from 2 raters.
    """
    return {
        "rater1": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        "rater2": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    }


@pytest.fixture
def no_agreement_data():
    """No agreement between two raters (random ratings).

    Returns
    -------
    dict
        Dictionary with completely different ratings from 2 raters.
    """
    np.random.seed(42)
    return {
        "rater1": np.random.randint(1, 4, size=100).tolist(),
        "rater2": np.random.randint(1, 4, size=100).tolist(),
    }


@pytest.fixture
def ratings_with_missing():
    """Ratings with missing data (None values).

    Returns
    -------
    dict
        Dictionary with ratings containing None values.
    """
    return {
        "rater1": [1, 2, None, 1, 2, 3, None, 2],
        "rater2": [1, 2, 3, None, 2, 2, 1, None],
        "rater3": [1, None, 3, 2, 2, 3, 1, 3],
    }
