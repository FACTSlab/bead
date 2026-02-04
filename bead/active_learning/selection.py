"""Item selectors for active learning.

This module implements sample selection algorithms that use uncertainty
strategies to intelligently select the most informative items for labeling
in the active learning loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from bead.active_learning.strategies import create_strategy
from bead.items.item import Item

if TYPE_CHECKING:
    from collections.abc import Callable

    from bead.active_learning.models.base import ActiveLearningModel
    from bead.config.active_learning import UncertaintySamplerConfig


class ItemSelector:
    """Base class for item selection algorithms.

    Item selectors determine which unlabeled items should be selected
    for annotation in each active learning iteration.

    Examples
    --------
    >>> selector = ItemSelector()
    >>> # Subclasses implement select() method
    """

    def select(
        self,
        items: list[Item],
        model: ActiveLearningModel,
        predict_fn: Callable[[ActiveLearningModel, Item], np.ndarray],
        budget: int,
    ) -> list[Item]:
        """Select items for annotation.

        Parameters
        ----------
        items : list[Item]
            Unlabeled items to select from.
        model : ActiveLearningModel
            Trained model for making predictions.
        predict_fn : Callable[[ActiveLearningModel, Item], np.ndarray]
            Function to get prediction probabilities from model.
            Should return array of shape (n_classes,) with probabilities.
        budget : int
            Number of items to select.

        Returns
        -------
        list[Item]
            Selected items for annotation.

        Examples
        --------
        >>> selector = UncertaintySampler()  # doctest: +SKIP
        >>> selected = selector.select(  # doctest: +SKIP
        ...     items, model, predict_fn, budget=10
        ... )
        >>> len(selected) <= 10  # doctest: +SKIP
        True
        """
        raise NotImplementedError("Subclasses must implement select()")


class UncertaintySampler(ItemSelector):
    """Uncertainty-based item selector.

    Selects items using uncertainty sampling strategies (entropy, margin,
    or least confidence). This is the main item selection algorithm for
    active learning in bead.

    Parameters
    ----------
    config : UncertaintySamplerConfig | None
        Configuration for the uncertainty sampler.

    Attributes
    ----------
    config : UncertaintySamplerConfig
        Configuration for the sampler.
    strategy : SamplingStrategy
        The underlying sampling strategy.

    Examples
    --------
    >>> import numpy as np
    >>> from uuid import uuid4
    >>> from bead.items.item import Item
    >>> from bead.config.active_learning import UncertaintySamplerConfig
    >>> # Create sampler
    >>> config = UncertaintySamplerConfig(method="entropy")
    >>> sampler = UncertaintySampler(config=config)
    >>> # Mock items
    >>> items = [Item(item_template_id=uuid4(), rendered_elements={}) for _ in range(5)]
    >>> # Mock model and predict function
    >>> def predict_fn(model, item):
    ...     return np.array([0.5, 0.5])  # Mock probabilities
    >>> # Select items
    >>> selected = sampler.select(items, None, predict_fn, budget=2)
    >>> len(selected)
    2
    """

    def __init__(
        self,
        config: UncertaintySamplerConfig | None = None,
    ) -> None:
        """Initialize uncertainty sampler.

        Parameters
        ----------
        config : UncertaintySamplerConfig | None
            Configuration for the sampler. If None, uses defaults.
        """
        self.config = config or UncertaintySamplerConfig()
        self.strategy = create_strategy(self.config.method)

    def select(
        self,
        items: list[Item],
        model: Any,
        predict_fn: Callable[[Any, Item], np.ndarray],
        budget: int,
    ) -> list[Item]:
        """Select items using uncertainty sampling.

        Parameters
        ----------
        items : list[Item]
            Unlabeled items to select from.
        model : Any
            Trained model for making predictions.
        predict_fn : Callable[[Any, Item], np.ndarray]
            Function to get prediction probabilities from model.
            Should return array of shape (n_classes,) for each item.
        budget : int
            Number of items to select.

        Returns
        -------
        list[Item]
            Selected items for annotation, ordered by uncertainty (most to least).

        Raises
        ------
        ValueError
            If items list is empty or budget is invalid.

        Examples
        --------
        >>> import numpy as np
        >>> from uuid import uuid4
        >>> from bead.items.item import Item
        >>> from bead.config.active_learning import UncertaintySamplerConfig
        >>> config = UncertaintySamplerConfig(method="entropy")
        >>> sampler = UncertaintySampler(config=config)
        >>> items = [
        ...     Item(item_template_id=uuid4(), rendered_elements={"text": "item1"}),
        ...     Item(item_template_id=uuid4(), rendered_elements={"text": "item2"}),
        ... ]
        >>> def predict_fn(model, item):
        ...     # First item is uncertain, second is confident
        ...     if "item1" in item.rendered_elements.get("text", ""):
        ...         return np.array([0.5, 0.5])
        ...     return np.array([0.9, 0.1])
        >>> selected = sampler.select(items, None, predict_fn, budget=1)
        >>> "item1" in selected[0].rendered_elements["text"]
        True
        """
        # Validate inputs
        if not items:
            raise ValueError("Items list cannot be empty")

        if budget <= 0:
            raise ValueError(f"Budget must be positive, got {budget}")

        # Handle case where budget >= number of items
        if budget >= len(items):
            return items.copy()

        # Compute predictions for all items
        probabilities = self._batch_predict(items, model, predict_fn)

        # Compute uncertainty scores
        scores = self.strategy.compute_scores(probabilities)

        # Select top k items
        selected_indices = self.strategy.select_top_k(scores, k=budget)

        # Return selected items (convert numpy array to list of Python ints)
        return [items[i] for i in selected_indices.tolist()]

    def _batch_predict(
        self,
        items: list[Item],
        model: Any,
        predict_fn: Callable[[Any, Item], np.ndarray],
    ) -> np.ndarray:
        """Compute predictions in batches.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        model : Any
            Trained model.
        predict_fn : Callable[[Any, Item], np.ndarray]
            Prediction function.

        Returns
        -------
        np.ndarray
            Prediction probabilities with shape (n_items, n_classes).

        Examples
        --------
        >>> import numpy as np
        >>> from uuid import uuid4
        >>> from bead.items.item import Item
        >>> sampler = UncertaintySampler()
        >>> items = [
        ...     Item(item_template_id=uuid4(), rendered_elements={})
        ...     for _ in range(3)
        ... ]
        >>> def predict_fn(model, item):
        ...     return np.array([0.6, 0.4])
        >>> probs = sampler._batch_predict(items, None, predict_fn)
        >>> probs.shape
        (3, 2)
        """
        all_probs = []

        # Process in batches
        batch_size = self.config.batch_size or 32
        for i in range(0, len(items), batch_size):
            batch_items = items[i : i + batch_size]

            # Get predictions for batch
            batch_probs = [predict_fn(model, item) for item in batch_items]

            all_probs.extend(batch_probs)

        # Stack into array
        return np.array(all_probs)


class RandomSelector(ItemSelector):
    """Random item selector (baseline).

    Selects items randomly without considering model predictions.
    Useful as a baseline for comparison with uncertainty-based methods.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.

    Attributes
    ----------
    rng : np.random.Generator
        Random number generator.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.item import Item
    >>> selector = RandomSelector(seed=42)
    >>> items = [
    ...     Item(item_template_id=uuid4(), rendered_elements={})
    ...     for _ in range(10)
    ... ]
    >>> selected = selector.select(items, None, None, budget=3)
    >>> len(selected)
    3
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize random selector.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        items: list[Item],
        model: Any,
        predict_fn: Callable[[Any, Item], np.ndarray],
        budget: int,
    ) -> list[Item]:
        """Select items randomly.

        Parameters
        ----------
        items : list[Item]
            Items to select from.
        model : Any
            Model (unused, kept for interface compatibility).
        predict_fn : Callable[[Any, Item], np.ndarray]
            Prediction function (unused, kept for interface compatibility).
        budget : int
            Number of items to select.

        Returns
        -------
        list[Item]
            Randomly selected items.

        Raises
        ------
        ValueError
            If items list is empty or budget is invalid.

        Examples
        --------
        >>> from uuid import uuid4
        >>> from bead.items.item import Item
        >>> selector = RandomSelector(seed=123)
        >>> items = [
        ...     Item(item_template_id=uuid4(), rendered_elements={})
        ...     for _ in range(5)
        ... ]
        >>> selected = selector.select(items, None, None, budget=2)
        >>> len(selected)
        2
        """
        # Validate inputs
        if not items:
            raise ValueError("Items list cannot be empty")

        if budget <= 0:
            raise ValueError(f"Budget must be positive, got {budget}")

        # Handle case where budget >= number of items
        if budget >= len(items):
            return items.copy()

        # Select random indices without replacement
        selected_indices = self.rng.choice(len(items), size=budget, replace=False)

        # Return selected items (convert numpy array to list of Python ints)
        return [items[i] for i in selected_indices.tolist()]
