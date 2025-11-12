"""Cloze (fill-in-the-blank) simulation strategy using MLM scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.item import Item
    from bead.items.item_template import ItemTemplate

__all__ = ["ClozeStrategy"]


class ClozeStrategy(SimulationStrategy):
    """MLM-based strategy for cloze (fill-in-the-blank) tasks.

    Uses masked language model scores to select fillers for unfilled slots.
    For constrained slots (with specific options), selects highest-scoring option.
    For unconstrained slots, uses rendered_elements or metadata as fallback.

    The strategy expects model outputs to contain MLM scores for each slot,
    stored as separate ModelOutput instances with operation="mlm_score" and
    inputs containing {"slot_name": slot_name, "candidate": candidate_value}.

    Examples
    --------
    >>> from bead.simulation.strategies.cloze import ClozeStrategy
    >>> strategy = ClozeStrategy()
    >>> # item with unfilled_slots and model_outputs with MLM scores
    >>> # response = strategy.simulate_response(item, template, "mlm_score", rng)
    >>> # Returns: {"determiner": "the", "verb": "chased", "object": "mouse"}
    """

    @property
    def supported_task_type(self) -> str:
        """Get supported task type.

        Returns
        -------
        str
            Always returns "cloze".
        """
        return "cloze"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item is compatible with cloze strategy.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template defining task.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if item_template.task_type != "cloze":
            msg = f"Expected task_type 'cloze', got '{item_template.task_type}'"
            raise ValueError(msg)

        if not item.unfilled_slots:
            raise ValueError("cloze task requires at least one unfilled slot")

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> dict[str, str]:
        """Simulate cloze response using MLM scores.

        For each unfilled slot, selects the filler with highest MLM score.
        Falls back to random selection or metadata if MLM scores unavailable.

        Parameters
        ----------
        item : Item
            Item to annotate.
        item_template : ItemTemplate
            Template defining task constraints.
        model_output_key : str
            Key identifying which model outputs to use (e.g., "mlm_score").
        rng : np.random.RandomState
            Random number generator for stochasticity.

        Returns
        -------
        dict[str, str]
            Dictionary mapping slot names to selected fillers.

        Examples
        --------
        >>> response = {"determiner": "the", "verb": "chased", "object": "mouse"}
        """
        response = {}

        for slot in item.unfilled_slots:
            slot_name = slot.slot_name

            # Try to get MLM scores for this slot
            slot_scores = self._get_slot_scores(item, slot_name, model_output_key)

            if slot_scores:
                # Select filler with highest score (with softmax sampling)
                fillers = list(slot_scores.keys())
                scores = np.array(list(slot_scores.values()))

                # Apply softmax to convert scores to probabilities
                exp_scores = np.exp(scores - np.max(scores))  # numerical stability
                probs = exp_scores / np.sum(exp_scores)

                # Sample from distribution
                selected_idx = rng.choice(len(fillers), p=probs)
                response[slot_name] = fillers[selected_idx]
            else:
                # Fallback: use ground truth if available, else placeholder
                response[slot_name] = self._get_fallback_filler(item, slot_name, rng)

        return response

    def _get_slot_scores(
        self, item: Item, slot_name: str, model_output_key: str
    ) -> dict[str, float]:
        """Extract MLM scores for a specific slot.

        Looks for ModelOutput instances where:
        - operation matches model_output_key (e.g., "mlm_score")
        - inputs contains {"slot_name": slot_name}
        - inputs contains "candidate" (the filler being scored)
        - output is the MLM score

        Parameters
        ----------
        item : Item
            Item containing model outputs.
        slot_name : str
            Name of the slot to get scores for.
        model_output_key : str
            Operation type to filter by.

        Returns
        -------
        dict[str, float]
            Mapping from candidate fillers to MLM scores.
        """
        scores = {}

        for model_output in item.model_outputs:
            if model_output.operation != model_output_key:
                continue

            inputs = model_output.inputs

            # Check if this output is for our slot
            if inputs.get("slot_name") != slot_name:
                continue

            candidate = inputs.get("candidate")
            if candidate is None:
                continue

            # Extract score
            score = model_output.output
            if isinstance(score, (int, float)):
                scores[str(candidate)] = float(score)

        return scores

    def _get_fallback_filler(
        self, item: Item, slot_name: str, rng: np.random.RandomState
    ) -> str:
        """Get fallback filler when MLM scores unavailable.

        Priority:
        1. Ground truth from item_metadata["ground_truth"][slot_name]
        2. Random common filler based on slot name pattern
        3. Generic placeholder

        Parameters
        ----------
        item : Item
            Item to get fallback from.
        slot_name : str
            Slot name.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        str
            Fallback filler.
        """
        # Try ground truth
        if hasattr(item, "item_metadata") and item.item_metadata:
            ground_truth = item.item_metadata.get("ground_truth")
            if isinstance(ground_truth, dict) and slot_name in ground_truth:
                return str(ground_truth[slot_name])

        # Common fallbacks by slot name patterns
        fallback_options = {
            "determiner": ["the", "a", "an", "this", "that"],
            "verb": ["is", "was", "has", "can", "will"],
            "noun": ["thing", "person", "place", "time", "way"],
            "adjective": ["good", "new", "old", "big", "small"],
            "adverb": ["very", "well", "just", "now", "here"],
            "preposition": ["in", "on", "at", "to", "for"],
        }

        # Match slot name to category
        slot_lower = slot_name.lower()
        for category, options in fallback_options.items():
            if category in slot_lower:
                return str(rng.choice(options))

        # Generic fallback
        return f"[{slot_name}]"
