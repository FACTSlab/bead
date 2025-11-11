"""Filling strategies for template population."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal, cast
from uuid import UUID

from bead.data.language_codes import LanguageCode, validate_iso639_code
from bead.dsl.evaluator import DSLEvaluator
from bead.items.item import Item
from bead.resources.constraints import ContextValue
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template
from bead.templates.adapters import HuggingFaceMLMAdapter, ModelOutputCache
from bead.templates.combinatorics import cartesian_product
from bead.templates.filler import FilledTemplate, TemplateFiller
from bead.templates.resolver import ConstraintResolver

# Type aliases for strategy configuration
ConfigValue = (
    int
    | str
    | bool
    | None
    | list[int]
    | ConstraintResolver
    | HuggingFaceMLMAdapter
    | ModelOutputCache
    | dict[str, int]
    | dict[str, bool]
)
StrategyConfig = dict[str, ConfigValue]


class FillingStrategy(ABC):
    """Abstract base class for template filling strategies.

    A filling strategy determines how to combine lexical items
    to fill template slots. Strategies differ in:
    - Selection criteria (all vs. sample)
    - Ordering (deterministic vs. random)
    - Grouping (balanced vs. unbalanced)

    Examples
    --------
    >>> strategy = ExhaustiveStrategy()
    >>> combinations = strategy.generate_combinations(slot_items)
    >>> len(list(combinations))
    12
    """

    @abstractmethod
    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate combinations of items for template slots.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to lists of valid items.

        Returns
        -------
        list[dict[str, LexicalItem]]
            List of slot-to-item mappings representing filled templates.

        Examples
        --------
        >>> slot_items = {
        ...     "subject": [item1, item2],
        ...     "verb": [item3, item4],
        ... }
        >>> combinations = strategy.generate_combinations(slot_items)
        >>> len(combinations)
        4
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name for metadata.

        Returns
        -------
        str
            Strategy name.
        """
        pass


class ExhaustiveStrategy(FillingStrategy):
    """Generate all possible combinations of slot fillers.

    This strategy produces the complete Cartesian product of all
    valid items for each slot. Use for small combinatorial spaces.

    **Warning**: Combinatorial explosion! With N slots and M items
    per slot, generates M^N combinations.

    Examples
    --------
    >>> strategy = ExhaustiveStrategy()
    >>> slot_items = {"a": [1, 2], "b": [3, 4]}
    >>> combinations = strategy.generate_combinations(slot_items)
    >>> len(combinations)
    4
    >>> combinations[0]
    {"a": 1, "b": 3}
    """

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "exhaustive"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate all combinations.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.

        Returns
        -------
        list[dict[str, LexicalItem]]
            All possible slot-to-item combinations.
        """
        if not slot_items:
            return []

        # Get ordered slot names and item lists
        slot_names = list(slot_items.keys())
        item_lists = [slot_items[name] for name in slot_names]

        # Generate all combinations
        combinations: list[dict[str, LexicalItem]] = []
        for combo_tuple in cartesian_product(*item_lists):
            combo_dict = dict(zip(slot_names, combo_tuple, strict=True))
            combinations.append(combo_dict)

        return combinations


class RandomStrategy(FillingStrategy):
    """Generate random sample of combinations.

    Sample combinations randomly with optional seeding for
    reproducibility. Use for large combinatorial spaces.

    Parameters
    ----------
    n_samples : int
        Number of combinations to generate.
    seed : int | None
        Random seed for reproducibility. Default: None.

    Examples
    --------
    >>> strategy = RandomStrategy(n_samples=10, seed=42)
    >>> combinations = strategy.generate_combinations(slot_items)
    >>> len(combinations)
    10
    """

    def __init__(self, n_samples: int, seed: int | None = None) -> None:
        """Initialize random strategy.

        Parameters
        ----------
        n_samples : int
            Number of combinations to generate.
        seed : int | None
            Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.seed = seed

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "random"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate random combinations.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.

        Returns
        -------
        list[dict[str, LexicalItem]]
            Randomly sampled combinations.
        """
        if not slot_items:
            return []

        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed)

        # Get ordered slot names and item lists
        slot_names = list(slot_items.keys())
        item_lists = [slot_items[name] for name in slot_names]

        # Generate random combinations
        combinations: list[dict[str, LexicalItem]] = []
        for _ in range(self.n_samples):
            combo_tuple = tuple(random.choice(items) for items in item_lists)
            combo_dict = dict(zip(slot_names, combo_tuple, strict=True))
            combinations.append(combo_dict)

        return combinations


class StratifiedStrategy(FillingStrategy):
    """Generate balanced sample across item groups.

    Ensure each group of items (e.g., by POS, features) is
    represented proportionally in the sample.

    Parameters
    ----------
    n_samples : int
        Total number of combinations to generate.
    grouping_property : str
        Property to group items by (e.g., "pos", "features.transitivity").
    seed : int | None
        Random seed for reproducibility. Default: None.

    Examples
    --------
    >>> strategy = StratifiedStrategy(
    ...     n_samples=20,
    ...     grouping_property="pos",
    ...     seed=42
    ... )
    >>> combinations = strategy.generate_combinations(slot_items)
    >>> # Ensures balanced representation of different POS values
    """

    def __init__(
        self,
        n_samples: int,
        grouping_property: str,
        seed: int | None = None,
    ) -> None:
        """Initialize stratified strategy.

        Parameters
        ----------
        n_samples : int
            Total number of combinations to generate.
        grouping_property : str
            Property to group items by.
        seed : int | None
            Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.grouping_property = grouping_property
        self.seed = seed

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "stratified"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate stratified combinations.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.

        Returns
        -------
        list[dict[str, LexicalItem]]
            Balanced combinations across groups.
        """
        if not slot_items:
            return []

        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed)

        # Group items by the specified property
        grouped_items: dict[str, dict[str, list[LexicalItem]]] = {}
        for slot_name, items in slot_items.items():
            slot_groups: dict[str, list[LexicalItem]] = {}
            for item in items:
                # Get property value (handle nested properties)
                value = self._get_property_value(item, self.grouping_property)
                if value not in slot_groups:
                    slot_groups[value] = []
                slot_groups[value].append(item)
            grouped_items[slot_name] = slot_groups

        # Sample proportionally from each group
        combinations: list[dict[str, LexicalItem]] = []
        slot_names = list(slot_items.keys())

        # Calculate samples per group
        # For simplicity, sample equally from all groups
        for _ in range(self.n_samples):
            combo_dict: dict[str, LexicalItem] = {}
            for slot_name in slot_names:
                slot_groups = grouped_items[slot_name]
                # Choose a random group, then a random item from that group
                if slot_groups:
                    group_key = random.choice(list(slot_groups.keys()))
                    item = random.choice(slot_groups[group_key])
                    combo_dict[slot_name] = item
            combinations.append(combo_dict)

        return combinations

    def _get_property_value(self, item: LexicalItem, property_path: str) -> str:
        """Get property value from item, handling nested properties.

        Parameters
        ----------
        item : LexicalItem
            Item to get property from.
        property_path : str
            Property path (e.g., "pos" or "features.transitivity").

        Returns
        -------
        str
            Property value as string.
        """
        parts = property_path.split(".")
        value = item
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return "unknown"

        # Convert to string for grouping
        if value is None:
            return "none"
        return str(value)


class MLMFillingStrategy(FillingStrategy):
    """Fill templates using masked language models with beam search.

    Uses pre-trained MLMs (BERT, RoBERTa, etc.) to propose linguistically
    plausible slot fillers. Supports beam search for multiple slots with
    configurable fill directions.

    Parameters
    ----------
    resolver : ConstraintResolver
        Constraint resolver for filtering candidates
    model_adapter : HuggingFaceMLMAdapter
        Loaded MLM adapter
    beam_size : int
        Beam search width (K best hypotheses)
    fill_direction : Literal
        Direction for filling slots. One of: "left_to_right", "right_to_left",
        "inside_out", "outside_in", "custom"
    custom_order : list[int] | None
        Custom slot fill order (slot indices)
    top_k : int
        Top-K candidates per slot from MLM
    cache : ModelOutputCache | None
        Cache for model predictions
    budget : int | None
        Maximum combinations to generate

    Examples
    --------
    >>> from bead.templates.adapters import HuggingFaceMLMAdapter, ModelOutputCache
    >>> adapter = HuggingFaceMLMAdapter("bert-base-uncased")
    >>> adapter.load_model()
    >>> cache = ModelOutputCache(Path("/tmp/cache"))
    >>> strategy = MLMFillingStrategy(
    ...     resolver=resolver,
    ...     model_adapter=adapter,
    ...     beam_size=5,
    ...     fill_direction="left_to_right",
    ...     cache=cache
    ... )
    >>> combinations = strategy.generate_combinations(slot_items)
    """

    def __init__(
        self,
        resolver: ConstraintResolver,
        model_adapter: HuggingFaceMLMAdapter,
        beam_size: int = 5,
        fill_direction: Literal[
            "left_to_right", "right_to_left", "inside_out", "outside_in", "custom"
        ] = "left_to_right",
        custom_order: list[int] | None = None,
        top_k: int = 20,
        cache: ModelOutputCache | None = None,
        budget: int | None = None,
        per_slot_max_fills: dict[str, int] | None = None,
        per_slot_enforce_unique: dict[str, bool] | None = None,
    ) -> None:
        """Initialize MLM strategy.

        Parameters
        ----------
        resolver : ConstraintResolver
            Constraint resolver
        model_adapter : HuggingFaceMLMAdapter
            MLM adapter (must be loaded)
        beam_size : int
            Beam width
        fill_direction : str
            Fill direction
        custom_order : list[int] | None
            Custom fill order
        top_k : int
            Top-K from MLM
        cache : ModelOutputCache | None
            Prediction cache
        budget : int | None
            Max combinations
        per_slot_max_fills : dict[str, int] | None
            Maximum number of unique fills per slot (after constraint filtering)
        per_slot_enforce_unique : dict[str, bool] | None
            Whether to enforce uniqueness for each slot across beam hypotheses
        """
        self.resolver = resolver
        self.model_adapter = model_adapter
        self.beam_size = beam_size
        self.fill_direction = fill_direction
        self.custom_order = custom_order
        self.top_k = top_k
        self.cache = cache
        self.budget = budget
        self.per_slot_max_fills = per_slot_max_fills or {}
        self.per_slot_enforce_unique = per_slot_enforce_unique or {}

        if not model_adapter.is_loaded():
            raise ValueError("Model adapter must be loaded before use")

        if fill_direction == "custom" and custom_order is None:
            raise ValueError("custom_order required when fill_direction is 'custom'")

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "mlm"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate combinations using MLM beam search.

        Note: This method adapts slot_items to template-based generation.
        The actual beam search is implemented in generate_from_template.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items (for constraint filtering)

        Returns
        -------
        list[dict[str, LexicalItem]]
            Combinations generated via beam search

        Raises
        ------
        NotImplementedError
            This method requires template context. Use generate_from_template instead.
        """
        raise NotImplementedError(
            "MLMFillingStrategy requires template context. "
            "Use TemplateFiller with MLMFillingStrategy, which calls "
            "generate_from_template internally."
        )

    def generate_from_template(
        self,
        template: Template,
        lexicons: list[Lexicon],
        language_code: LanguageCode | None = None,
    ) -> Iterator[dict[str, LexicalItem]]:
        """Generate combinations from template using beam search.

        Parameters
        ----------
        template : Template
            Template to fill
        lexicons : list[Lexicon]
            Lexicons for constraint resolution
        language_code : LanguageCode | None
            Language filter

        Yields
        ------
        dict[str, LexicalItem]
            Slot-to-item mappings
        """
        # Get slot names and order
        slot_names = list(template.slots.keys())
        if not slot_names:
            return

        fill_order = self._get_fill_order(len(slot_names))

        # Initialize beam with empty hypothesis
        # Each beam item: (filled_slots_dict, cumulative_log_prob)
        beam: list[tuple[dict[str, LexicalItem], float]] = [({}, 0.0)]

        # Track seen items per slot (for uniqueness enforcement)
        seen_items_per_slot: dict[str, set[UUID]] = {
            slot_name: set() for slot_name in slot_names
        }

        # Fill slots in order
        for slot_idx in fill_order:
            slot_name = slot_names[slot_idx]
            slot = template.slots[slot_name]

            new_beam: list[tuple[dict[str, LexicalItem], float]] = []

            # Check if uniqueness is enforced for this slot
            enforce_unique = self.per_slot_enforce_unique.get(slot_name, False)
            max_fills = self.per_slot_max_fills.get(slot_name, None)

            # Expand each hypothesis in current beam
            for filled_slots, cum_log_prob in beam:
                # Get MLM candidates for this slot
                candidates = self._get_mlm_candidates(
                    template,
                    slot_names,
                    slot_idx,
                    filled_slots,
                    slot,
                    lexicons,
                    language_code,
                    seen_items=seen_items_per_slot[slot_name]
                    if enforce_unique
                    else None,
                    max_fills=max_fills,
                )

                # Add each candidate to beam
                for item, log_prob in candidates:
                    new_filled = filled_slots.copy()
                    new_filled[slot_name] = item
                    new_log_prob = cum_log_prob + log_prob
                    new_beam.append((new_filled, new_log_prob))

                    # Track seen items if uniqueness is enforced
                    if enforce_unique:
                        seen_items_per_slot[slot_name].add(item.id)

            # Prune beam to top-K by score (length-normalized)
            if new_beam:
                # Length-normalize scores
                num_filled = len(new_beam[0][0])
                scored_beam = [
                    (filled, log_prob / num_filled, log_prob)
                    for filled, log_prob in new_beam
                ]
                scored_beam.sort(key=lambda x: x[1], reverse=True)

                # Keep top beam_size
                beam = [
                    (filled, cum_log_prob)
                    for filled, _, cum_log_prob in scored_beam[: self.beam_size]
                ]
            else:
                # No valid candidates - empty beam
                beam = []
                break

        # Yield final hypotheses
        count = 0
        for filled_slots, _ in beam:
            if self.budget and count >= self.budget:
                break
            yield filled_slots
            count += 1

    def _get_fill_order(self, num_slots: int) -> list[int]:
        """Get slot fill order based on fill_direction.

        Parameters
        ----------
        num_slots : int
            Number of slots

        Returns
        -------
        list[int]
            Slot indices in fill order
        """
        if self.fill_direction == "custom":
            if self.custom_order is None:
                raise ValueError("custom_order not set")
            return self.custom_order

        indices = list(range(num_slots))

        if self.fill_direction == "left_to_right":
            return indices
        elif self.fill_direction == "right_to_left":
            return list(reversed(indices))
        elif self.fill_direction == "inside_out":
            # Alternate from center outward
            mid = num_slots // 2
            order: list[int] = []
            for i in range(num_slots):
                if i % 2 == 0:
                    order.append(mid + i // 2)
                else:
                    order.append(mid - (i + 1) // 2)
            return [idx for idx in order if 0 <= idx < num_slots]
        elif self.fill_direction == "outside_in":
            # Alternate from edges inward
            order: list[int] = []
            left, right = 0, num_slots - 1
            while left <= right:
                order.append(left)
                if left != right:
                    order.append(right)
                left += 1
                right -= 1
            return order
        else:
            raise ValueError(f"Unknown fill_direction: {self.fill_direction}")

    def _get_mlm_candidates(
        self,
        template: Template,
        slot_names: list[str],
        slot_idx: int,
        filled_slots: dict[str, LexicalItem],
        slot: Slot,
        lexicons: list[Lexicon],
        language_code: LanguageCode | None,
        seen_items: set[UUID] | None = None,
        max_fills: int | None = None,
    ) -> list[tuple[LexicalItem, float]]:
        """Get MLM candidates for a slot.

        Parameters
        ----------
        template : Template
            Template being filled
        slot_names : list[str]
            Ordered slot names
        slot_idx : int
            Index of slot to fill
        filled_slots : dict[str, LexicalItem]
            Already-filled slots
        slot : object
            Slot object
        lexicons : list[Lexicon]
            Lexicons for lookup
        language_code : LanguageCode | None
            Language filter
        seen_items : set | None
            Set of item IDs already used for this slot (for uniqueness enforcement)
        max_fills : int | None
            Maximum number of candidates to return (applied after filtering)

        Returns
        -------
        list[tuple[LexicalItem, float]]
            (item, log_prob) pairs, limited by max_fills and uniqueness
        """
        # Normalize language code to ISO 639-3
        if language_code is not None:
            language_code = validate_iso639_code(language_code)

        # Create masked text
        masked_text = self._create_masked_text(
            template, slot_names, filled_slots, slot_idx
        )

        # Get predictions from MLM (with cache)
        if self.cache:
            predictions = self.cache.get(
                self.model_adapter.model_name,
                masked_text,
                0,  # First mask position
                self.top_k,
            )
        else:
            predictions = None

        if predictions is None:
            predictions = self.model_adapter.predict_masked_token(
                masked_text,
                mask_position=0,
                top_k=self.top_k,
            )
            if self.cache:
                self.cache.set(
                    self.model_adapter.model_name,
                    masked_text,
                    0,
                    self.top_k,
                    predictions,
                )

        # Filter by constraints and find matching lexical items
        candidates: list[tuple[LexicalItem, float]] = []
        for token, log_prob in predictions:
            # Find matching items in lexicons
            for lexicon in lexicons:
                for item in lexicon.items.values():
                    # Skip if already seen (uniqueness enforcement)
                    if seen_items is not None and item.id in seen_items:
                        continue

                    # Match lemma and language
                    if item.lemma.lower() == token.lower():
                        if language_code is None or item.language_code == language_code:
                            # Check slot constraints
                            if slot.constraints:
                                # Evaluate constraints using resolver
                                if self.resolver.evaluate_slot_constraints(
                                    item, slot.constraints
                                ):
                                    candidates.append((item, log_prob))
                            else:
                                candidates.append((item, log_prob))

        # Apply max_fills limit (take top-N by log probability)
        if max_fills is not None and len(candidates) > max_fills:
            # Already sorted by log_prob (descending) from MLM predictions
            # But need to ensure we take highest scoring ones
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:max_fills]

        return candidates

    def _create_masked_text(
        self,
        template: Template,
        slot_names: list[str],
        filled_slots: dict[str, LexicalItem],
        current_slot_idx: int,
    ) -> str:
        """Create text with mask token for current slot.

        Parameters
        ----------
        template : Template
            Template
        slot_names : list[str]
            Slot names
        filled_slots : dict[str, LexicalItem]
            Filled slots
        current_slot_idx : int
            Current slot index

        Returns
        -------
        str
            Text with [MASK] token
        """
        mask_token = self.model_adapter.get_mask_token()
        text = template.template_string

        # Replace filled slots with lemmas
        for slot_name, item in filled_slots.items():
            placeholder = f"{{{slot_name}}}"
            text = text.replace(placeholder, item.lemma)

        # Replace current slot with mask
        current_slot_name = slot_names[current_slot_idx]
        current_placeholder = f"{{{current_slot_name}}}"
        text = text.replace(current_placeholder, mask_token)

        # Replace remaining unfilled slots with mask for context
        for slot_name in slot_names:
            placeholder = f"{{{slot_name}}}"
            if placeholder in text:
                text = text.replace(placeholder, mask_token)

        return text


class StrategyFiller(TemplateFiller):
    """Strategy-based template filling for simple templates.

    Fast filling using enumeration strategies (exhaustive, random, stratified).
    Does NOT handle template-level multi-slot constraints (Template.constraints).

    For templates with multi-slot constraints requiring agreement or
    relational checks, use CSPFiller instead.

    Parameters
    ----------
    lexicon : Lexicon
        Lexicon containing candidate items.
    strategy : FillingStrategy
        Strategy for generating combinations.

    Examples
    --------
    >>> from bead.templates.strategies import StrategyFiller, ExhaustiveStrategy
    >>> filler = StrategyFiller(lexicon, ExhaustiveStrategy())
    >>> filled = filler.fill(template)
    >>> len(filled)
    12
    """

    def __init__(self, lexicon: Lexicon, strategy: FillingStrategy) -> None:
        self.lexicon = lexicon
        self.strategy = strategy
        self.resolver = ConstraintResolver()

    def fill(
        self,
        template: Template,
        language_code: LanguageCode | None = None,
    ) -> list[FilledTemplate]:
        """Fill template with lexical items using strategy.

        Parameters
        ----------
        template : Template
            Template to fill.
        language_code : LanguageCode | None
            Optional language code to filter items.

        Returns
        -------
        list[FilledTemplate]
            List of all filled template instances.

        Raises
        ------
        ValueError
            If any slot has no valid items.
        """
        # 1. Resolve slot constraints
        slot_items = self._resolve_slot_constraints(template, language_code)

        # 2. Check for empty slots
        empty_slots = [name for name, items in slot_items.items() if not items]
        if empty_slots:
            raise ValueError(f"No valid items for slots: {empty_slots}")

        # 3. Generate combinations using strategy
        combinations = self.strategy.generate_combinations(slot_items)

        # 4. Create FilledTemplate instances
        filled_templates: list[FilledTemplate] = []
        for combo in combinations:
            rendered = self._render_template(template, combo)
            filled = FilledTemplate(
                template_id=str(template.id),
                template_name=template.name,
                slot_fillers=combo,
                rendered_text=rendered,
                strategy_name=self.strategy.name,
            )
            filled_templates.append(filled)

        return filled_templates

    def _resolve_slot_constraints(
        self,
        template: Template,
        language_code: LanguageCode | None,
    ) -> dict[str, list[LexicalItem]]:
        """Resolve constraints for each slot.

        Parameters
        ----------
        template : Template
            Template with slots and constraints.
        language_code : LanguageCode | None
            Optional language filter.

        Returns
        -------
        dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.
        """
        slot_items: dict[str, list[LexicalItem]] = {}

        # Normalize language code if provided
        normalized_lang = validate_iso639_code(language_code) if language_code else None

        for slot_name, slot in template.slots.items():
            candidates = list(self.lexicon.items.values())

            # Filter by language code
            if normalized_lang:
                candidates = [
                    item for item in candidates if item.language_code == normalized_lang
                ]

            # Apply slot constraints
            if slot.constraints:
                for constraint in slot.constraints:
                    filtered: list[LexicalItem] = []
                    for item in candidates:
                        eval_context: dict[
                            str, ContextValue | LexicalItem | FilledTemplate | Item
                        ] = {"self": item}
                        if constraint.context:
                            eval_context.update(constraint.context)

                        evaluator = DSLEvaluator()
                        try:
                            if evaluator.evaluate(constraint.expression, eval_context):
                                filtered.append(item)
                        except Exception:
                            continue
                    candidates = filtered

            slot_items[slot_name] = candidates

        return slot_items

    def _render_template(
        self, template: Template, slot_fillers: dict[str, LexicalItem]
    ) -> str:
        """Render template string with slot fillers.

        Parameters
        ----------
        template : Template
            Template with template_string.
        slot_fillers : dict[str, LexicalItem]
            Items filling each slot.

        Returns
        -------
        str
            Rendered template string.
        """
        rendered = template.template_string
        for slot_name, item in slot_fillers.items():
            placeholder = f"{{{slot_name}}}"
            rendered = rendered.replace(placeholder, item.lemma)
        return rendered

    def count_combinations(self, template: Template) -> int:
        """Count total possible combinations for template.

        Parameters
        ----------
        template : Template
            Template to count combinations for.

        Returns
        -------
        int
            Total number of possible combinations.
        """
        slot_items = self._resolve_slot_constraints(template, None)

        if not slot_items:
            return 0

        count = 1
        for items in slot_items.values():
            count *= len(items)

        return count


class MixedFillingStrategy(FillingStrategy):
    """Fill different template slots using different strategies.

    Allows per-slot strategy specification, enabling workflows like:
    - Fill nouns/verbs exhaustively
    - Fill adjectives via MLM based on noun context

    This strategy operates in phases:
    1. Phase 1: Fill slots assigned to non-MLM strategies (exhaustive, random, etc.)
    2. Phase 2: For each Phase 1 combination, fill remaining slots via MLM

    Parameters
    ----------
    slot_strategies : dict[str, tuple[FillingStrategy, dict]]
        Mapping of slot names to (strategy, config) tuples.
        Config is strategy-specific kwargs.
    default_strategy : FillingStrategy | None
        Default strategy for slots not explicitly specified.

    Examples
    --------
    >>> exhaustive = ExhaustiveStrategy()
    >>> mlm_config = {
    ...     "resolver": resolver,
    ...     "model_adapter": mlm_adapter,
    ...     "top_k": 5
    ... }
    >>> strategy = MixedFillingStrategy(
    ...     slot_strategies={
    ...         "noun": (exhaustive, {}),
    ...         "verb": (exhaustive, {}),
    ...         "adjective": ("mlm", mlm_config)
    ...     }
    ... )
    """

    def __init__(
        self,
        slot_strategies: dict[str, tuple[str | FillingStrategy, StrategyConfig]],
        default_strategy: FillingStrategy | None = None,
    ) -> None:
        """Initialize mixed strategy.

        Parameters
        ----------
        slot_strategies : dict[str, tuple[str | FillingStrategy, StrategyConfig]]
            Mapping slot names to (strategy_name, config) or
            (strategy_instance, config). strategy_name can be:
            "exhaustive", "random", "stratified", "mlm"
        default_strategy : FillingStrategy | None
            Default strategy for unspecified slots.
        """
        self.slot_strategies = slot_strategies
        self.default_strategy = default_strategy or ExhaustiveStrategy()

        # Separate slots by phase
        self.phase1_slots: list[str] = []  # Non-MLM slots
        self.phase2_slots: list[str] = []  # MLM slots
        self.phase1_strategies: dict[str, FillingStrategy] = {}
        self.mlm_configs: dict[str, StrategyConfig] = {}

        for slot_name, (strategy, config) in slot_strategies.items():
            strategy_name = strategy if isinstance(strategy, str) else strategy.name

            if strategy_name == "mlm":
                self.phase2_slots.append(slot_name)
                self.mlm_configs[slot_name] = config
            else:
                self.phase1_slots.append(slot_name)
                # Instantiate strategy if needed
                if isinstance(strategy, str):
                    self.phase1_strategies[slot_name] = self._instantiate_strategy(
                        strategy, config
                    )
                else:
                    self.phase1_strategies[slot_name] = strategy

    def _instantiate_strategy(
        self, strategy_name: str, config: StrategyConfig
    ) -> FillingStrategy:
        """Instantiate strategy from name and config.

        Parameters
        ----------
        strategy_name : str
            Strategy name: "exhaustive", "random", "stratified"
        config : dict
            Strategy-specific configuration

        Returns
        -------
        FillingStrategy
            Instantiated strategy

        Raises
        ------
        ValueError
            If strategy name is unknown
        """
        if strategy_name == "exhaustive":
            return ExhaustiveStrategy()
        elif strategy_name == "random":
            return RandomStrategy(
                n_samples=cast(int, config.get("n_samples", 100)),
                seed=cast(int | None, config.get("seed")),
            )
        elif strategy_name == "stratified":
            return StratifiedStrategy(
                n_samples=cast(int, config.get("n_samples", 100)),
                grouping_property=cast(str, config.get("grouping_property", "pos")),
                seed=cast(int | None, config.get("seed")),
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "mixed"

    def generate_combinations(
        self,
        slot_items: dict[str, list[LexicalItem]],
    ) -> list[dict[str, LexicalItem]]:
        """Generate combinations using mixed strategies.

        Note: This method signature is required by FillingStrategy,
        but MixedFillingStrategy with MLM requires template context.
        Use generate_from_template instead.

        Parameters
        ----------
        slot_items : dict[str, list[LexicalItem]]
            Mapping of slot names to valid items

        Returns
        -------
        list[dict[str, LexicalItem]]
            Generated combinations

        Raises
        ------
        NotImplementedError
            If any slot uses MLM strategy (requires template context)
        """
        if self.phase2_slots:
            raise NotImplementedError(
                "MixedFillingStrategy with MLM slots requires template context. "
                "Use StrategyFiller or CSPFiller, which call generate_from_template."
            )

        # If no MLM slots, just use phase 1 strategies
        # This is a simplified case: all slots use non-MLM strategies

        # For each slot, generate its combinations independently
        slot_combinations: dict[str, list[LexicalItem]] = {}

        for slot_name, items in slot_items.items():
            if slot_name in self.phase1_strategies:
                strategy = self.phase1_strategies[slot_name]
                # Generate combinations for just this slot
                combos = strategy.generate_combinations({slot_name: items})
                slot_combinations[slot_name] = [c[slot_name] for c in combos]
            else:
                # Use default strategy
                combos = self.default_strategy.generate_combinations({slot_name: items})
                slot_combinations[slot_name] = [c[slot_name] for c in combos]

        # Generate cartesian product of all slot combinations
        slot_names = list(slot_items.keys())
        item_lists = [slot_combinations[name] for name in slot_names]

        combinations: list[dict[str, LexicalItem]] = []
        for combo_tuple in cartesian_product(*item_lists):
            combo_dict = dict(zip(slot_names, combo_tuple, strict=True))
            combinations.append(combo_dict)

        return combinations

    def generate_from_template(
        self,
        template: Template,
        lexicons: list[Lexicon],
        language_code: LanguageCode | None = None,
    ) -> Iterator[dict[str, LexicalItem]]:
        """Generate combinations from template using mixed strategies.

        Phase 1: Fill non-MLM slots using their assigned strategies
        Phase 2: For each Phase 1 combination, fill MLM slots using beam search

        Parameters
        ----------
        template : Template
            Template to fill
        lexicons : list[Lexicon]
            Lexicons for constraint resolution
        language_code : LanguageCode | None
            Language filter

        Yields
        ------
        dict[str, LexicalItem]
            Complete slot-to-item mappings
        """
        # Phase 1: Fill non-MLM slots
        if not self.phase1_slots:
            # No phase 1 slots - just use MLM for all phase 2 slots
            phase1_combinations: list[dict[str, LexicalItem]] = [{}]
        else:
            phase1_combinations = self._generate_phase1_combinations(
                template, lexicons, language_code
            )

        # Phase 2: Fill MLM slots for each phase 1 combination
        if not self.phase2_slots:
            # No phase 2 slots - just yield phase 1 combinations
            yield from phase1_combinations
        else:
            for partial_combo in phase1_combinations:
                # Fill remaining slots with MLM
                yield from self._fill_mlm_slots(
                    template, partial_combo, lexicons, language_code
                )

    def _generate_phase1_combinations(
        self,
        template: Template,
        lexicons: list[Lexicon],
        language_code: LanguageCode | None,
    ) -> list[dict[str, LexicalItem]]:
        """Generate combinations for phase 1 (non-MLM) slots.

        Parameters
        ----------
        template : Template
            Template being filled
        lexicons : list[Lexicon]
            Lexicons for items
        language_code : LanguageCode | None
            Language filter

        Returns
        -------
        list[dict[str, LexicalItem]]
            Partial combinations (only phase 1 slots filled)
        """
        # Get valid items for each phase 1 slot
        slot_items: dict[str, list[LexicalItem]] = {}
        normalized_lang = validate_iso639_code(language_code) if language_code else None

        for slot_name in self.phase1_slots:
            if slot_name not in template.slots:
                continue

            slot = template.slots[slot_name]
            candidates: list[LexicalItem] = []

            # Collect items from all lexicons
            for lexicon in lexicons:
                for item in lexicon.items.values():
                    # Filter by language
                    if normalized_lang and item.language_code != normalized_lang:
                        continue
                    # Check slot constraints
                    if slot.constraints:
                        eval_context: dict[str, ContextValue | LexicalItem] = {
                            "self": item
                        }
                        # Add constraint context
                        for constraint in slot.constraints:
                            if constraint.context:
                                eval_context.update(constraint.context)
                            # Evaluate
                            evaluator = DSLEvaluator()
                            try:
                                # Cast to expected context type
                                typed_context = cast(
                                    dict[
                                        str,
                                        ContextValue
                                        | LexicalItem
                                        | FilledTemplate
                                        | Item,
                                    ],
                                    eval_context,
                                )
                                if not evaluator.evaluate(
                                    constraint.expression, typed_context
                                ):
                                    continue
                            except Exception:
                                continue
                    candidates.append(item)

            slot_items[slot_name] = candidates

        # Generate combinations using per-slot strategies
        # For each slot, we need to apply its strategy independently,
        # then take cartesian product

        # Collect combinations per slot
        slot_combos: dict[str, list[LexicalItem]] = {}

        for slot_name in self.phase1_slots:
            if slot_name not in slot_items:
                continue

            items = slot_items[slot_name]
            strategy = self.phase1_strategies.get(slot_name, self.default_strategy)

            # Generate combinations for this slot
            combos = strategy.generate_combinations({slot_name: items})
            slot_combos[slot_name] = [c[slot_name] for c in combos]

        # Cartesian product of all phase 1 slots
        if not slot_combos:
            return [{}]

        slot_names = list(slot_combos.keys())
        item_lists = [slot_combos[name] for name in slot_names]

        combinations: list[dict[str, LexicalItem]] = []
        for combo_tuple in cartesian_product(*item_lists):
            combo_dict = dict(zip(slot_names, combo_tuple, strict=True))
            combinations.append(combo_dict)

        return combinations

    def _fill_mlm_slots(
        self,
        template: Template,
        partial_filling: dict[str, LexicalItem],
        lexicons: list[Lexicon],
        language_code: LanguageCode | None,
    ) -> Iterator[dict[str, LexicalItem]]:
        """Fill MLM slots given a partial filling from phase 1.

        Parameters
        ----------
        template : Template
            Template being filled
        partial_filling : dict[str, LexicalItem]
            Already-filled slots from phase 1
        lexicons : list[Lexicon]
            Lexicons for items
        language_code : LanguageCode | None
            Language filter

        Yields
        ------
        dict[str, LexicalItem]
            Complete fillings with MLM slots added
        """
        if not self.phase2_slots or not self.mlm_configs:
            yield partial_filling
            return

        # Get base config from first MLM slot (model adapter, resolver, etc.)
        first_mlm_slot = self.phase2_slots[0]
        base_config = self.mlm_configs[first_mlm_slot].copy()

        # Extract per-slot max_fills and enforce_unique settings
        per_slot_max_fills: dict[str, int] = {}
        per_slot_enforce_unique: dict[str, bool] = {}

        for slot_name in self.phase2_slots:
            config = self.mlm_configs[slot_name]
            if "max_fills" in config:
                per_slot_max_fills[slot_name] = cast(int, config["max_fills"])
            if "enforce_unique" in config:
                per_slot_enforce_unique[slot_name] = cast(
                    bool, config["enforce_unique"]
                )

        # Remove per-slot settings from base config
        # (they're not MLMFillingStrategy params)
        base_config.pop("max_fills", None)
        base_config.pop("enforce_unique", None)

        # Add per-slot dicts to config
        base_config["per_slot_max_fills"] = per_slot_max_fills
        base_config["per_slot_enforce_unique"] = per_slot_enforce_unique

        # Create MLM strategy with properly typed config
        mlm_strategy = MLMFillingStrategy(
            resolver=cast(ConstraintResolver, base_config["resolver"]),
            model_adapter=cast(HuggingFaceMLMAdapter, base_config["model_adapter"]),
            beam_size=cast(int, base_config.get("beam_size", 5)),
            fill_direction=cast(
                Literal[
                    "left_to_right",
                    "right_to_left",
                    "inside_out",
                    "outside_in",
                    "custom",
                ],
                base_config.get("fill_direction", "left_to_right"),
            ),
            custom_order=cast(list[int] | None, base_config.get("custom_order")),
            top_k=cast(int, base_config.get("top_k", 20)),
            cache=cast(ModelOutputCache | None, base_config.get("cache")),
            budget=cast(int | None, base_config.get("budget")),
            per_slot_max_fills=per_slot_max_fills,
            per_slot_enforce_unique=per_slot_enforce_unique,
        )

        # Create a modified template with only MLM slots
        mlm_template = self._create_mlm_template(template, partial_filling)

        # Generate completions via MLM
        for mlm_filling in mlm_strategy.generate_from_template(
            mlm_template, lexicons, language_code
        ):
            # Combine partial + MLM fillings
            complete = partial_filling.copy()
            complete.update(mlm_filling)
            yield complete

    def _create_mlm_template(
        self, template: Template, partial_filling: dict[str, LexicalItem]
    ) -> Template:
        """Create template with phase1 slots already filled.

        Parameters
        ----------
        template : Template
            Original template
        partial_filling : dict[str, LexicalItem]
            Items filling phase 1 slots

        Returns
        -------
        Template
            Modified template with phase1 slots replaced by text
        """
        # Replace phase 1 slots in template string with their fillings
        modified_string = template.template_string
        for slot_name, item in partial_filling.items():
            placeholder = f"{{{slot_name}}}"
            modified_string = modified_string.replace(placeholder, item.lemma)

        # Create new template with only phase 2 (MLM) slots
        mlm_slots = {
            name: slot
            for name, slot in template.slots.items()
            if name in self.phase2_slots
        }

        # Create modified template
        modified_template = Template(
            name=f"{template.name}_mlm",
            template_string=modified_string,
            slots=mlm_slots,
            constraints=template.constraints,
            language_code=template.language_code,
        )

        return modified_template
