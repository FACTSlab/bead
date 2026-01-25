"""Data models for constructed experimental items."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field, field_validator

from bead.data.base import BeadBaseModel

# Type aliases for JSON-serializable metadata values
type MetadataValue = (
    str | int | float | bool | None | dict[str, MetadataValue] | list[MetadataValue]
)


# Factory functions for default values with explicit types
def _empty_uuid_list() -> list[UUID]:
    """Return empty UUID list."""
    return []


def _empty_unfilled_slot_list() -> list[UnfilledSlot]:
    """Return empty UnfilledSlot list."""
    return []


def _empty_model_output_list() -> list[ModelOutput]:
    """Return empty ModelOutput list."""
    return []


def _empty_item_list() -> list[Item]:
    """Return empty Item list."""
    return []


def _empty_str_dict() -> dict[str, str]:
    """Return empty string-to-string dict."""
    return {}


def _empty_uuid_bool_dict() -> dict[UUID, bool]:
    """Return empty UUID-to-bool dict."""
    return {}


def _empty_metadata_dict() -> dict[str, MetadataValue]:
    """Return empty metadata dict."""
    return {}


def _empty_str_list() -> list[str]:
    """Return empty string list."""
    return []


class UnfilledSlot(BeadBaseModel):
    """An unfilled slot in a cloze task item.

    Represents a slot in a partially filled template where the participant
    must provide a response. The UI widget for collecting the response is
    inferred from the slot's constraints at deployment time.

    Attributes
    ----------
    slot_name : str
        Name of the unfilled template slot.
    position : int
        Token index position in the rendered text.
    constraint_ids : list[UUID]
        UUIDs of constraints that apply to this slot.

    Examples
    --------
    >>> from uuid import UUID
    >>> # Extensional constraint slot (will render as dropdown)
    >>> UnfilledSlot(
    ...     slot_name="determiner",
    ...     position=0,
    ...     constraint_ids=[UUID("12345678-1234-5678-1234-567812345678")]
    ... )
    >>> # Unconstrained slot (will render as text input)
    >>> UnfilledSlot(
    ...     slot_name="adjective",
    ...     position=2,
    ...     constraint_ids=[]
    ... )
    """

    slot_name: str = Field(..., description="Template slot name")
    position: int = Field(..., description="Token position in rendered text")
    constraint_ids: list[UUID] = Field(
        default_factory=_empty_uuid_list, description="Constraint UUIDs for this slot"
    )

    @field_validator("slot_name")
    @classmethod
    def validate_slot_name(cls, v: str) -> str:
        """Validate slot name is not empty.

        Parameters
        ----------
        v : str
            Slot name to validate.

        Returns
        -------
        str
            Validated slot name.

        Raises
        ------
        ValueError
            If slot name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Slot name cannot be empty")
        return v.strip()


class ModelOutput(BeadBaseModel):
    """Output from a model computation.

    Attributes
    ----------
    model_name : str
        Name/identifier of the model.
    model_version : str
        Version of the model.
    operation : str
        Operation performed (e.g., "log_probability", "nli", "embedding").
    inputs : dict[str, MetadataValue]
        Inputs to the model.
    output : MetadataValue
        Model output.
    cache_key : str
        Cache key for this computation.
    computation_metadata : dict[str, MetadataValue]
        Metadata about the computation (timestamp, device, etc.).

    Examples
    --------
    >>> output = ModelOutput(
    ...     model_name="gpt2",
    ...     model_version="latest",
    ...     operation="log_probability",
    ...     inputs={"text": "The cat broke the vase"},
    ...     output=-12.4,
    ...     cache_key="abc123..."
    ... )
    """

    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    operation: str = Field(..., description="Operation type")
    inputs: dict[str, MetadataValue] = Field(..., description="Model inputs")
    output: MetadataValue = Field(..., description="Model output")
    cache_key: str = Field(..., description="Cache key")
    computation_metadata: dict[str, MetadataValue] = Field(
        default_factory=_empty_metadata_dict, description="Computation metadata"
    )

    @field_validator("model_name", "model_version", "operation", "cache_key")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Validate required string fields are not empty.

        Parameters
        ----------
        v : str
            String value to validate.

        Returns
        -------
        str
            Validated string.

        Raises
        ------
        ValueError
            If string is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class Item(BeadBaseModel):
    """A constructed experimental item.

    Items are discrete stimuli presented to participants or models
    for judgment collection. They are constructed from item templates
    and filled templates.

    Attributes
    ----------
    item_template_id : UUID
        UUID of the item template this was constructed from.
    filled_template_refs : list[UUID]
        UUIDs of filled templates used in this item.
    rendered_elements : dict[str, str]
        Rendered text for each element (by element_name).
    options : list[str]
        Choice options for forced_choice/multi_select tasks. Each string
        is one option text. Order matters (first option is displayed first).
    unfilled_slots : list[UnfilledSlot]
        Unfilled slots for cloze tasks (UI widgets inferred from constraints).
    model_outputs : list[ModelOutput]
        All model computations for this item.
    constraint_satisfaction : dict[UUID, bool]
        Constraint UUIDs mapped to satisfaction status.
    item_metadata : dict[str, MetadataValue]
        Additional metadata for this item.

    Examples
    --------
    >>> # Simple item
    >>> item = Item(
    ...     item_template_id=UUID("..."),
    ...     filled_template_refs=[UUID("...")],
    ...     rendered_elements={"sentence": "The cat broke the vase"}
    ... )
    >>> # Forced-choice item with options
    >>> fc_item = Item(
    ...     item_template_id=UUID("..."),
    ...     options=["The cat sat on the mat.", "The cats sat on the mat."],
    ...     item_metadata={"n_options": 2}
    ... )
    >>> # Cloze item with unfilled slots
    >>> cloze_item = Item(
    ...     item_template_id=UUID("..."),
    ...     rendered_elements={"sentence": "The ___ cat ___ the ___"},
    ...     unfilled_slots=[
    ...         UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[...]),
    ...         UnfilledSlot(slot_name="verb", position=2, constraint_ids=[...])
    ...     ]
    ... )
    """

    item_template_id: UUID = Field(..., description="ItemTemplate ID")
    filled_template_refs: list[UUID] = Field(
        default_factory=_empty_uuid_list, description="Filled template UUIDs"
    )
    rendered_elements: dict[str, str] = Field(
        default_factory=_empty_str_dict, description="Rendered element text"
    )
    options: list[str] = Field(
        default_factory=_empty_str_list,
        description="Choice options for forced_choice/multi_select tasks",
    )
    unfilled_slots: list[UnfilledSlot] = Field(
        default_factory=_empty_unfilled_slot_list,
        description="Unfilled slots for cloze tasks",
    )
    model_outputs: list[ModelOutput] = Field(
        default_factory=_empty_model_output_list, description="Model computations"
    )
    constraint_satisfaction: dict[UUID, bool] = Field(
        default_factory=_empty_uuid_bool_dict,
        description="Constraint satisfaction status",
    )
    item_metadata: dict[str, MetadataValue] = Field(
        default_factory=_empty_metadata_dict, description="Additional metadata"
    )

    def get_model_output(
        self,
        model_name: str,
        operation: str,
        inputs: dict[str, MetadataValue] | None = None,
    ) -> ModelOutput | None:
        """Get a specific model output.

        Parameters
        ----------
        model_name : str
            Name of the model.
        operation : str
            Operation type.
        inputs : dict[str, MetadataValue] | None
            Optional input filter.

        Returns
        -------
        ModelOutput | None
            The model output if found, None otherwise.

        Examples
        --------
        >>> output = item.get_model_output("gpt2", "log_probability")
        >>> if output:
        ...     print(f"Log prob: {output.output}")
        """
        for output in self.model_outputs:
            if output.model_name == model_name and output.operation == operation:
                if inputs is None or output.inputs == inputs:
                    return output
        return None

    def add_model_output(self, output: ModelOutput) -> None:
        """Add a model output to this item.

        Parameters
        ----------
        output : ModelOutput
            Model output to add.

        Examples
        --------
        >>> item.add_model_output(my_output)
        >>> print(f"Item now has {len(item.model_outputs)} model outputs")
        """
        self.model_outputs.append(output)
        self.update_modified_time()


class ItemCollection(BeadBaseModel):
    """A collection of constructed items.

    Attributes
    ----------
    name : str
        Name of this collection.
    source_template_collection_id : UUID
        UUID of the source item template collection.
    source_filled_collection_id : UUID
        UUID of the source filled template collection.
    items : list[Item]
        The constructed items.
    construction_stats : dict[str, int]
        Statistics about item construction.

    Examples
    --------
    >>> collection = ItemCollection(
    ...     name="acceptability_items",
    ...     source_template_collection_id=UUID("..."),
    ...     source_filled_collection_id=UUID("...")
    ... )
    >>> collection.add_item(item)
    """

    name: str = Field(..., description="Collection name")
    source_template_collection_id: UUID = Field(
        ..., description="Source template collection UUID"
    )
    source_filled_collection_id: UUID = Field(
        ..., description="Source filled collection UUID"
    )
    items: list[Item] = Field(
        default_factory=_empty_item_list, description="Constructed items"
    )
    construction_stats: dict[str, int] = Field(
        default_factory=dict, description="Construction statistics"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate collection name is not empty.

        Parameters
        ----------
        v : str
            Collection name to validate.

        Returns
        -------
        str
            Validated collection name.

        Raises
        ------
        ValueError
            If name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        return v.strip()

    def add_item(self, item: Item) -> None:
        """Add an item to the collection.

        Parameters
        ----------
        item : Item
            Item to add.

        Examples
        --------
        >>> collection.add_item(my_item)
        >>> print(f"Collection now has {len(collection.items)} items")
        """
        self.items.append(item)
        self.update_modified_time()
