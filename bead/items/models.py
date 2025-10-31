"""Data models for experimental items and item templates."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, ValidationInfo, field_validator

from bead.data.base import BeadBaseModel

# Type aliases for JSON-serializable metadata values
type MetadataValue = (
    str | int | float | bool | None | dict[str, MetadataValue] | list[MetadataValue]
)


# Factory functions for default values with explicit types
def _empty_uuid_list() -> list[UUID]:
    """Return empty UUID list."""
    return []


def _empty_item_element_list() -> list[ItemElement]:
    """Return empty ItemElement list."""
    return []


def _empty_unfilled_slot_list() -> list[UnfilledSlot]:
    """Return empty UnfilledSlot list."""
    return []


def _empty_model_output_list() -> list[ModelOutput]:
    """Return empty ModelOutput list."""
    return []


def _empty_item_template_list() -> list[ItemTemplate]:
    """Return empty ItemTemplate list."""
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


def _empty_display_format_dict() -> dict[str, str | int | float | bool]:
    """Return empty display format dict."""
    return {}


# Type aliases for judgment and task types
JudgmentType = Literal[
    "acceptability",  # Linguistic acceptability/grammaticality/naturalness
    "inference",  # Semantic relationship (NLI: entailment/neutral/contradiction)
    "similarity",  # Semantic similarity/distance/relatedness
    "plausibility",  # Likelihood/plausibility of events or statements
    "comprehension",  # Understanding/recall of content
    "preference",  # Subjective preference between alternatives
]

TaskType = Literal[
    "forced_choice",  # Pick exactly one option (UI: radio buttons)
    "multi_select",  # Pick one or more options (UI: checkboxes)
    "ordinal_scale",  # Value on ordered discrete scale (UI: Likert, slider)
    "magnitude",  # Unbounded numeric value (UI: number input)
    "binary",  # Yes/no, true/false (UI: toggle, buttons)
    "categorical",  # Pick from unordered categories (UI: dropdown, radio)
    "free_text",  # Open-ended text (UI: text input, textarea)
    "cloze",  # Fill-in-the-blank with unfilled slots (UI: inferred)
]

ElementRefType = Literal["text", "filled_template_ref"]

PresentationMode = Literal["static", "self_paced", "timed_sequence"]

ChunkingUnit = Literal[
    "character",
    "word",
    "sentence",
    "constituent",
    "custom",
]

ParseType = Literal["constituency", "dependency"]


class ChunkingSpec(BeadBaseModel):
    """Specification for text segmentation in incremental presentation.

    Defines how to segment text for self-paced reading or timed sequence
    presentation. Supports character-level, word-level, sentence-level,
    constituent-based (with parsing), or custom boundary segmentation.

    Attributes
    ----------
    unit : ChunkingUnit
        Segmentation unit type.
    parse_type : ParseType | None
        Type of parsing for constituent chunking ("constituency" or "dependency").
    constituent_labels : list[str] | None
        Labels for constituent chunking. For constituency parsing, these are
        constituent types (e.g., ["NP", "VP", "S"]). For dependency parsing,
        these are dependency relations (e.g., ["nsubj", "dobj", "root"]).
    parser : Literal["stanza", "spacy"] | None
        Parser library to use for constituent chunking.
    parse_language : str | None
        ISO 639 language code for parser (e.g., "en", "es", "zh").
    custom_boundaries : list[int] | None
        Token indices for custom chunking boundaries.

    Examples
    --------
    >>> # Word-by-word chunking
    >>> ChunkingSpec(unit="word")
    >>> # Chunk by noun phrases (constituency)
    >>> ChunkingSpec(
    ...     unit="constituent",
    ...     parse_type="constituency",
    ...     constituent_labels=["NP"],
    ...     parser="stanza",
    ...     parse_language="en"
    ... )
    >>> # Chunk by subjects and objects (dependency)
    >>> ChunkingSpec(
    ...     unit="constituent",
    ...     parse_type="dependency",
    ...     constituent_labels=["nsubj", "dobj"],
    ...     parser="spacy",
    ...     parse_language="en"
    ... )
    >>> # Custom boundaries at specific token positions
    >>> ChunkingSpec(unit="custom", custom_boundaries=[0, 3, 7, 10])
    """

    unit: ChunkingUnit = Field(..., description="Segmentation unit type")
    parse_type: ParseType | None = Field(
        default=None, description="Parsing type for constituent chunking"
    )
    constituent_labels: list[str] | None = Field(
        default=None,
        description="Constituent or dependency labels for chunking",
    )
    parser: Literal["stanza", "spacy"] | None = Field(
        default=None, description="Parser library"
    )
    parse_language: str | None = Field(
        default=None, description="ISO 639 language code"
    )
    custom_boundaries: list[int] | None = Field(
        default=None, description="Custom token boundary indices"
    )


class TimingParams(BeadBaseModel):
    """Timing parameters for stimulus presentation.

    Defines timing constraints for timed sequence presentations,
    including per-chunk duration, inter-stimulus intervals, and
    response timeouts.

    Attributes
    ----------
    duration_ms : int | None
        Duration in milliseconds to display each chunk (for timed sequences).
    isi_ms : int | None
        Inter-stimulus interval in milliseconds between chunks.
    timeout_ms : int | None
        Maximum time in milliseconds to wait for response.
    mask_char : str | None
        Character to use for masking non-current chunks (e.g., "_").
    cumulative : bool
        If True, show all previous chunks; if False, show only current chunk.

    Examples
    --------
    >>> # RSVP (Rapid Serial Visual Presentation)
    >>> TimingParams(
    ...     duration_ms=250,
    ...     isi_ms=50,
    ...     cumulative=False,
    ...     mask_char="_"
    ... )
    >>> # Self-paced with timeout
    >>> TimingParams(timeout_ms=5000, cumulative=True)
    """

    duration_ms: int | None = Field(
        default=None, description="Per-chunk display duration (ms)"
    )
    isi_ms: int | None = Field(default=None, description="Inter-stimulus interval (ms)")
    timeout_ms: int | None = Field(default=None, description="Response timeout (ms)")
    mask_char: str | None = Field(default=None, description="Masking character")
    cumulative: bool = Field(
        default=True, description="Show all previous chunks or only current"
    )


class TaskSpec(BeadBaseModel):
    """Parameters for the response collection task.

    Specifies task-specific parameters like prompts, options, scale bounds,
    validation rules, etc. The appropriate parameters depend on the task_type
    specified in ItemTemplate. The task_type itself is not included here since
    it's part of the ItemTemplate structure.

    Attributes
    ----------
    prompt : str
        Question or instruction shown to participants.
    scale_bounds : tuple[int, int] | None
        Min and max values for ordinal_scale task.
    scale_labels : dict[int, str] | None
        Optional labels for specific scale points (ordinal_scale).
    options : list[str] | None
        Available options for forced_choice, multi_select, or categorical tasks.
        For forced_choice/multi_select: element names to choose from.
        For categorical: category labels.
    min_selections : int | None
        Minimum number of selections required (multi_select only).
    max_selections : int | None
        Maximum number of selections allowed (multi_select only).
    text_validation_pattern : str | None
        Regular expression pattern for validating free_text responses.
    max_length : int | None
        Maximum character length for free_text responses.

    Examples
    --------
    >>> # Ordinal scale task (e.g., acceptability rating)
    >>> TaskSpec(
    ...     prompt="How natural does this sentence sound?",
    ...     scale_bounds=(1, 7),
    ...     scale_labels={1: "Very unnatural", 7: "Very natural"}
    ... )
    >>> # Categorical task (e.g., NLI)
    >>> TaskSpec(
    ...     prompt="What is the relationship?",
    ...     options=["Entailment", "Neutral", "Contradiction"]
    ... )
    >>> # Binary task
    >>> TaskSpec(
    ...     prompt="Is this sentence grammatical?"
    ... )
    >>> # Forced choice task (e.g., minimal pair)
    >>> TaskSpec(
    ...     prompt="Which sounds more natural?",
    ...     options=["sentence_a", "sentence_b"]
    ... )
    >>> # Multi-select task (e.g., select all grammatical)
    >>> TaskSpec(
    ...     prompt="Select all grammatical sentences:",
    ...     options=["sent_a", "sent_b", "sent_c"],
    ...     min_selections=1
    ... )
    >>> # Free text task
    >>> TaskSpec(
    ...     prompt="Who performed the action?",
    ...     max_length=50
    ... )
    """

    prompt: str = Field(..., description="Participant prompt/question")
    scale_bounds: tuple[int, int] | None = Field(
        default=None, description="Scale bounds for ordinal_scale task"
    )
    scale_labels: dict[int, str] | None = Field(
        default=None, description="Labels for scale points"
    )
    options: list[str] | None = Field(
        default=None,
        description="Options for forced_choice/multi_select/categorical tasks",
    )
    min_selections: int | None = Field(
        default=None, description="Minimum selections for multi_select task"
    )
    max_selections: int | None = Field(
        default=None, description="Maximum selections for multi_select task"
    )
    text_validation_pattern: str | None = Field(
        default=None, description="Regex pattern for text validation"
    )
    max_length: int | None = Field(default=None, description="Maximum text length")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty.

        Parameters
        ----------
        v : str
            Prompt to validate.

        Returns
        -------
        str
            Validated prompt.

        Raises
        ------
        ValueError
            If prompt is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class PresentationSpec(BeadBaseModel):
    """Specification of stimulus presentation method.

    Defines how stimuli are displayed to participants (static, self-paced,
    or timed sequence), including segmentation and timing parameters.
    Separate from judgment specification to maintain clean separation
    of concerns.

    Attributes
    ----------
    mode : PresentationMode
        Presentation mode (static, self_paced, or timed_sequence).
    chunking : ChunkingSpec | None
        Chunking specification for incremental presentations.
    timing : TimingParams | None
        Timing parameters for timed presentations.
    display_format : dict[str, str | int | float | bool]
        Additional display formatting options.

    Examples
    --------
    >>> # Static presentation (default)
    >>> PresentationSpec(mode="static")
    >>> # Self-paced word-by-word reading
    >>> PresentationSpec(
    ...     mode="self_paced",
    ...     chunking=ChunkingSpec(unit="word")
    ... )
    >>> # Self-paced by noun phrases
    >>> PresentationSpec(
    ...     mode="self_paced",
    ...     chunking=ChunkingSpec(
    ...         unit="constituent",
    ...         parse_type="constituency",
    ...         constituent_labels=["NP"],
    ...         parser="stanza",
    ...         parse_language="en"
    ...     )
    ... )
    >>> # RSVP (timed sequence)
    >>> PresentationSpec(
    ...     mode="timed_sequence",
    ...     chunking=ChunkingSpec(unit="word"),
    ...     timing=TimingParams(duration_ms=250, isi_ms=50, cumulative=False)
    ... )
    """

    mode: PresentationMode = Field(..., description="Presentation mode")
    chunking: ChunkingSpec | None = Field(
        default=None, description="Chunking specification"
    )
    timing: TimingParams | None = Field(default=None, description="Timing parameters")
    display_format: dict[str, str | int | float | bool] = Field(
        default_factory=_empty_display_format_dict,
        description="Display formatting options",
    )


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


class ItemElement(BeadBaseModel):
    """A structured element within an item template.

    ItemElements represent distinct parts of a complex item,
    such as context, target sentence, question, or response options.
    Elements can be static text or references to filled templates.

    Attributes
    ----------
    element_type : ElementRefType
        Type of element ("text" or "filled_template_ref").
    element_name : str
        Unique name for this element within the item.
    content : str | None
        Static text content (for text elements).
    filled_template_ref_id : UUID | None
        UUID of filled template (for reference elements).
    element_metadata : dict[str, MetadataValue]
        Additional element-specific metadata.
    order : int | None
        Display order for this element (optional).

    Examples
    --------
    >>> # Text element
    >>> context = ItemElement(
    ...     element_type="text",
    ...     element_name="context",
    ...     content="Mary loves books.",
    ...     order=1
    ... )
    >>> # Template reference element
    >>> target = ItemElement(
    ...     element_type="filled_template_ref",
    ...     element_name="target",
    ...     filled_template_ref_id=UUID("..."),
    ...     order=2
    ... )
    """

    element_type: ElementRefType = Field(..., description="Type of element")
    element_name: str = Field(..., description="Unique element name within item")
    content: str | None = Field(default=None, description="Static text content")
    filled_template_ref_id: UUID | None = Field(
        default=None, description="Filled template reference"
    )
    element_metadata: dict[str, MetadataValue] = Field(
        default_factory=_empty_metadata_dict, description="Element-specific metadata"
    )
    order: int | None = Field(
        default=None, description="Display order for this element"
    )

    @field_validator("element_name")
    @classmethod
    def validate_element_name(cls, v: str) -> str:
        """Validate element name is not empty.

        Parameters
        ----------
        v : str
            Element name to validate.

        Returns
        -------
        str
            Validated element name.

        Raises
        ------
        ValueError
            If name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Element name cannot be empty")
        return v.strip()

    @property
    def is_text(self) -> bool:
        """Check if this is a text element.

        Returns
        -------
        bool
            True if element_type is "text".
        """
        return self.element_type == "text"

    @property
    def is_template_ref(self) -> bool:
        """Check if this references a filled template.

        Returns
        -------
        bool
            True if element_type is "filled_template_ref".
        """
        return self.element_type == "filled_template_ref"


class ItemTemplate(BeadBaseModel):
    """Template specification for constructing experimental items.

    ItemTemplate defines how to construct an experimental item with three
    orthogonal dimensions: what semantic property to measure (judgment_type),
    how to collect the response (task_type), and how to present the stimulus
    (presentation_spec).

    This is distinct from Template (in bead.resources.structures), which defines
    linguistic structure. ItemTemplate defines experimental structure.

    Attributes
    ----------
    name : str
        Template name (e.g., "acceptability_rating").
    description : str | None
        Human-readable description of this item template.
    judgment_type : JudgmentType
        Semantic property being measured (acceptability, inference, etc.).
    task_type : TaskType
        Response collection method (forced_choice, ordinal_scale, etc.).
    elements : list[ItemElement]
        Elements that compose this item.
    constraints : list[UUID]
        UUIDs of constraints on items (typically model-based).
    task_spec : TaskSpec
        Task-specific parameters (prompt, options, scale bounds, etc.).
    presentation_spec : PresentationSpec
        Specification of how to present stimuli.
    presentation_order : list[str] | None
        Order to present elements (by element_name).
    template_metadata : dict[str, MetadataValue]
        Additional template metadata.

    Examples
    --------
    >>> # Acceptability judgment with ordinal scale task
    >>> template = ItemTemplate(
    ...     name="acceptability_rating",
    ...     judgment_type="acceptability",
    ...     task_type="ordinal_scale",
    ...     task_spec=TaskSpec(
    ...         prompt="How natural is this sentence?",
    ...         scale_bounds=(1, 7),
    ...         scale_labels={1: "Very unnatural", 7: "Very natural"}
    ...     ),
    ...     presentation_spec=PresentationSpec(mode="static"),
    ...     elements=[
    ...         ItemElement(
    ...             element_type="filled_template_ref",
    ...             element_name="sentence",
    ...             filled_template_ref_id=UUID("...")
    ...         )
    ...     ]
    ... )
    >>> # Minimal pair: acceptability judgment with forced choice task
    >>> minimal_pair = ItemTemplate(
    ...     name="minimal_pair",
    ...     judgment_type="acceptability",
    ...     task_type="forced_choice",
    ...     elements=[
    ...         ItemElement(
    ...             element_type="text", element_name="sent_a", content="Who..."
    ...         ),
    ...         ItemElement(
    ...             element_type="text", element_name="sent_b", content="Whom..."
    ...         )
    ...     ],
    ...     task_spec=TaskSpec(
    ...         prompt="Which sounds more natural?",
    ...         options=["sent_a", "sent_b"]
    ...     ),
    ...     presentation_spec=PresentationSpec(mode="static")
    ... )
    >>> # Odd-man-out: similarity judgment with forced choice task
    >>> odd_man_out = ItemTemplate(
    ...     name="odd_man_out",
    ...     judgment_type="similarity",
    ...     task_type="forced_choice",
    ...     elements=[...],  # 4 elements
    ...     task_spec=TaskSpec(
    ...         prompt="Which is most different?",
    ...         options=["opt_a", "opt_b", "opt_c", "opt_d"]
    ...     ),
    ...     presentation_spec=PresentationSpec(mode="static")
    ... )
    """

    name: str = Field(..., description="Template name")
    description: str | None = Field(default=None, description="Template description")
    judgment_type: JudgmentType = Field(
        ..., description="Semantic property being measured"
    )
    task_type: TaskType = Field(..., description="Response collection method")
    elements: list[ItemElement] = Field(
        default_factory=_empty_item_element_list, description="Item elements"
    )
    constraints: list[UUID] = Field(
        default_factory=_empty_uuid_list, description="Constraint UUIDs"
    )
    task_spec: TaskSpec = Field(..., description="Task-specific parameters")
    presentation_spec: PresentationSpec = Field(
        ..., description="Presentation specification"
    )
    presentation_order: list[str] | None = Field(
        default=None, description="Element presentation order"
    )
    template_metadata: dict[str, MetadataValue] = Field(
        default_factory=_empty_metadata_dict, description="Additional metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate template name is not empty.

        Parameters
        ----------
        v : str
            Template name to validate.

        Returns
        -------
        str
            Validated template name.

        Raises
        ------
        ValueError
            If name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Template name cannot be empty")
        return v.strip()

    @field_validator("elements")
    @classmethod
    def validate_unique_element_names(cls, v: list[ItemElement]) -> list[ItemElement]:
        """Validate all element names are unique within template.

        Parameters
        ----------
        v : list[ItemElement]
            List of elements to validate.

        Returns
        -------
        list[ItemElement]
            Validated elements.

        Raises
        ------
        ValueError
            If duplicate element names found.
        """
        if not v:
            return v

        names = [elem.element_name for elem in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate element names: {set(duplicates)}")

        return v

    @field_validator("presentation_order", mode="after")
    @classmethod
    def validate_presentation_order(
        cls, v: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        """Validate presentation_order matches element names.

        Parameters
        ----------
        v : list[str] | None
            Presentation order list to validate.
        info : ValidationInfo
            Pydantic validation info containing other field values.

        Returns
        -------
        list[str] | None
            Validated presentation order.

        Raises
        ------
        ValueError
            If presentation_order contains names not in elements,
            or is missing names from elements.
        """
        if v is None:
            return v

        # Get elements from validation info
        elements = info.data.get("elements", [])
        if not elements:
            return v

        element_names = {e.element_name for e in elements}
        order_names = set(v)

        # Check for names in order that aren't in elements
        extra = order_names - element_names
        if extra:
            raise ValueError(
                f"presentation_order contains element names not in elements: {extra}"
            )

        # Check for names in elements that aren't in order
        missing = element_names - order_names
        if missing:
            raise ValueError(
                f"presentation_order missing element names from elements: {missing}"
            )

        return v

    def get_element_by_name(self, name: str) -> ItemElement | None:
        """Get an element by its name.

        Parameters
        ----------
        name : str
            Element name to search for.

        Returns
        -------
        ItemElement | None
            Element with matching name, or None if not found.

        Examples
        --------
        >>> elem = template.get_element_by_name("sentence")
        >>> if elem:
        ...     print(elem.element_type)
        """
        for elem in self.elements:
            if elem.element_name == name:
                return elem
        return None

    def get_template_ref_elements(self) -> list[ItemElement]:
        """Get all elements that reference filled templates.

        Returns
        -------
        list[ItemElement]
            Elements with element_type="filled_template_ref".

        Examples
        --------
        >>> refs = template.get_template_ref_elements()
        >>> print(f"Found {len(refs)} template references")
        """
        return [elem for elem in self.elements if elem.is_template_ref]


class ItemTemplateCollection(BeadBaseModel):
    """A collection of item templates.

    Attributes
    ----------
    name : str
        Name of this collection.
    description : str | None
        Description of this collection.
    templates : list[ItemTemplate]
        Item templates in this collection.

    Examples
    --------
    >>> collection = ItemTemplateCollection(
    ...     name="acceptability_study",
    ...     description="Templates for acceptability judgments"
    ... )
    >>> collection.add_template(template)
    """

    name: str = Field(..., description="Collection name")
    description: str | None = Field(default=None, description="Collection description")
    templates: list[ItemTemplate] = Field(
        default_factory=_empty_item_template_list, description="Item templates"
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

    def add_template(self, template: ItemTemplate) -> None:
        """Add a template to the collection.

        Parameters
        ----------
        template : ItemTemplate
            Template to add.

        Examples
        --------
        >>> collection.add_template(my_template)
        >>> print(f"Collection now has {len(collection.templates)} templates")
        """
        self.templates.append(template)
        self.update_modified_time()


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
