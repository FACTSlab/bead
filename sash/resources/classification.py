"""Linguistic classification models for lexical items and templates.

This module provides models for grouping lexical items and templates by
linguistic properties. These classifications enable cross-linguistic analysis
and alignment, supporting both monolingual and multilingual classification.

LexicalItemClass and TemplateClass are NOT subclasses of Lexicon and
TemplateCollection. This is a deliberate architectural choice:
- Lexicon/TemplateCollection: Operational resource management for experiments
- LexicalItemClass/TemplateClass: Analytical linguistic classification

Primary use cases:
- Cross-linguistic analysis and comparison
- Aligning resources across languages for meta-analysis
- Combining experimental results by linguistic class
- Linguistic typology studies
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from sash.data.base import SashBaseModel
from sash.data.language_codes import validate_iso639_code
from sash.resources.models import LexicalItem
from sash.resources.structures import Template


def _empty_str_list() -> list[str]:
    """Create an empty string list."""
    return []


def _empty_item_dict() -> dict[UUID, LexicalItem]:
    """Create an empty lexical item dictionary."""
    return {}


def _empty_template_dict() -> dict[UUID, Template]:
    """Create an empty template dictionary."""
    return {}


def _empty_metadata_dict() -> dict[str, Any]:
    """Create an empty metadata dictionary."""
    return {}


class LexicalItemClass(SashBaseModel):
    """Groups lexical items that share linguistic properties.

    LexicalItemClass represents a linguistic classification that can span
    a single language (e.g., "all causative verbs in English") or multiple
    languages (e.g., "all causative verbs across English, Korean, Zulu").

    Primary use cases:
    - Cross-linguistic analysis and comparison
    - Aligning lexical items across languages for meta-analysis
    - Combining experimental results by lexical class
    - Linguistic typology studies

    NOT typically used for:
    - Experiment generation (use Lexicon for that)
    - Resource storage (use Lexicon for that)

    Attributes
    ----------
    name : str
        Name of this lexical item class.
    description : str | None
        Description of the classification (e.g., "Causative verbs").
    property_name : str
        The linguistic property that defines this class (e.g., "causative",
        "transitive", "stative").
    property_value : Any | None
        Optional specific value for the property (e.g., True, "agentive").
    items : dict[UUID, LexicalItem]
        Dictionary of lexical items in this class, indexed by UUID.
    tags : list[str]
        Tags for organization and search.
    class_metadata : dict[str, Any]
        Additional metadata about this classification.

    Examples
    --------
    >>> # Monolingual classification
    >>> causative_en = LexicalItemClass(
    ...     name="causative_verbs_en",
    ...     description="Causative verbs in English",
    ...     property_name="causative",
    ...     property_value=True
    ... )
    >>> # Multilingual cross-linguistic classification
    >>> causatives_multi = LexicalItemClass(
    ...     name="causative_verbs_crossling",
    ...     description="Causative verbs across EN, KO, ZU",
    ...     property_name="causative",
    ...     property_value=True
    ... )
    >>> english_break = LexicalItem(lemma="break", language_code="en")
    >>> korean_kkakta = LexicalItem(lemma="kkakta", language_code="ko")
    >>> causatives_multi.add(english_break)
    >>> causatives_multi.add(korean_kkakta)
    >>> len(causatives_multi)
    2
    >>> causatives_multi.is_multilingual()
    True
    >>> for lang in causatives_multi.languages():
    ...     items = causatives_multi.get_items_by_language(lang)
    ...     print(f"{lang}: {len(items)} causative verbs")
    en: 1 causative verbs
    ko: 1 causative verbs
    """

    name: str
    description: str | None = None
    property_name: str
    property_value: Any | None = None
    items: dict[UUID, LexicalItem] = Field(default_factory=_empty_item_dict)
    tags: list[str] = Field(default_factory=_empty_str_list)
    class_metadata: dict[str, Any] = Field(default_factory=_empty_metadata_dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is non-empty.

        Parameters
        ----------
        v : str
            The name to validate.

        Returns
        -------
        str
            The validated name.

        Raises
        ------
        ValueError
            If name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("property_name")
    @classmethod
    def validate_property_name(cls, v: str) -> str:
        """Validate that property_name is non-empty.

        Parameters
        ----------
        v : str
            The property name to validate.

        Returns
        -------
        str
            The validated property name.

        Raises
        ------
        ValueError
            If property_name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_name must be non-empty")
        return v

    def languages(self) -> set[str]:
        """Return set of language codes present in this class.

        Items without language_code are excluded from the result.

        Returns
        -------
        set[str]
            Set of language codes (lowercase) found in this class.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> cls.add(LexicalItem(lemma="break", language_code="en"))
        >>> cls.add(LexicalItem(lemma="kkakta", language_code="ko"))
        >>> cls.languages()
        {'en', 'ko'}
        """
        return {
            item.language_code.lower()
            for item in self.items.values()
            if item.language_code is not None
        }

    def get_items_by_language(self, language_code: str) -> list[LexicalItem]:
        """Filter items by language code.

        Accepts both ISO 639-1 (2-letter) and ISO 639-3 (3-letter) codes.
        The query code is normalized to ISO 639-3 for comparison.

        Parameters
        ----------
        language_code : str
            Language code to filter by (e.g., "en", "eng", "ko", "kor").

        Returns
        -------
        list[LexicalItem]
            List of items matching the language code.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> cls.add(LexicalItem(lemma="break", language_code="en"))
        >>> cls.add(LexicalItem(lemma="kkakta", language_code="ko"))
        >>> en_items = cls.get_items_by_language("en")
        >>> len(en_items)
        1
        >>> en_items[0].lemma
        'break'
        """
        # Normalize the query language code to ISO 639-3
        try:
            normalized_code = validate_iso639_code(language_code)
            if normalized_code is None:
                return []
            lang_normalized = normalized_code.lower()
        except ValueError:
            # Invalid language code, return empty list
            return []

        return [
            item
            for item in self.items.values()
            if item.language_code is not None
            and item.language_code.lower() == lang_normalized
        ]

    def is_monolingual(self) -> bool:
        """Check if class contains only one language.

        Returns
        -------
        bool
            True if class contains items from only one language (or no items).

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> cls.add(LexicalItem(lemma="break", language_code="en"))
        >>> cls.is_monolingual()
        True
        >>> cls.add(LexicalItem(lemma="kkakta", language_code="ko"))
        >>> cls.is_monolingual()
        False
        """
        return len(self.languages()) <= 1

    def is_multilingual(self) -> bool:
        """Check if class contains multiple languages.

        Returns
        -------
        bool
            True if class contains items from more than one language.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> cls.add(LexicalItem(lemma="break", language_code="en"))
        >>> cls.is_multilingual()
        False
        >>> cls.add(LexicalItem(lemma="kkakta", language_code="ko"))
        >>> cls.is_multilingual()
        True
        """
        return len(self.languages()) > 1

    def add(self, item: LexicalItem) -> None:
        """Add a lexical item to the class.

        Parameters
        ----------
        item : LexicalItem
            The item to add.

        Raises
        ------
        ValueError
            If item with same ID already exists.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> item = LexicalItem(lemma="break")
        >>> cls.add(item)
        >>> len(cls)
        1
        """
        if item.id in self.items:
            raise ValueError(f"Item with ID {item.id} already exists in class")
        self.items[item.id] = item
        self.update_modified_time()

    def remove(self, item_id: UUID) -> LexicalItem:
        """Remove and return an item by ID.

        Parameters
        ----------
        item_id : UUID
            The ID of the item to remove.

        Returns
        -------
        LexicalItem
            The removed item.

        Raises
        ------
        KeyError
            If item ID not found.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> item = LexicalItem(lemma="break")
        >>> cls.add(item)
        >>> removed = cls.remove(item.id)
        >>> removed.lemma
        'break'
        >>> len(cls)
        0
        """
        if item_id not in self.items:
            raise KeyError(f"Item with ID {item_id} not found in class")
        item = self.items.pop(item_id)
        self.update_modified_time()
        return item

    def get(self, item_id: UUID) -> LexicalItem | None:
        """Get an item by ID, or None if not found.

        Parameters
        ----------
        item_id : UUID
            The ID of the item to get.

        Returns
        -------
        LexicalItem | None
            The item if found, None otherwise.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> item = LexicalItem(lemma="break")
        >>> cls.add(item)
        >>> retrieved = cls.get(item.id)
        >>> retrieved.lemma  # doctest: +SKIP
        'break'
        >>> from uuid import uuid4
        >>> cls.get(uuid4()) is None
        True
        """
        return self.items.get(item_id)

    def __len__(self) -> int:
        """Return number of items in class.

        Returns
        -------
        int
            Number of items in the class.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> len(cls)
        0
        >>> cls.add(LexicalItem(lemma="break"))
        >>> len(cls)
        1
        """
        return len(self.items)

    def __contains__(self, item_id: UUID) -> bool:
        """Check if item ID is in class.

        Parameters
        ----------
        item_id : UUID
            The item ID to check.

        Returns
        -------
        bool
            True if item ID exists in class.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> item = LexicalItem(lemma="break")
        >>> cls.add(item)
        >>> item.id in cls
        True
        """
        return item_id in self.items

    def __iter__(self) -> Iterator[LexicalItem]:  # type: ignore[override]
        """Iterate over items in class.

        Returns
        -------
        Iterator[LexicalItem]
            Iterator over lexical items.

        Examples
        --------
        >>> cls = LexicalItemClass(name="test", property_name="causative")
        >>> cls.add(LexicalItem(lemma="break"))
        >>> cls.add(LexicalItem(lemma="open"))
        >>> [item.lemma for item in cls]
        ['break', 'open']
        """
        return iter(self.items.values())


class TemplateClass(SashBaseModel):
    """Groups templates that share linguistic properties.

    TemplateClass represents a linguistic classification that can span
    a single language (e.g., "transitive templates in English that vary
    only in adjuncts") or multiple languages (e.g., "causative-inchoative
    alternation templates across languages").

    Primary use cases:
    - Cross-linguistic analysis and comparison
    - Identifying systematic variation patterns (e.g., adjunct variation)
    - Aligning templates across languages for meta-analysis
    - Combining experimental results by template class
    - Linguistic typology studies

    NOT typically used for:
    - Experiment generation (use TemplateCollection for that)
    - Operational template storage (use TemplateCollection for that)

    Attributes
    ----------
    name : str
        Name of this template class.
    description : str | None
        Description of the classification (e.g., "Transitive with adjunct variation").
    property_name : str
        The linguistic property that defines this class (e.g., "transitive",
        "causative_inchoative", "wh_question").
    property_value : Any | None
        Optional specific value for the property.
    templates : dict[UUID, Template]
        Dictionary of templates in this class, indexed by UUID.
    tags : list[str]
        Tags for organization and search.
    class_metadata : dict[str, Any]
        Additional metadata about this classification.

    Examples
    --------
    >>> from sash.resources.structures import Slot
    >>> # Monolingual classification
    >>> transitive_en = TemplateClass(
    ...     name="transitive_templates_en",
    ...     description="Transitive templates in English",
    ...     property_name="transitive",
    ...     property_value=True
    ... )
    >>> # Multilingual cross-linguistic classification
    >>> transitives_multi = TemplateClass(
    ...     name="transitive_templates_crossling",
    ...     description="Transitive templates across languages",
    ...     property_name="transitive",
    ...     property_value=True
    ... )
    >>> en_template = Template(
    ...     name="svo",
    ...     template_string="{subject} {verb} {object}.",
    ...     slots={"subject": Slot(name="subject"), "verb": Slot(name="verb"),
    ...            "object": Slot(name="object")},
    ...     language_code="en"
    ... )
    >>> transitives_multi.add(en_template)
    >>> len(transitives_multi)
    1
    """

    name: str
    description: str | None = None
    property_name: str
    property_value: Any | None = None
    templates: dict[UUID, Template] = Field(default_factory=_empty_template_dict)
    tags: list[str] = Field(default_factory=_empty_str_list)
    class_metadata: dict[str, Any] = Field(default_factory=_empty_metadata_dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is non-empty.

        Parameters
        ----------
        v : str
            The name to validate.

        Returns
        -------
        str
            The validated name.

        Raises
        ------
        ValueError
            If name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("property_name")
    @classmethod
    def validate_property_name(cls, v: str) -> str:
        """Validate that property_name is non-empty.

        Parameters
        ----------
        v : str
            The property name to validate.

        Returns
        -------
        str
            The validated property name.

        Raises
        ------
        ValueError
            If property_name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_name must be non-empty")
        return v

    def languages(self) -> set[str]:
        """Return set of language codes present in this class.

        Templates without language_code are excluded from the result.

        Returns
        -------
        set[str]
            Set of language codes (lowercase) found in this class.

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="en_svo",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        ...     language_code="en"
        ... )
        >>> cls.add(t1)
        >>> cls.languages()
        {'en'}
        """
        return {
            template.language_code.lower()
            for template in self.templates.values()
            if template.language_code is not None
        }

    def get_templates_by_language(self, language_code: str) -> list[Template]:
        """Filter templates by language code.

        Accepts both ISO 639-1 (2-letter) and ISO 639-3 (3-letter) codes.
        The query code is normalized to ISO 639-3 for comparison.

        Parameters
        ----------
        language_code : str
            Language code to filter by (e.g., "en", "eng", "ko", "kor").

        Returns
        -------
        list[Template]
            List of templates matching the language code.

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="en_svo",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        ...     language_code="en"
        ... )
        >>> cls.add(t1)
        >>> en_templates = cls.get_templates_by_language("en")
        >>> len(en_templates)
        1
        >>> en_templates[0].name
        'en_svo'
        """
        # Normalize the query language code to ISO 639-3
        try:
            normalized_code = validate_iso639_code(language_code)
            if normalized_code is None:
                return []
            lang_normalized = normalized_code.lower()
        except ValueError:
            # Invalid language code, return empty list
            return []

        return [
            template
            for template in self.templates.values()
            if template.language_code is not None
            and template.language_code.lower() == lang_normalized
        ]

    def is_monolingual(self) -> bool:
        """Check if class contains only one language.

        Returns
        -------
        bool
            True if class contains templates from only one language (or no templates).

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="en_svo",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        ...     language_code="en"
        ... )
        >>> cls.add(t1)
        >>> cls.is_monolingual()
        True
        """
        return len(self.languages()) <= 1

    def is_multilingual(self) -> bool:
        """Check if class contains multiple languages.

        Returns
        -------
        bool
            True if class contains templates from more than one language.

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="en_svo",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        ...     language_code="en"
        ... )
        >>> cls.add(t1)
        >>> cls.is_multilingual()
        False
        """
        return len(self.languages()) > 1

    def add(self, template: Template) -> None:
        """Add a template to the class.

        Parameters
        ----------
        template : Template
            The template to add.

        Raises
        ------
        ValueError
            If template with same ID already exists.

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="svo",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")}
        ... )
        >>> cls.add(t1)
        >>> len(cls)
        1
        """
        if template.id in self.templates:
            raise ValueError(f"Template with ID {template.id} already exists in class")
        self.templates[template.id] = template
        self.update_modified_time()

    def remove(self, template_id: UUID) -> Template:
        """Remove and return a template by ID.

        Parameters
        ----------
        template_id : UUID
            The ID of the template to remove.

        Returns
        -------
        Template
            The removed template.

        Raises
        ------
        KeyError
            If template ID not found.

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="svo",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")}
        ... )
        >>> cls.add(t1)
        >>> removed = cls.remove(t1.id)
        >>> removed.name
        'svo'
        >>> len(cls)
        0
        """
        if template_id not in self.templates:
            raise KeyError(f"Template with ID {template_id} not found in class")
        template = self.templates.pop(template_id)
        self.update_modified_time()
        return template

    def get(self, template_id: UUID) -> Template | None:
        """Get a template by ID, or None if not found.

        Parameters
        ----------
        template_id : UUID
            The ID of the template to get.

        Returns
        -------
        Template | None
            The template if found, None otherwise.

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="svo",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")}
        ... )
        >>> cls.add(t1)
        >>> retrieved = cls.get(t1.id)
        >>> retrieved.name  # doctest: +SKIP
        'svo'
        >>> from uuid import uuid4
        >>> cls.get(uuid4()) is None
        True
        """
        return self.templates.get(template_id)

    def __len__(self) -> int:
        """Return number of templates in class.

        Returns
        -------
        int
            Number of templates in the class.

        Examples
        --------
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> len(cls)
        0
        """
        return len(self.templates)

    def __contains__(self, template_id: UUID) -> bool:
        """Check if template ID is in class.

        Parameters
        ----------
        template_id : UUID
            The template ID to check.

        Returns
        -------
        bool
            True if template ID exists in class.

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="svo",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")}
        ... )
        >>> cls.add(t1)
        >>> t1.id in cls
        True
        """
        return template_id in self.templates

    def __iter__(self) -> Iterator[Template]:  # type: ignore[override]
        """Iterate over templates in class.

        Returns
        -------
        Iterator[Template]
            Iterator over templates.

        Examples
        --------
        >>> from sash.resources.structures import Slot
        >>> cls = TemplateClass(name="test", property_name="transitive")
        >>> t1 = Template(
        ...     name="svo1",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")}
        ... )
        >>> t2 = Template(
        ...     name="svo2",
        ...     template_string="{s} {v} {o}.",
        ...     slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")}
        ... )
        >>> cls.add(t1)
        >>> cls.add(t2)
        >>> [t.name for t in cls]
        ['svo1', 'svo2']
        """
        return iter(self.templates.values())
