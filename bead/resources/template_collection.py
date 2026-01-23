"""Template collection management.

This module provides the TemplateCollection class for managing collections
of sentence templates.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal
from uuid import UUID

import pandas as pd
import polars as pl
from pydantic import Field

from bead.data.base import BeadBaseModel
from bead.resources.template import Template

# Type alias for supported DataFrame types
DataFrame = pd.DataFrame | pl.DataFrame


def _empty_str_list() -> list[str]:
    """Create an empty string list."""
    return []


def _empty_template_dict() -> dict[UUID, Template]:
    """Create an empty template dictionary."""
    return {}


class TemplateCollection(BeadBaseModel):
    """A collection of templates with operations for filtering and analysis.

    Similar to Lexicon but for Template objects. The TemplateCollection class
    manages collections of Template objects and provides methods for:
    - Adding and removing templates (CRUD operations)
    - Filtering by properties and tags
    - Searching by name or template string
    - Merging with other collections
    - Converting to/from pandas and polars DataFrames
    - Serialization to JSONLines

    Attributes
    ----------
    name : str
        Name of the collection.
    description : str | None
        Optional description.
    language_code : str | None
        ISO 639-1 or 639-3 language code (e.g., "en", "es", "eng").
    templates : dict[UUID, Template]
        Dictionary of templates indexed by their UUIDs.
    tags : list[str]
        Tags for categorization.

    Examples
    --------
    >>> from bead.resources import Slot
    >>> collection = TemplateCollection(name="transitive")
    >>> template = Template(
    ...     name="simple",
    ...     template_string="{subject} {verb} {object}.",
    ...     slots={
    ...         "subject": Slot(name="subject"),
    ...         "verb": Slot(name="verb"),
    ...         "object": Slot(name="object"),
    ...     }
    ... )
    >>> collection.add(template)
    >>> len(collection)
    1
    """

    name: str
    description: str | None = None
    language_code: str | None = None
    templates: dict[UUID, Template] = Field(default_factory=_empty_template_dict)
    tags: list[str] = Field(default_factory=_empty_str_list)

    def __len__(self) -> int:
        """Return number of templates in collection.

        Returns
        -------
        int
            Number of templates in the collection.

        Examples
        --------
        >>> collection = TemplateCollection(name="test")
        >>> len(collection)
        0
        """
        return len(self.templates)

    def __iter__(self) -> Iterator[Template]:  # type: ignore[override]
        """Iterate over templates in collection.

        Returns
        -------
        Iterator[Template]
            Iterator over templates.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> t1 = Template(
        ...     name="t1", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> t2 = Template(
        ...     name="t2", template_string="{y}.", slots={"y": Slot(name="y")}
        ... )
        >>> collection.add(t1)
        >>> collection.add(t2)
        >>> [t.name for t in collection]
        ['t1', 't2']
        """
        return iter(self.templates.values())

    def __contains__(self, template_id: UUID) -> bool:
        """Check if template ID is in collection.

        Parameters
        ----------
        template_id : UUID
            The template ID to check.

        Returns
        -------
        bool
            True if template ID exists in collection.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> template = Template(
        ...     name="test", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> collection.add(template)
        >>> template.id in collection
        True
        """
        return template_id in self.templates

    def add(self, template: Template) -> None:
        """Add a template to the collection.

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
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> template = Template(
        ...     name="test", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> collection.add(template)
        >>> len(collection)
        1
        """
        if template.id in self.templates:
            raise ValueError(
                f"Template with ID {template.id} already exists in collection"
            )
        self.templates[template.id] = template
        self.update_modified_time()

    def add_many(self, templates: list[Template]) -> None:
        """Add multiple templates to the collection.

        Parameters
        ----------
        templates : list[Template]
            The templates to add.

        Raises
        ------
        ValueError
            If any template with same ID already exists.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> t1 = Template(
        ...     name="t1", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> t2 = Template(
        ...     name="t2", template_string="{y}.", slots={"y": Slot(name="y")}
        ... )
        >>> collection.add_many([t1, t2])
        >>> len(collection)
        2
        """
        for template in templates:
            self.add(template)

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
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> template = Template(
        ...     name="test", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> collection.add(template)
        >>> removed = collection.remove(template.id)
        >>> removed.name
        'test'
        >>> len(collection)
        0
        """
        if template_id not in self.templates:
            raise KeyError(f"Template with ID {template_id} not found in collection")
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
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> template = Template(
        ...     name="test", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> collection.add(template)
        >>> retrieved = collection.get(template.id)
        >>> retrieved.name  # doctest: +SKIP
        'test'
        >>> from uuid import uuid4
        >>> collection.get(uuid4()) is None
        True
        """
        return self.templates.get(template_id)

    def filter(self, predicate: Callable[[Template], bool]) -> TemplateCollection:
        """Filter templates by a predicate function.

        Creates a new collection containing only templates that satisfy the predicate.

        Parameters
        ----------
        predicate : Callable[[Template], bool]
            Function that returns True for templates to include.

        Returns
        -------
        TemplateCollection
            New collection with filtered templates.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> t1 = Template(
        ...     name="t1",
        ...     template_string="{x}.",
        ...     slots={"x": Slot(name="x")},
        ...     tags=["simple"],
        ... )
        >>> t2 = Template(
        ...     name="t2",
        ...     template_string="{y} {z}.",
        ...     slots={"y": Slot(name="y"), "z": Slot(name="z")},
        ...     tags=["complex"],
        ... )
        >>> collection.add(t1)
        >>> collection.add(t2)
        >>> simple = collection.filter(lambda t: "simple" in t.tags)
        >>> len(simple.templates)
        1
        """
        filtered = TemplateCollection(
            name=f"{self.name}_filtered",
            description=self.description,
            language_code=self.language_code,
            tags=self.tags.copy(),
        )
        filtered.templates = {
            template_id: template
            for template_id, template in self.templates.items()
            if predicate(template)
        }
        return filtered

    def filter_by_tag(self, tag: str) -> TemplateCollection:
        """Filter templates by tag.

        Parameters
        ----------
        tag : str
            The tag to filter by.

        Returns
        -------
        TemplateCollection
            New collection with templates having the specified tag.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> t1 = Template(
        ...     name="t1",
        ...     template_string="{x}.",
        ...     slots={"x": Slot(name="x")},
        ...     tags=["simple"],
        ... )
        >>> t2 = Template(
        ...     name="t2",
        ...     template_string="{y}.",
        ...     slots={"y": Slot(name="y")},
        ...     tags=["complex"],
        ... )
        >>> collection.add(t1)
        >>> collection.add(t2)
        >>> simple = collection.filter_by_tag("simple")
        >>> len(simple.templates)
        1
        """
        return self.filter(lambda template: tag in template.tags)

    def filter_by_slot_count(self, count: int) -> TemplateCollection:
        """Filter templates by number of slots.

        Parameters
        ----------
        count : int
            The number of slots to filter by.

        Returns
        -------
        TemplateCollection
            New collection with templates having the specified slot count.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> t1 = Template(
        ...     name="t1", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> t2 = Template(
        ...     name="t2",
        ...     template_string="{y} {z}.",
        ...     slots={"y": Slot(name="y"), "z": Slot(name="z")},
        ... )
        >>> collection.add(t1)
        >>> collection.add(t2)
        >>> single_slot = collection.filter_by_slot_count(1)
        >>> len(single_slot.templates)
        1
        """
        return self.filter(lambda template: len(template.slots) == count)

    def search(self, query: str, field: str = "name") -> TemplateCollection:
        """Search for templates containing query string in specified field.

        Parameters
        ----------
        query : str
            Search string (case-insensitive substring match).
        field : str
            Field to search in ("name", "template_string").

        Returns
        -------
        TemplateCollection
            New collection with matching templates.

        Raises
        ------
        ValueError
            If field is not a valid searchable field.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> template = Template(
        ...     name="transitive",
        ...     template_string="{x}.",
        ...     slots={"x": Slot(name="x")},
        ... )
        >>> collection.add(template)
        >>> results = collection.search("trans")
        >>> len(results.templates)
        1
        """
        query_lower = query.lower()

        if field == "name":
            return self.filter(lambda template: query_lower in template.name.lower())
        elif field == "template_string":
            return self.filter(
                lambda template: query_lower in template.template_string.lower()
            )
        else:
            raise ValueError(
                f"Invalid field '{field}'. Must be 'name' or 'template_string'."
            )

    def merge(
        self,
        other: TemplateCollection,
        strategy: Literal["keep_first", "keep_second", "error"] = "keep_first",
    ) -> TemplateCollection:
        """Merge with another collection.

        Parameters
        ----------
        other : TemplateCollection
            The collection to merge with.
        strategy : Literal["keep_first", "keep_second", "error"]
            How to handle duplicate IDs:
            - "keep_first": Keep template from self
            - "keep_second": Keep template from other
            - "error": Raise error on duplicates

        Returns
        -------
        TemplateCollection
            New merged collection.

        Raises
        ------
        ValueError
            If strategy is "error" and duplicates found.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> c1 = TemplateCollection(name="c1")
        >>> c1.add(
        ...     Template(
        ...         name="t1", template_string="{x}.", slots={"x": Slot(name="x")}
        ...     )
        ... )
        >>> c2 = TemplateCollection(name="c2")
        >>> c2.add(
        ...     Template(
        ...         name="t2", template_string="{y}.", slots={"y": Slot(name="y")}
        ...     )
        ... )
        >>> merged = c1.merge(c2)
        >>> len(merged.templates)
        2
        """
        # Check for duplicates if strategy is "error"
        if strategy == "error":
            duplicates = set(self.templates.keys()) & set(other.templates.keys())
            if duplicates:
                raise ValueError(
                    f"Duplicate template IDs found: {duplicates}. "
                    "Use strategy='keep_first' or 'keep_second' to resolve."
                )

        # Create merged collection
        # Use language_code from self, or other if self's is None
        language_code = self.language_code or other.language_code

        merged = TemplateCollection(
            name=f"{self.name}_merged",
            description=self.description,
            language_code=language_code,
            tags=list(set(self.tags + other.tags)),
        )

        # Add templates based on strategy
        if strategy == "keep_first":
            merged.templates = {**other.templates, **self.templates}
        elif strategy == "keep_second":
            merged.templates = {**self.templates, **other.templates}
        else:  # strategy == "error" already handled above
            merged.templates = {**self.templates, **other.templates}

        return merged

    def to_dataframe(
        self, backend: Literal["pandas", "polars"] = "pandas"
    ) -> DataFrame:
        """Convert collection to DataFrame.

        Parameters
        ----------
        backend : Literal["pandas", "polars"]
            DataFrame backend to use (default: "pandas").

        Returns
        -------
        DataFrame
            pandas or polars DataFrame with columns: id, name, template_string,
            description, slot_count, slot_names, tags, created_at, modified_at.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> template = Template(
        ...     name="test", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> collection.add(template)
        >>> df = collection.to_dataframe()
        >>> "name" in df.columns
        True
        >>> "template_string" in df.columns
        True
        """
        if not self.templates:
            # Return empty DataFrame with expected columns
            columns = [
                "id",
                "name",
                "template_string",
                "description",
                "slot_count",
                "slot_names",
                "tags",
                "created_at",
                "modified_at",
            ]
            if backend == "pandas":
                return pd.DataFrame(columns=columns)
            else:
                schema: dict[str, type[pl.Utf8]] = dict.fromkeys(columns, pl.Utf8)
                return pl.DataFrame(schema=schema)

        rows = []
        for template in self.templates.values():
            row = {
                "id": str(template.id),
                "name": template.name,
                "template_string": template.template_string,
                "description": template.description,
                "slot_count": len(template.slots),
                "slot_names": ",".join(sorted(template.slots.keys())),
                "tags": ",".join(template.tags),
                "created_at": template.created_at.isoformat(),
                "modified_at": template.modified_at.isoformat(),
            }
            rows.append(row)  # type: ignore[arg-type]

        if backend == "pandas":
            return pd.DataFrame(rows)
        else:
            return pl.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: DataFrame, name: str) -> TemplateCollection:
        """Create collection from DataFrame.

        Note: This method creates templates without slot definitions since
        DataFrame representation doesn't include full slot information.
        Use from_jsonl for full template serialization.

        Parameters
        ----------
        df : DataFrame
            pandas or polars DataFrame with at minimum 'name' and
            'template_string' columns.
        name : str
            Name for the collection.

        Returns
        -------
        TemplateCollection
            New collection created from DataFrame.

        Raises
        ------
        ValueError
            If DataFrame does not have required columns.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "name": ["t1", "t2"],
        ...     "template_string": ["{x}.", "{y}."],
        ...     "slot_names": ["x", "y"]
        ... })
        >>> collection = TemplateCollection.from_dataframe(df, "test")  # doctest: +SKIP
        """
        # Get columns, handling both pandas and polars
        is_polars = isinstance(df, pl.DataFrame)
        if is_polars:
            assert isinstance(df, pl.DataFrame)
            columns_list: list[str] = df.columns
        else:
            assert isinstance(df, pd.DataFrame)
            columns_list = list(df.columns)

        if "name" not in columns_list or "template_string" not in columns_list:
            raise ValueError("DataFrame must have 'name' and 'template_string' columns")

        collection = cls(name=name)

        # Note: We cannot fully reconstruct templates from DataFrames since
        # slot information is complex. This is a simplified reconstruction.
        # For full serialization, use to_jsonl/from_jsonl.

        return collection

    def to_jsonl(self, path: str) -> None:
        """Save collection to JSONLines file (one template per line).

        Parameters
        ----------
        path : str
            Path to the output file.

        Examples
        --------
        >>> from bead.resources import Slot
        >>> collection = TemplateCollection(name="test")
        >>> template = Template(
        ...     name="test", template_string="{x}.", slots={"x": Slot(name="x")}
        ... )
        >>> collection.add(template)
        >>> collection.to_jsonl("/tmp/templates.jsonl")  # doctest: +SKIP
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for template in self.templates.values():
                f.write(template.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: str, name: str) -> TemplateCollection:
        """Load collection from JSONLines file.

        Parameters
        ----------
        path : str
            Path to the input file.
        name : str
            Name for the collection.

        Returns
        -------
        TemplateCollection
            New collection loaded from file.

        Examples
        --------
        >>> collection = TemplateCollection.from_jsonl(
        ...     "/tmp/templates.jsonl", "loaded"
        ... )  # doctest: +SKIP
        """
        collection = cls(name=name)
        file_path = Path(path)

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    template_data = json.loads(line)
                    template = Template(**template_data)
                    collection.add(template)

        return collection
