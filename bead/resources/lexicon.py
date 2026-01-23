"""Lexicon management for collections of lexical items.

This module provides the Lexicon class for managing, querying, and manipulating
collections of lexical items. It supports filtering, searching, merging, and
conversion to/from pandas and polars DataFrames.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pandas as pd
import polars as pl
from pydantic import Field

from bead.data.base import BeadBaseModel
from bead.data.language_codes import LanguageCode
from bead.resources.lexical_item import LexicalItem

# Type alias for supported DataFrame types
DataFrame = pd.DataFrame | pl.DataFrame


def _empty_str_list() -> list[str]:
    """Create an empty string list."""
    return []


def _empty_item_dict() -> dict[UUID, LexicalItem]:
    """Create an empty item dictionary."""
    return {}


class Lexicon(BeadBaseModel):
    """A collection of lexical items with operations for filtering and analysis.

    The Lexicon class manages collections of LexicalItem objects and provides
    methods for:
    - Adding and removing items (CRUD operations)
    - Filtering by properties, features, and attributes
    - Searching by text
    - Merging with other lexicons
    - Converting to/from pandas and polars DataFrames
    - Serialization to JSONLines

    Attributes
    ----------
    name : str
        Name of the lexicon.
    description : str | None
        Optional description of the lexicon's purpose.
    language_code : LanguageCode | None
        ISO 639-1 (2-letter) or ISO 639-3 (3-letter) language code.
        Examples: "en", "eng", "ko", "kor", "zu", "zul".
        Automatically validated and normalized to lowercase.
    items : dict[UUID, LexicalItem]
        Dictionary of items indexed by their UUIDs.
    tags : list[str]
        Tags for categorizing the lexicon.

    Examples
    --------
    >>> lexicon = Lexicon(name="verbs")
    >>> item = LexicalItem(lemma="walk", pos="VERB")
    >>> lexicon.add(item)
    >>> len(lexicon)
    1
    >>> verbs = lexicon.filter_by_pos("VERB")
    >>> len(verbs.items)
    1
    """

    name: str
    description: str | None = None
    language_code: LanguageCode | None = None
    items: dict[UUID, LexicalItem] = Field(default_factory=_empty_item_dict)
    tags: list[str] = Field(default_factory=_empty_str_list)

    def __len__(self) -> int:
        """Return number of items in lexicon.

        Returns
        -------
        int
            Number of items in the lexicon.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> len(lexicon)
        0
        >>> lexicon.add(LexicalItem(lemma="test"))
        >>> len(lexicon)
        1
        """
        return len(self.items)

    def __iter__(self) -> Iterator[LexicalItem]:  # type: ignore[override]
        """Iterate over items in lexicon.

        Returns
        -------
        Iterator[LexicalItem]
            Iterator over lexical items.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> lexicon.add(LexicalItem(lemma="walk"))
        >>> lexicon.add(LexicalItem(lemma="run"))
        >>> [item.lemma for item in lexicon]
        ['walk', 'run']
        """
        return iter(self.items.values())

    def __contains__(self, item_id: UUID) -> bool:
        """Check if item ID is in lexicon.

        Parameters
        ----------
        item_id : UUID
            The item ID to check.

        Returns
        -------
        bool
            True if item ID exists in lexicon.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> item = LexicalItem(lemma="test")
        >>> lexicon.add(item)
        >>> item.id in lexicon
        True
        """
        return item_id in self.items

    def add(self, item: LexicalItem) -> None:
        """Add a lexical item to the lexicon.

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
        >>> lexicon = Lexicon(name="test")
        >>> item = LexicalItem(lemma="walk")
        >>> lexicon.add(item)
        >>> len(lexicon)
        1
        """
        if item.id in self.items:
            raise ValueError(f"Item with ID {item.id} already exists in lexicon")
        self.items[item.id] = item
        self.update_modified_time()

    def add_many(self, items: list[LexicalItem]) -> None:
        """Add multiple items to the lexicon.

        Parameters
        ----------
        items : list[LexicalItem]
            The items to add.

        Raises
        ------
        ValueError
            If any item with same ID already exists.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> items = [LexicalItem(lemma="walk"), LexicalItem(lemma="run")]
        >>> lexicon.add_many(items)
        >>> len(lexicon)
        2
        """
        for item in items:
            self.add(item)

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
        >>> lexicon = Lexicon(name="test")
        >>> item = LexicalItem(lemma="walk")
        >>> lexicon.add(item)
        >>> removed = lexicon.remove(item.id)
        >>> removed.lemma
        'walk'
        >>> len(lexicon)
        0
        """
        if item_id not in self.items:
            raise KeyError(f"Item with ID {item_id} not found in lexicon")
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
        >>> lexicon = Lexicon(name="test")
        >>> item = LexicalItem(lemma="walk")
        >>> lexicon.add(item)
        >>> retrieved = lexicon.get(item.id)
        >>> retrieved.lemma  # doctest: +SKIP
        'walk'
        >>> from uuid import uuid4
        >>> lexicon.get(uuid4()) is None
        True
        """
        return self.items.get(item_id)

    def filter(self, predicate: Callable[[LexicalItem], bool]) -> Lexicon:
        """Filter items by a predicate function.

        Creates a new lexicon containing only items that satisfy the predicate.

        Parameters
        ----------
        predicate : Callable[[LexicalItem], bool]
            Function that returns True for items to include.

        Returns
        -------
        Lexicon
            New lexicon with filtered items.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> lexicon.add(LexicalItem(lemma="walk", pos="VERB"))
        >>> lexicon.add(LexicalItem(lemma="dog", pos="NOUN"))
        >>> verbs = lexicon.filter(lambda item: item.pos == "VERB")
        >>> len(verbs.items)
        1
        """
        filtered = Lexicon(
            name=f"{self.name}_filtered",
            description=self.description,
            language_code=self.language_code,
            tags=self.tags.copy(),
        )
        filtered.items = {
            item_id: item for item_id, item in self.items.items() if predicate(item)
        }
        return filtered

    def filter_by_pos(self, pos: str) -> Lexicon:
        """Filter items by part of speech.

        Parameters
        ----------
        pos : str
            The part of speech to filter by.

        Returns
        -------
        Lexicon
            New lexicon with items matching the POS.

        Examples
        --------
        >>> lexicon = Lexicon(name="test", language_code="eng")
        >>> lexicon.add(LexicalItem(
        ...     lemma="walk", language_code="eng", features={"pos": "VERB"}
        ... ))
        >>> lexicon.add(LexicalItem(
        ...     lemma="dog", language_code="eng", features={"pos": "NOUN"}
        ... ))
        >>> verbs = lexicon.filter_by_pos("VERB")
        >>> len(verbs.items)
        1
        """
        return self.filter(
            lambda item: item.features.get("pos") is not None
            and item.features.get("pos") == pos
        )

    def filter_by_lemma(self, lemma: str) -> Lexicon:
        """Filter items by lemma (exact match).

        Parameters
        ----------
        lemma : str
            The lemma to filter by.

        Returns
        -------
        Lexicon
            New lexicon with items matching the lemma.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> lexicon.add(LexicalItem(lemma="walk"))
        >>> lexicon.add(LexicalItem(lemma="run"))
        >>> results = lexicon.filter_by_lemma("walk")
        >>> len(results.items)
        1
        """
        return self.filter(lambda item: item.lemma == lemma)

    def filter_by_feature(self, feature_name: str, feature_value: Any) -> Lexicon:
        """Filter items by a specific feature value.

        Parameters
        ----------
        feature_name : str
            The name of the feature.
        feature_value : Any
            The value to match.

        Returns
        -------
        Lexicon
            New lexicon with items having the specified feature value.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> lexicon.add(LexicalItem(lemma="walk", features={"tense": "present"}))
        >>> lexicon.add(LexicalItem(lemma="walked", features={"tense": "past"}))
        >>> present = lexicon.filter_by_feature("tense", "present")
        >>> len(present.items)
        1
        """
        return self.filter(
            lambda item: feature_name in item.features
            and item.features[feature_name] == feature_value
        )

    def filter_by_attribute(self, attr_name: str, attr_value: Any) -> Lexicon:
        """Filter items by a specific attribute value.

        Parameters
        ----------
        attr_name : str
            The name of the attribute.
        attr_value : Any
            The value to match.

        Returns
        -------
        Lexicon
            New lexicon with items having the specified attribute value.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> lexicon.add(LexicalItem(
        ...     lemma="walk", language_code="eng", features={"frequency": 1000}
        ... ))
        >>> lexicon.add(LexicalItem(
        ...     lemma="saunter", language_code="eng", features={"frequency": 10}
        ... ))
        >>> high_freq = lexicon.filter_by_attribute("frequency", 1000)
        >>> len(high_freq.items)
        1
        """
        return self.filter(
            lambda item: attr_name in item.features
            and item.features[attr_name] == attr_value
        )

    def search(self, query: str, field: str = "lemma") -> Lexicon:
        """Search for items containing query string in specified field.

        Parameters
        ----------
        query : str
            Search string (case-insensitive substring match).
        field : str
            Field to search in ("lemma", "pos", "form").

        Returns
        -------
        Lexicon
            New lexicon with matching items.

        Raises
        ------
        ValueError
            If field is not a valid searchable field.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> lexicon.add(LexicalItem(lemma="walk"))
        >>> lexicon.add(LexicalItem(lemma="run"))
        >>> results = lexicon.search("wa")
        >>> len(results.items)
        1
        """
        query_lower = query.lower()

        if field == "lemma":
            return self.filter(lambda item: query_lower in item.lemma.lower())
        elif field == "pos":
            return self.filter(
                lambda item: item.features.get("pos") is not None
                and query_lower in str(item.features.get("pos")).lower()
            )
        elif field == "form":
            return self.filter(
                lambda item: item.form is not None and query_lower in item.form.lower()
            )
        else:
            raise ValueError(
                f"Invalid field '{field}'. Must be 'lemma', 'pos', or 'form'."
            )

    def merge(
        self,
        other: Lexicon,
        strategy: Literal["keep_first", "keep_second", "error"] = "keep_first",
    ) -> Lexicon:
        """Merge with another lexicon.

        Parameters
        ----------
        other : Lexicon
            The lexicon to merge with.
        strategy : Literal["keep_first", "keep_second", "error"]
            How to handle duplicate IDs:
            - "keep_first": Keep item from self
            - "keep_second": Keep item from other
            - "error": Raise error on duplicates

        Returns
        -------
        Lexicon
            New merged lexicon.

        Raises
        ------
        ValueError
            If strategy is "error" and duplicates found.

        Examples
        --------
        >>> lex1 = Lexicon(name="lex1")
        >>> lex1.add(LexicalItem(lemma="walk"))
        >>> lex2 = Lexicon(name="lex2")
        >>> lex2.add(LexicalItem(lemma="run"))
        >>> merged = lex1.merge(lex2)
        >>> len(merged.items)
        2
        """
        # Check for duplicates if strategy is "error"
        if strategy == "error":
            duplicates = set(self.items.keys()) & set(other.items.keys())
            if duplicates:
                raise ValueError(
                    f"Duplicate item IDs found: {duplicates}. "
                    "Use strategy='keep_first' or 'keep_second' to resolve."
                )

        # Create merged lexicon
        # Use language_code from self, or other if self's is None
        language_code = self.language_code or other.language_code

        merged = Lexicon(
            name=f"{self.name}_merged",
            description=self.description,
            language_code=language_code,
            tags=list(set(self.tags + other.tags)),
        )

        # Add items based on strategy
        if strategy == "keep_first":
            merged.items = {**other.items, **self.items}
        elif strategy == "keep_second":
            merged.items = {**self.items, **other.items}
        else:  # strategy == "error" already handled above
            merged.items = {**self.items, **other.items}

        return merged

    def to_dataframe(
        self, backend: Literal["pandas", "polars"] = "pandas"
    ) -> DataFrame:
        """Convert lexicon to DataFrame.

        Parameters
        ----------
        backend : Literal["pandas", "polars"]
            DataFrame backend to use (default: "pandas").

        Returns
        -------
        DataFrame
            pandas or polars DataFrame with columns: id, lemma, pos, form,
            source, created_at, modified_at, plus separate columns for
            each feature and attribute.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> lexicon.add(LexicalItem(lemma="walk", pos="VERB"))
        >>> df = lexicon.to_dataframe()
        >>> "lemma" in df.columns
        True
        >>> "pos" in df.columns
        True
        """
        if not self.items:
            # Return empty DataFrame with expected columns
            columns = [
                "id",
                "lemma",
                "pos",
                "form",
                "source",
                "created_at",
                "modified_at",
            ]
            if backend == "pandas":
                return pd.DataFrame(columns=columns)
            else:
                schema: dict[str, type[pl.Utf8]] = dict.fromkeys(columns, pl.Utf8)
                return pl.DataFrame(schema=schema)

        rows = []
        for item in self.items.values():
            row = {
                "id": str(item.id),
                "lemma": item.lemma,
                "form": item.form,
                "language_code": item.language_code,
                "source": item.source,
                "created_at": item.created_at.isoformat(),
                "modified_at": item.modified_at.isoformat(),
            }

            # Add features with "feature_" prefix
            for key, value in item.features.items():
                row[f"feature_{key}"] = value

            rows.append(row)  # type: ignore[arg-type]

        if backend == "pandas":
            return pd.DataFrame(rows)
        else:
            return pl.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: DataFrame, name: str) -> Lexicon:
        """Create lexicon from DataFrame.

        Parameters
        ----------
        df : DataFrame
            pandas or polars DataFrame with at minimum a 'lemma' column.
        name : str
            Name for the lexicon.

        Returns
        -------
        Lexicon
            New lexicon created from DataFrame.

        Raises
        ------
        ValueError
            If DataFrame does not have a 'lemma' column.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"lemma": ["walk", "run"], "pos": ["VERB", "VERB"]})
        >>> lexicon = Lexicon.from_dataframe(df, "verbs")
        >>> len(lexicon.items)
        2
        """
        # Check if it's a polars DataFrame
        is_polars = isinstance(df, pl.DataFrame)

        # Get columns, handling both pandas and polars
        if is_polars:
            assert isinstance(df, pl.DataFrame)
            columns_list: list[str] = df.columns
        else:
            assert isinstance(df, pd.DataFrame)
            columns_list = list(df.columns)

        if "lemma" not in columns_list:
            raise ValueError("DataFrame must have a 'lemma' column")

        lexicon = cls(name=name)

        # Convert to dict format for iteration
        rows: list[dict[str, Any]]
        if is_polars:
            assert isinstance(df, pl.DataFrame)
            rows = df.to_dicts()
        else:
            assert isinstance(df, pd.DataFrame)
            rows = df.to_dict("records")  # type: ignore[assignment]

        for row in rows:
            # Extract base fields
            item_data: dict[str, Any] = {"lemma": row["lemma"]}

            # Helper function to check for null values
            def is_not_null(value: Any) -> bool:
                if is_polars:
                    return value is not None
                else:
                    return pd.notna(value)  # type: ignore[no-any-return]

            # Handle language_code (required field)
            if "language_code" in row and is_not_null(row["language_code"]):
                item_data["language_code"] = row["language_code"]
            else:
                item_data["language_code"] = "eng"  # Default to English

            if "form" in row and is_not_null(row["form"]):
                item_data["form"] = row["form"]
            if "source" in row and is_not_null(row["source"]):
                item_data["source"] = row["source"]

            # Extract features (columns with "feature_" prefix, "pos", or "attr_" prefix)  # noqa: E501
            features: dict[str, Any] = {}
            if "pos" in row and is_not_null(row["pos"]):
                features["pos"] = row["pos"]
            for col in columns_list:
                if col.startswith("feature_") and is_not_null(row[col]):
                    feature_name: str = col[len("feature_") :]
                    features[feature_name] = row[col]
                elif col.startswith("attr_") and is_not_null(row[col]):
                    attr_name: str = col[len("attr_") :]
                    features[attr_name] = row[col]

            if features:
                item_data["features"] = features

            item = LexicalItem(**item_data)
            lexicon.add(item)

        return lexicon

    def to_jsonl(self, path: str) -> None:
        """Save lexicon to JSONLines file (one item per line).

        Parameters
        ----------
        path : str
            Path to the output file.

        Examples
        --------
        >>> lexicon = Lexicon(name="test")
        >>> lexicon.add(LexicalItem(lemma="walk"))
        >>> lexicon.to_jsonl("/tmp/lexicon.jsonl")  # doctest: +SKIP
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for item in self.items.values():
                f.write(item.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: str, name: str) -> Lexicon:
        """Load lexicon from JSONLines file.

        Parameters
        ----------
        path : str
            Path to the input file.
        name : str
            Name for the lexicon.

        Returns
        -------
        Lexicon
            New lexicon loaded from file.

        Examples
        --------
        >>> lexicon = Lexicon.from_jsonl(
        ...     "/tmp/lexicon.jsonl", "loaded"
        ... )  # doctest: +SKIP
        """
        lexicon = cls(name=name)
        file_path = Path(path)

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item_data = json.loads(line)
                    item = LexicalItem(**item_data)
                    lexicon.add(item)

        return lexicon
