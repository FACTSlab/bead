"""Repository pattern for data access with optional caching.

This module provides a generic Repository class that implements CRUD operations
for Pydantic models, with optional in-memory caching for efficient access.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

from bead.data.serialization import (
    append_jsonlines,
    read_jsonlines,
    write_jsonlines,
)


class Repository[T: BaseModel]:
    """Generic repository for CRUD operations on Pydantic models.

    Provides create, read, update, delete operations with JSONLines file storage
    and optional in-memory caching for efficient data access.

    Type Parameters
    ---------------
    T : BaseModel
        Pydantic model type this repository manages

    Parameters
    ----------
    model_class : type[T]
        The Pydantic model class this repository manages
    storage_path : Path
        Path to the JSONLines file for persistent storage
    use_cache : bool, optional
        Whether to use in-memory caching (default: True)

    Attributes
    ----------
    model_class : type[T]
        The Pydantic model class
    storage_path : Path
        Path to storage file
    use_cache : bool
        Whether caching is enabled
    cache : dict[UUID, T]
        In-memory cache of objects by ID

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.data.base import BeadBaseModel
    >>> class MyModel(BeadBaseModel):
    ...     name: str
    >>> repo = Repository[MyModel](
    ...     model_class=MyModel,
    ...     storage_path=Path("data/models.jsonl"),
    ...     use_cache=True
    ... )
    >>> obj = MyModel(name="test")
    >>> repo.add(obj)
    >>> loaded = repo.get(obj.id)
    >>> loaded.name
    'test'
    >>> repo.count()
    1
    """

    def __init__(
        self, model_class: type[T], storage_path: Path, use_cache: bool = True
    ) -> None:
        self.model_class = model_class
        self.storage_path = storage_path
        self.use_cache = use_cache
        self.cache: dict[UUID, T] = {}

        # Load cache on init if enabled and file exists
        if self.use_cache and self.storage_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        """Load all objects from storage into cache.

        This is a private method called during initialization if caching is
        enabled and the storage file exists.
        """
        objects = read_jsonlines(self.storage_path, self.model_class)
        self.cache = {obj.id: obj for obj in objects}  # type: ignore[attr-defined]

    def get(self, object_id: UUID) -> T | None:
        """Get object by ID.

        Parameters
        ----------
        object_id : UUID
            ID of the object to retrieve

        Returns
        -------
        T | None
            The object if found, None otherwise

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> obj = MyModel(name="test")
        >>> repo.add(obj)
        >>> loaded = repo.get(obj.id)
        >>> loaded is not None
        True
        """
        if self.use_cache:
            return self.cache.get(object_id)
        else:
            # Scan file for object
            if not self.storage_path.exists():
                return None
            objects = read_jsonlines(self.storage_path, self.model_class)
            for obj in objects:
                if obj.id == object_id:  # type: ignore[attr-defined]
                    return obj
            return None

    def get_all(self) -> list[T]:
        """Get all objects.

        Returns
        -------
        list[T]
            List of all objects in the repository

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> repo.add(MyModel(name="test1"))
        >>> repo.add(MyModel(name="test2"))
        >>> len(repo.get_all())
        2
        """
        if self.use_cache:
            return list(self.cache.values())
        else:
            if not self.storage_path.exists():
                return []
            return read_jsonlines(self.storage_path, self.model_class)

    def add(self, obj: T) -> None:
        """Add single object to repository.

        Appends the object to the storage file and updates cache if enabled.

        Parameters
        ----------
        obj : T
            Object to add

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> obj = MyModel(name="test")
        >>> repo.add(obj)
        >>> repo.exists(obj.id)
        True
        """
        # Create parent directories if needed
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        append_jsonlines([obj], self.storage_path)

        # Update cache
        if self.use_cache:
            self.cache[obj.id] = obj  # type: ignore[attr-defined]

    def add_many(self, objects: list[T]) -> None:
        """Add multiple objects to repository.

        Appends all objects to the storage file and updates cache if enabled.

        Parameters
        ----------
        objects : list[T]
            List of objects to add

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> objs = [MyModel(name="test1"), MyModel(name="test2")]
        >>> repo.add_many(objs)
        >>> repo.count()
        2
        """
        if not objects:
            return

        # Create parent directories if needed
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        append_jsonlines(objects, self.storage_path)

        # Update cache
        if self.use_cache:
            for obj in objects:
                self.cache[obj.id] = obj  # type: ignore[attr-defined]

    def update(self, obj: T) -> None:
        """Update existing object.

        Rewrites the entire storage file with the updated object.

        Parameters
        ----------
        obj : T
            Object to update (must have existing ID)

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> obj = MyModel(name="test")
        >>> repo.add(obj)
        >>> obj.name = "updated"
        >>> repo.update(obj)
        >>> loaded = repo.get(obj.id)
        >>> loaded.name
        'updated'
        """
        # Update in cache
        if self.use_cache:
            self.cache[obj.id] = obj  # type: ignore[attr-defined]

        # Rewrite file
        objects = list(self.cache.values()) if self.use_cache else self.get_all()
        # Replace the object in the list
        objects = [o if o.id != obj.id else obj for o in objects]  # type: ignore[attr-defined]

        # Create parent directories if needed
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        write_jsonlines(objects, self.storage_path)

    def delete(self, object_id: UUID) -> None:
        """Delete object by ID.

        Rewrites the entire storage file without the deleted object.

        Parameters
        ----------
        object_id : UUID
            ID of object to delete

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> obj = MyModel(name="test")
        >>> repo.add(obj)
        >>> repo.delete(obj.id)
        >>> repo.exists(obj.id)
        False
        """
        # Remove from cache
        if self.use_cache:
            self.cache.pop(object_id, None)

        # Rewrite file without the object
        objects = list(self.cache.values()) if self.use_cache else self.get_all()
        objects = [o for o in objects if o.id != object_id]  # type: ignore[attr-defined]

        if objects:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonlines(objects, self.storage_path)
        elif self.storage_path.exists():
            # If no objects left, delete the file
            self.storage_path.unlink()

    def exists(self, object_id: UUID) -> bool:
        """Check if object exists.

        Parameters
        ----------
        object_id : UUID
            ID of object to check

        Returns
        -------
        bool
            True if object exists, False otherwise

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> obj = MyModel(name="test")
        >>> repo.add(obj)
        >>> repo.exists(obj.id)
        True
        """
        return self.get(object_id) is not None

    def count(self) -> int:
        """Count objects in repository.

        Returns
        -------
        int
            Number of objects

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> repo.count()
        0
        >>> repo.add(MyModel(name="test"))
        >>> repo.count()
        1
        """
        if self.use_cache:
            return len(self.cache)
        else:
            if not self.storage_path.exists():
                return 0
            return len(read_jsonlines(self.storage_path, self.model_class))

    def clear(self) -> None:
        """Clear all objects and delete storage file.

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"))
        >>> repo.add(MyModel(name="test"))
        >>> repo.clear()
        >>> repo.count()
        0
        """
        # Clear cache
        self.cache.clear()

        # Delete file
        if self.storage_path.exists():
            self.storage_path.unlink()

    def rebuild_cache(self) -> None:
        """Rebuild cache from storage.

        Reloads all objects from storage into the cache. Useful if the storage
        file was modified externally.

        Examples
        --------
        >>> repo = Repository[MyModel](MyModel, Path("data.jsonl"), use_cache=True)
        >>> repo.rebuild_cache()
        """
        if not self.storage_path.exists():
            self.cache.clear()
        else:
            self._load_cache()
