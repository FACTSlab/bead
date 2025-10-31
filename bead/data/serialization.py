"""JSONLines serialization utilities for bead package.

This module provides functions for reading, writing, streaming, and appending
Pydantic models to/from JSONLines format files. JSONLines is a convenient format
for storing multiple JSON objects, with one object per line.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence


class SerializationError(Exception):
    """Exception raised when serialization to JSONLines fails.

    This exception is raised when writing Pydantic objects to JSONLines
    format encounters an error, such as file I/O issues or validation failures.
    """

    pass


class DeserializationError(Exception):
    """Exception raised when deserialization from JSONLines fails.

    This exception is raised when reading JSONLines format into Pydantic objects
    encounters an error, such as file not found, invalid JSON, or validation failures.
    """

    pass


def write_jsonlines[T: BaseModel](
    objects: Sequence[T],
    path: Path | str,
    validate: bool = True,
    append: bool = False,
) -> None:
    """Write Pydantic objects to JSONLines file.

    Serializes a sequence of Pydantic model instances to a JSONLines file,
    with one JSON object per line. Each object is validated before writing
    if validate=True.

    Parameters
    ----------
    objects : Sequence[T]
        Sequence of Pydantic model instances to serialize
    path : Path | str
        Path to the output file
    validate : bool, optional
        Whether to validate objects before writing (default: True)
    append : bool, optional
        Whether to append to existing file or overwrite (default: False)

    Raises
    ------
    SerializationError
        If writing fails due to I/O error or validation failure

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.data.base import BeadBaseModel
    >>> class TestModel(BeadBaseModel):
    ...     name: str
    >>> objects = [TestModel(name="test1"), TestModel(name="test2")]
    >>> write_jsonlines(objects, Path("output.jsonl"))  # doctest: +SKIP
    """
    path = Path(path)
    mode = "a" if append else "w"

    try:
        with path.open(mode, encoding="utf-8") as f:
            for obj in objects:
                # model_dump_json() handles validation if needed
                json_str = obj.model_dump_json()
                f.write(json_str + "\n")
    except (OSError, ValidationError) as e:
        raise SerializationError(f"Failed to write to {path}: {e}") from e


def read_jsonlines[T: BaseModel](
    path: Path | str,
    model_class: type[T],
    validate: bool = True,
    skip_errors: bool = False,
) -> list[T]:
    """Read JSONLines file into list of Pydantic objects.

    Deserializes a JSONLines file into a list of Pydantic model instances.
    Each line should contain a valid JSON object. Empty lines are skipped.

    Parameters
    ----------
    path : Path | str
        Path to the input file
    model_class : type[T]
        Pydantic model class to deserialize into
    validate : bool, optional
        Whether to validate objects during parsing (default: True)
    skip_errors : bool, optional
        Whether to skip invalid lines or raise error (default: False)

    Returns
    -------
    list[T]
        List of deserialized Pydantic objects

    Raises
    ------
    DeserializationError
        If reading fails due to file not found, invalid JSON, or validation failure
        (unless skip_errors=True)

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.data.base import BeadBaseModel
    >>> class TestModel(BeadBaseModel):
    ...     name: str
    >>> objects = read_jsonlines(Path("input.jsonl"), TestModel)  # doctest: +SKIP
    """
    path = Path(path)
    objects: list[T] = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    obj = model_class.model_validate_json(line)
                    objects.append(obj)
                except ValidationError as e:
                    if skip_errors:
                        continue
                    raise DeserializationError(
                        f"Failed to parse line {line_num} in {path}: {e}"
                    ) from e
    except OSError as e:
        raise DeserializationError(f"Failed to read from {path}: {e}") from e

    return objects


def stream_jsonlines[T: BaseModel](
    path: Path | str,
    model_class: type[T],
    validate: bool = True,
) -> Iterator[T]:
    """Stream JSONLines file as iterator of Pydantic objects.

    Memory-efficient iterator that yields Pydantic model instances one at a time
    from a JSONLines file. Useful for processing large files without loading
    everything into memory.

    Parameters
    ----------
    path : Path | str
        Path to the input file
    model_class : type[T]
        Pydantic model class to deserialize into
    validate : bool, optional
        Whether to validate objects during parsing (default: True)

    Yields
    ------
    T
        Pydantic model instances one at a time

    Raises
    ------
    DeserializationError
        If reading fails due to file not found, invalid JSON, or validation failure

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.data.base import BeadBaseModel
    >>> class TestModel(BeadBaseModel):
    ...     name: str
    >>> for obj in stream_jsonlines(Path("input.jsonl"), TestModel):  # doctest: +SKIP
    ...     print(obj.name)
    """
    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    obj = model_class.model_validate_json(line)
                    yield obj
                except ValidationError as e:
                    raise DeserializationError(
                        f"Failed to parse line {line_num} in {path}: {e}"
                    ) from e
    except OSError as e:
        raise DeserializationError(f"Failed to read from {path}: {e}") from e


def append_jsonlines[T: BaseModel](
    objects: Sequence[T],
    path: Path | str,
    validate: bool = True,
) -> None:
    """Append Pydantic objects to existing JSONLines file.

    Convenience wrapper around write_jsonlines with append=True. Adds objects
    to the end of an existing JSONLines file, or creates a new file if it
    doesn't exist.

    Parameters
    ----------
    objects : Sequence[T]
        Sequence of Pydantic model instances to serialize
    path : Path | str
        Path to the output file
    validate : bool, optional
        Whether to validate objects before writing (default: True)

    Raises
    ------
    SerializationError
        If appending fails due to I/O error or validation failure

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.data.base import BeadBaseModel
    >>> class TestModel(BeadBaseModel):
    ...     name: str
    >>> objects = [TestModel(name="test3"), TestModel(name="test4")]
    >>> append_jsonlines(objects, Path("output.jsonl"))  # doctest: +SKIP
    """
    write_jsonlines(objects, path, validate=validate, append=True)
