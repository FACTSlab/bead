"""Bead-aware interpolation resolvers.

Importing this module registers ``${bead.path:rel}`` against the
:mod:`didactic.settings` resolver registry, so any composition run
after :mod:`bead.config` is imported can use it. The resolver reads
the in-flight tree through :func:`didactic.settings.lookup`, which
shares the engine's cycle detection: a reference cycle through
``bead.path`` is reported as an :class:`InterpolationError` rather
than recursing until the interpreter gives up.

Resolvers
---------
- ``${bead.path:rel}``: join ``rel`` against the value at
  ``paths.data_dir`` in the composed config. Convenient shorthand
  for ``${paths.data_dir}/rel``.

Post-validation anchor resolution (``${bead.anchor:name[,attr]}``)
needs a validated :class:`AnnotationProtocol` and therefore lives at
post-validation time. See :func:`resolve_anchor_attributes`.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import cast

from didactic.settings import InterpolationError, lookup, register_resolver


def _bead_path(*args: str) -> str:
    """``${bead.path:rel}``: join ``rel`` against ``paths.data_dir``.

    ``paths.data_dir`` is read through :func:`didactic.settings.lookup`,
    so it is itself interpolated first and a missing key raises the
    engine's own unresolved-reference error. The result uses forward
    slashes (:class:`~pathlib.PurePosixPath`); callers wrap it in
    :class:`pathlib.Path` as needed.
    """
    if not args or not args[0]:
        raise InterpolationError("bead.path requires a relative path")
    data_dir = lookup("paths.data_dir")
    if not isinstance(data_dir, str):
        raise InterpolationError(
            f"paths.data_dir resolved to {type(data_dir).__name__}, expected str"
        )
    return str(PurePosixPath(data_dir) / ",".join(args))


register_resolver("bead.path", _bead_path, replace=True)


# ---------------------------------------------------------------------------
# Post-validation anchor resolution
# ---------------------------------------------------------------------------


_ANCHOR_PATTERN: re.Pattern[str] = re.compile(r"\$\{bead\.anchor:([^}]+)\}")


def resolve_anchor_attributes(
    text: str,
    *,
    protocol: object,
) -> str:
    """Replace ``${bead.anchor:name[,attr]}`` references in ``text``.

    Used by application code after the protocol is materialized.
    ``attr`` defaults to ``"canonical_prompt"`` and may be any
    attribute name on :class:`~bead.protocol.SemanticAnchor`.

    Parameters
    ----------
    text : str
        Text containing ``${bead.anchor:name}`` or
        ``${bead.anchor:name,attr}`` expressions.
    protocol : AnnotationProtocol
        Validated protocol whose ``family_by_name(name).anchor`` is
        consulted.

    Returns
    -------
    str
        ``text`` with every recognized expression substituted.
    """

    def _replace(match: re.Match[str]) -> str:
        spec = match.group(1)
        if "," in spec:
            name, _, attr = spec.partition(",")
            name, attr = name.strip(), attr.strip()
        else:
            name, attr = spec.strip(), "canonical_prompt"
        family = cast("object", protocol.family_by_name(name))  # type: ignore[attr-defined]
        anchor = cast("object", family.anchor)  # type: ignore[attr-defined]
        return str(getattr(anchor, attr))

    return _ANCHOR_PATTERN.sub(_replace, text)
