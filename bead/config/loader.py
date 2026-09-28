"""Bead-specific entrypoint to the composition engine.

A thin wrapper around :func:`didactic.settings.compose` that binds
the schema to :class:`~bead.config.config.BeadConfig` and starts from
the profile defaults declared in :mod:`bead.config.profiles`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from didactic.settings import ConfigValue, compose

# Importing this module registers the ${bead.path:...} resolver
# against the didactic.settings interpolation registry.
from bead.config import resolvers as _bead_resolvers
from bead.config.config import BeadConfig
from bead.config.profiles import get_profile

_ = _bead_resolvers


def load_config(
    config_path: Path | str | None = None,
    *,
    profile: str = "default",
    overrides: Sequence[str] = (),
    extra: Sequence[Path | str] = (),
    **kw_overrides: ConfigValue,
) -> BeadConfig:
    """Compose a :class:`BeadConfig` from a profile, file, and overrides.

    Precedence (lowest to highest):

      1. Profile defaults (``bead.config.profiles.get_profile``), the
         base layer.
      2. Each fragment listed in the primary file's ``defaults: [...]``
         key, in order.
      3. The primary file body.
      4. Each ``extra`` overlay file, in order.
      5. ``overrides``: dotted-key ``key=value`` strings, whose values
         are read by the leaf's declared type.
      6. ``kw_overrides``: the ``key__sub=value`` keyword form. Each
         key is rewritten as ``"key.sub"`` and the value is merged as
         given, after ``overrides``.

    Every layer, the profile included, is checked against
    :class:`BeadConfig` as it is merged, so an unknown key is refused
    as a :class:`~didactic.settings.ConfigError` naming its dotted
    path and the layer that set it. Interpolation is resolved last;
    the resolved tree is validated as a :class:`BeadConfig`.

    Parameters
    ----------
    config_path : Path | str | None, optional
        Primary YAML or TOML file.
    profile : str, optional
        Profile name (``"default"``, ``"dev"``, ``"prod"``,
        ``"test"``).
    overrides : Sequence[str], optional
        CLI-style overrides (``["paths.data_dir=/tmp"]``).
    extra : Sequence[Path | str], optional
        Additional overlay files merged after the primary YAML.
    **kw_overrides : ConfigValue
        Keyword overrides; ``__`` separates nested levels.

    Returns
    -------
    BeadConfig
        Fully composed and validated configuration.
    """
    profile_dict: dict[str, ConfigValue] = json.loads(
        get_profile(profile).model_dump_json()
    )
    return compose(
        config_path,
        schema=BeadConfig,
        base=profile_dict,
        overlays=extra,
        overrides=[
            *overrides,
            *((key.replace("__", "."), value) for key, value in kw_overrides.items()),
        ],
    )
