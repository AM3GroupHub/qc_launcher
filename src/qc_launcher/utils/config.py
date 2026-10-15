"""Configuration helpers: deprecated-key aliases shared by the CLI, drivers and tasks."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any


class ConfigDeprecationWarning(UserWarning):
    """A configuration key uses a deprecated spelling that is still accepted."""


# section -> canonical key -> deprecated spellings that are still accepted
DEPRECATED_KEYS: dict[str, dict[str, tuple[str, ...]]] = {
    "driver": {"threads": ("thread",)},
    "driver.solvent": {"solvent_descriptors": ("solvent_descriptor",)},
    "opt": {"delta0": ("delta",)},
    "md": {"tdamp": ("dtamp",)},
    "neb": {"max_neb_steps": ("steps",)},
    "gsm": {"num_nodes": ("max_nodes",), "max_gsm_steps": ("max_cycles",)},
}


def rename_deprecated_keys(config: Mapping[str, Any] | None, section: str) -> dict[str, Any]:
    """Return a copy of ``config`` with deprecated keys of ``section`` renamed to their canonical names.

    Each hit emits a :class:`ConfigDeprecationWarning`. When both spellings are present the canonical
    key wins and the deprecated one is dropped. Unknown sections are copied unchanged.
    """
    result = dict(config or {})
    for canonical, old_names in DEPRECATED_KEYS.get(section, {}).items():
        for old in old_names:
            if old not in result:
                continue
            value = result.pop(old)
            if canonical in result:
                warnings.warn(
                    f"[{section}] both '{old}' (deprecated) and '{canonical}' are set; using '{canonical}'.",
                    ConfigDeprecationWarning,
                )
            else:
                warnings.warn(
                    f"[{section}] key '{old}' is deprecated, use '{canonical}' instead.",
                    ConfigDeprecationWarning,
                )
                result[canonical] = value
    return result
