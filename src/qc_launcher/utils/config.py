"""Configuration helpers: deprecated-key aliases shared by the CLI, drivers and tasks."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any


class ConfigDeprecationWarning(UserWarning):
    """A configuration key uses a deprecated spelling that is still accepted."""


# section -> canonical key -> deprecated spellings that are still accepted
DEPRECATED_KEYS: dict[str, dict[str, tuple[str, ...]]] = {
    "driver": {
        "threads": ("thread",),
        "verbosity": ("verbose",),
        "scf_max_iter": ("scf_max_cycle", "max_iter"),
        "dispersion": ("disp",),
        "numerical_hessian": ("numerical_hess",),
        "finite_diff_step": ("finite_diff_eps",),
        "electronic_temperature": ("etemp",),
        "solvation": ("solvent",),
    },
    "driver.solvation": {
        "model": ("method",),
        "epsilon": ("eps",),
        "solvent_descriptors": ("solvent_descriptor",),
    },
    "opt": {"delta0": ("delta",)},
    "freq": {"bav": ("Bav",)},
    "irc": {"trajectory": ("irc_trajectory",), "outputfile": ("irc_output",), "max_steps": ("steps",)},
    "md": {"tdamp": ("dtamp",)},
    "neb": {"max_steps": ("max_neb_steps", "steps"), "endpoint_max_steps": ("max_opt_steps",)},
    "autoneb": {"max_steps": ("max_neb_steps",), "endpoint_max_steps": ("max_opt_steps",)},
    # gsm keys follow the pygsm fork's parameter names (X1X1010/pygsm, snake_case API)
    "gsm": {
        "nnodes": ("num_nodes", "max_nodes"),
        "max_iters": ("max_gsm_steps", "max_cycles"),
        "opt_steps": ("max_opt_steps",),
        "string_id": ("ID",),
        "conv_gmax": ("fmax",),
        "conv_ediff": ("ediff",),
        "linesearch": ("line_search",),
        "endpoint_max_steps": ("endpoint_opt_steps",),
    },
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
