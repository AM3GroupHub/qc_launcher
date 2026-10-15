"""Task runners. Use :func:`load_task` so optional dependencies (sella, pygsm) are imported lazily."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from qc_launcher.utils.optional import is_missing_package, missing_optional_dependency

# task name -> (module under qc_launcher.tasks, runner function)
_TASKS: dict[str, tuple[str, str]] = {
    "sp": ("sp", "run_sp"),
    "grad": ("sp", "run_grad"),
    "opt": ("opt", "run_opt"),
    "freq": ("freq", "run_freq"),
    "irc": ("irc", "run_irc"),
    "neb": ("neb", "run_neb"),
    "autoneb": ("autoneb", "run_autoneb"),
    "gsm": ("gsm", "run_gsm"),
    "md": ("md", "run_md"),
}

# task name -> (import name, extra name, install hint)
_OPTIONAL_DEPS: dict[str, tuple[str, str, str | None]] = {
    "opt": ("sella", "sella", "pip install sella"),
    "irc": ("sella", "sella", "pip install sella"),
    "gsm": ("pygsm", "gsm", None),
}

__all__ = ["load_task"]


def load_task(name: str) -> Callable[..., Any]:
    """Import and return the ``run_<name>`` function of a task module."""
    if name not in _TASKS:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {', '.join(sorted(_TASKS))}.")
    module_name, func_name = _TASKS[name]
    try:
        module = import_module(f"qc_launcher.tasks.{module_name}")
    except ModuleNotFoundError as exc:
        dep = _OPTIONAL_DEPS.get(name)
        if dep is not None and is_missing_package(exc, dep[0]):
            raise missing_optional_dependency(
                f"`{name}` task", dep[1], package_name=dep[0], install_hint=dep[2]
            ) from exc
        raise
    return getattr(module, func_name)
