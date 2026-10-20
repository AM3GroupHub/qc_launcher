"""Task runners. Use :func:`load_task` so optional dependencies (sella, geomeTRIC, pygsm) are imported lazily.

The ``opt`` task dispatches on its ``optimizer`` key to :mod:`.opt_sella` (default) or :mod:`.opt_tric` (geomeTRIC)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from qc_launcher.utils.optional import is_missing_package, missing_optional_dependency

# task name -> (module under qc_launcher.tasks, runner function)
_TASKS: dict[str, tuple[str, str]] = {
    "sp": ("sp", "run_sp"),
    "grad": ("sp", "run_grad"),
    "opt": ("opt_sella", "run_opt_sella"),  # replaced by the ``optimizer`` dispatch in load_task
    "freq": ("freq", "run_freq"),
    "irc": ("irc", "run_irc"),
    "neb": ("neb", "run_neb"),
    "autoneb": ("autoneb", "run_autoneb"),
    "gsm": ("gsm", "run_gsm"),
    "md": ("md", "run_md"),
}

# opt.optimizer -> (module under qc_launcher.tasks, runner function)
OPTIMIZERS: dict[str, tuple[str, str]] = {
    "sella": ("opt_sella", "run_opt_sella"),
    "geometric": ("opt_tric", "run_opt_geometric"),
}

# task module -> (import name, extra name, install hint)
_OPTIONAL_DEPS: dict[str, tuple[str, str, str | None]] = {
    "opt_sella": ("sella", "sella", "pip install sella"),
    "opt_tric": ("geometric", "geometric", "pip install geometric"),
    "irc": ("sella", "sella", "pip install sella"),
    "gsm": ("pygsm", "gsm", None),
}

__all__ = ["OPTIMIZERS", "load_task", "run_opt"]


def _load(module_name: str, func_name: str, label: str) -> Callable[..., Any]:
    try:
        module = import_module(f"qc_launcher.tasks.{module_name}")
    except ModuleNotFoundError as exc:
        dep = _OPTIONAL_DEPS.get(module_name)
        if dep is not None and is_missing_package(exc, dep[0]):
            raise missing_optional_dependency(label, dep[1], package_name=dep[0], install_hint=dep[2]) from exc
        raise
    return getattr(module, func_name)


def run_opt(driver: Any, config: dict, filename: str = "molecule") -> Any:
    """``optimizer`` (sella | geometric) selects the engine; ``ts``, ``calc_hess``, ``trajectory``, ``outputfile``,
    ``constraints``, ``max_steps`` and the convergence criteria mean the same for both. See ``opt_sella.run_opt_sella``
    and ``opt_tric.run_opt_geometric`` for the engine-specific keys."""
    optimizer = str(config.get("optimizer", "sella")).lower()
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Use one of {tuple(OPTIMIZERS)}.")
    module_name, func_name = OPTIMIZERS[optimizer]
    return _load(module_name, func_name, f"`opt` task with optimizer '{optimizer}'")(driver, config, filename)


def load_task(name: str) -> Callable[..., Any]:
    """Import and return the ``run_<name>`` function of a task module (``opt`` returns the optimizer dispatch)."""
    if name not in _TASKS:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {', '.join(sorted(_TASKS))}.")
    if name == "opt":
        return run_opt
    module_name, func_name = _TASKS[name]
    return _load(module_name, func_name, f"`{name}` task")
