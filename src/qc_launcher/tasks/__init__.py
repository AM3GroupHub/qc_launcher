from __future__ import annotations

from importlib import import_module


TASK_MODULES = {
    "run_sp": ".sp",
    "run_grad": ".sp",
    "run_opt": ".opt",
    "run_freq": ".freq",
    "run_irc": ".irc",
    "run_neb": ".neb",
    "run_gsm": ".gsm",
    "run_md": ".md",
}

__all__ = list(TASK_MODULES)


def __getattr__(name: str):
    if name not in TASK_MODULES:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = import_module(TASK_MODULES[name], package=__name__)
    return getattr(module, name)
