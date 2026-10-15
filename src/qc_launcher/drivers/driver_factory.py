from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from ase import Atoms

from qc_launcher.utils.config import rename_deprecated_keys
from qc_launcher.utils.optional import is_missing_package, missing_optional_dependency

from .base_driver import BaseDriver

# driver name -> (module, class, optional (import name, extra name))
_DRIVER_MAP: dict[str, tuple[str, str, tuple[str, str] | None]] = {
    "pyscf": ("pyscf_driver", "PySCFDriver", None),
    "mace": ("mace_driver", "MACEDriver", ("mace", "mace")),
    "uma": ("uma_driver", "UMADriver", ("fairchem", "fairchem")),
    "tblite": ("tblite_driver", "TBLiteDriver", ("tblite", "tblite")),
}


def get_driver(name: str, atoms: Atoms | None, config: Mapping[str, Any], verbose: bool = True) -> BaseDriver:
    """Instantiate the driver called ``name`` (case-insensitive) with the ``driver`` config block."""
    key = name.lower()
    if key not in _DRIVER_MAP:
        raise ValueError(f"Unknown driver '{name}'. Available drivers: {', '.join(sorted(_DRIVER_MAP))}.")
    module_name, class_name, optional = _DRIVER_MAP[key]
    try:
        module = import_module(f".{module_name}", package=__package__)
    except ModuleNotFoundError as exc:
        if optional is not None and is_missing_package(exc, optional[0]):
            raise missing_optional_dependency(f"driver '{key}'", optional[1], package_name=optional[0]) from exc
        raise
    driver_class = getattr(module, class_name)
    return driver_class(atoms, rename_deprecated_keys(config, "driver"), verbose=verbose)


def driver_from_config(driver_block: Mapping[str, Any], atoms: Atoms | None, *, verbose: bool = True) -> BaseDriver:
    """Build a driver from a YAML ``driver`` block (``name`` plus driver-specific keys)."""
    block = dict(driver_block)
    try:
        name = block.pop("name")
    except KeyError:
        raise ValueError("The `driver` block requires a `name` key.") from None
    return get_driver(name, atoms, block, verbose=verbose)
