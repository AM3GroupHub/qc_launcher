import importlib
from ase import Atoms

from .base_driver import BaseDriver


_DRIVER_MAP = {
    "pyscf": (".pyscf_driver", "PySCFDriver"),
    "mace": (".mace_driver", "MACEDriver"),
    "uma": (".uma_driver", "UMADriver"),
    "tblite": (".tblite_driver", "TBLiteDriver"),
}

def get_driver(name: str, atoms: Atoms, config: dict, verbose: bool = True) -> BaseDriver:
    """
    Factory function to get a driver instance by name.
    """
    name = name.lower()
    if name not in _DRIVER_MAP:
        raise ValueError(f"Unknown driver: {name}")
    module_name, class_name = _DRIVER_MAP[name]
    module = importlib.import_module(module_name, package=__package__)
    driver_class = getattr(module, class_name)
    return driver_class(atoms, config, verbose=verbose)
