from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

from qc_launcher.utils.optional import is_missing_package, missing_optional_dependency

try:
    from pysisyphus.calculators.Calculator import Calculator
except ModuleNotFoundError as exc:
    if is_missing_package(exc, "pysisyphus"):
        raise missing_optional_dependency(
            "pysisyphus calculator bridge",
            "pysisyphus",
        ) from exc
    raise

from .driver_factory import driver_from_config


class QCSisyphusCalc(Calculator):
    conf_key = "qc_launcher"

    def __init__(self, driver: dict, charge: int = None, mult: int = None, **kwargs) -> None:
        driver_obj = driver_from_config(driver, atoms=None, verbose=False)
        super().__init__(charge=0 if charge is None else charge, mult=1 if mult is None else mult, **kwargs)
        self.driver = driver_obj

    def prepare_atoms(self, atoms: list[str], coords: np.ndarray, **prepare_kwargs) -> Atoms:
        symbols = [symbol.capitalize() for symbol in atoms]  # ensure symbols are capitalized
        ase_atoms = Atoms(
            symbols=symbols,
            positions=coords.reshape(-1, 3) * Bohr,  # convert from Bohr to Angstrom
        )
        charge = self.charge
        mult = self.mult
        ase_atoms.info["charge"] = charge
        ase_atoms.info["multiplicity"] = mult
        return ase_atoms

    def get_energy(self, atoms: list[str], coords: np.ndarray, **prepare_kwargs) -> dict:
        ase_atoms = self.prepare_atoms(atoms, coords, **prepare_kwargs)
        return {"energy": self.driver.compute_energy(ase_atoms) / Hartree}

    def get_forces(self, atoms: list[str], coords: np.ndarray, **prepare_kwargs) -> dict:
        # pysisyphus expects the energy alongside the forces
        results = self.get_energy(atoms, coords, **prepare_kwargs)
        ase_atoms = self.prepare_atoms(atoms, coords, **prepare_kwargs)
        results["forces"] = self.driver.compute_forces(ase_atoms).ravel() * (Bohr / Hartree)
        return results

    def get_hessian(self, atoms: list[str], coords: np.ndarray, **prepare_kwargs) -> dict:
        results = self.get_energy(atoms, coords, **prepare_kwargs)
        ase_atoms = self.prepare_atoms(atoms, coords, **prepare_kwargs)
        hessian = self.driver.compute_hessian(atoms=ase_atoms, use_cache=True, hess_format="ase")
        results["hessian"] = hessian * (Bohr**2 / Hartree)  # eV/Ang^2 -> Eh/Bohr^2
        return results
