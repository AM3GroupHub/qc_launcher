from __future__ import annotations

from typing import List

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

from .driver_factory import get_driver


class QCSisyphusCalc(Calculator):
    conf_key = "qc_launcher"
    def __init__(
        self,
        driver: dict,
        charge: int = None,
        mult: int = None,
        **kwargs
    ) -> None:
        # load driver
        driver_config = dict(driver)
        driver_name = driver_config.pop("name")
        driver_obj = get_driver(
            name=driver_name, atoms=None, config=driver_config, verbose=False,
        ) 
        # read charge and multiplicity from driver.atoms.info, default to 0 and 1 if not present
        driver_info = {} if driver_obj.atoms is None else driver_obj.atoms.info
        charge = driver_info.get("charge", 0) if charge is None else charge
        mult = driver_info.get("multiplicity", 1) if mult is None else mult
        # initialize the parent Calculator class with charge and multiplicity
        super().__init__(charge=charge, mult=mult, **kwargs)
        self.driver = driver_obj

    def prepare_atoms(self, atoms: List[str], coords: np.ndarray, **prepare_kwargs) -> Atoms:
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

    def get_energy(self, atoms: List[str], coords: np.ndarray, **prepare_kwargs) -> dict:
        ase_atoms = self.prepare_atoms(atoms, coords, **prepare_kwargs)
        energy, unit = self.driver._compute_energy_impl(ase_atoms)
        results = {}
        if unit == "Eh":
            results["energy"] = energy
        elif unit == "eV":
            results["energy"] = energy / Hartree
        else:
            raise ValueError("Invalid energy unit. Use 'Eh' or 'eV'.")
        return results

    def get_forces(self, atoms: List[str], coords: np.ndarray, **prepare_kwargs) -> np.ndarray:
        # energy is required when pysisyphus calls get_forces
        results = self.get_energy(atoms, coords, **prepare_kwargs)
        ase_atoms = self.prepare_atoms(atoms, coords, **prepare_kwargs)
        forces, unit = self.driver._compute_forces_impl(ase_atoms)
        if unit == "Eh/Bohr":
            results["forces"] = forces.reshape(-1)
        elif unit == "eV/Ang":  # convert from eV/Ang to Eh/Bohr
            results["forces"] = forces.reshape(-1) * (Bohr / Hartree)
        else:
            raise ValueError("Invalid forces unit. Use 'eV/Ang' or 'Eh/Bohr'.")
        return results

    def get_hessian(self, atoms: List[str], coords: np.ndarray, **prepare_kwargs) -> np.ndarray:
        # energy is required when pysisyphus calls get_hessian
        results = self.get_energy(atoms, coords, **prepare_kwargs)
        ase_atoms = self.prepare_atoms(atoms, coords, **prepare_kwargs)
        hessian = self.driver.compute_hessian(atoms=ase_atoms, use_cache=True, hess_format="ase")
        hessian *= Bohr**2 / Hartree  # convert from eV/Ang^2 to Eh/Bohr^2
        results["hessian"] = hessian
        return results
