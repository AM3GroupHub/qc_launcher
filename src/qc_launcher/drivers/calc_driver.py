"""Shared base for drivers whose energies and forces come from an ASE calculator (the MLIP backends)."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Hartree

from .base_driver import BaseDriver, dummy_pyscf_mf


class CalculatorDriver(BaseDriver):
    """Energies and forces are delegated to the ASE calculator returned by :meth:`build_calc`.

    Subclasses implement ``build_calc`` and ``_compute_hessian_impl``. The calculator is built lazily on
    first use and reused; ``driver.calc`` exposes it.
    """

    def __init__(self, atoms: Atoms | None, config: dict, **kwargs):
        super().__init__(atoms=atoms, config=config, **kwargs)
        self._calc: Calculator | None = None
        self._add_spin_tag()

    @abstractmethod
    def build_calc(self) -> Calculator:
        """Construct the ASE calculator from ``self.config``."""

    @property
    def calc(self) -> Calculator:
        if self._calc is None:
            self._calc = self.build_calc()
        return self._calc

    @staticmethod
    def resolve_device(config: dict) -> str:
        """``config['device']`` if given, else 'cuda' when available, else 'cpu'."""
        device = config.get("device")
        if device is not None:
            return str(device)
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _add_spin_tag(self) -> None:
        """MACE and FAIRChem read the spin multiplicity from ``atoms.info['spin']``."""
        if self.atoms is not None:
            self.atoms.info["spin"] = self.atoms.info.get("multiplicity", 1)

    def update_atoms(self, atoms: Atoms | None) -> None:
        super().update_atoms(atoms)
        self._add_spin_tag()

    def to_ase_calc(self) -> Calculator:
        return self.calc

    def to_pyscf_mf(self):
        if self.atoms is None:
            raise ValueError("Atoms must be provided to build the pyscf stand-in.")
        return dummy_pyscf_mf(self.atoms, self.calc.get_potential_energy(self.atoms) / Hartree)

    def _compute_energy_impl(self, atoms: Atoms | None = None) -> tuple[float, str]:
        self.update_atoms(atoms)
        return float(self.calc.get_potential_energy(self.atoms)), "eV"

    def _compute_forces_impl(self, atoms: Atoms | None = None) -> tuple[np.ndarray, str]:
        self.update_atoms(atoms)
        return np.asarray(self.calc.get_forces(self.atoms)), "eV/Ang"
