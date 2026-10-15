"""Test doubles shared by the unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from qc_launcher.drivers import BaseDriver


class HarmonicCalculator(Calculator):
    """E = k/2 * sum |r|^2 (eV), F = -k r."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties or self.implemented_properties, system_changes)
        pos = self.atoms.get_positions()
        self.results = {"energy": 0.5 * self.k * float((pos**2).sum()), "forces": -self.k * pos}


class FakeDriver(BaseDriver):
    """Harmonic-trap driver with analytic forces and Hessian; counts calls for cache tests."""

    def __init__(self, atoms: Atoms | None, config: dict | None = None, verbose: bool = False):
        super().__init__(atoms, dict(config or {}), verbose)
        self.k = float(self.config.get("k", 1.0))
        self.energy_calls = 0
        self.hessian_calls = 0

    def to_ase_calc(self):
        return HarmonicCalculator(self.k)

    def to_pyscf_mf(self):
        from pyscf import gto

        mol = gto.M(
            atom=[(s, p) for s, p in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions())],
            charge=self.atoms.info.get("charge", 0),
            spin=self.atoms.info.get("multiplicity", 1) - 1,
            verbose=0,
        )
        return SimpleNamespace(mol=mol, e_tot=self._compute_energy_impl(None)[0] / 27.211386)

    def _compute_energy_impl(self, atoms=None):
        self.update_atoms(atoms)
        self.energy_calls += 1
        pos = self.atoms.get_positions()
        return 0.5 * self.k * float((pos**2).sum()), "eV"

    def _compute_forces_impl(self, atoms=None):
        self.update_atoms(atoms)
        return -self.k * self.atoms.get_positions(), "eV/Ang"

    def _compute_hessian_impl(self, atoms):
        self.update_atoms(atoms)
        self.hessian_calls += 1
        return self.k * np.eye(3 * len(self.atoms))
