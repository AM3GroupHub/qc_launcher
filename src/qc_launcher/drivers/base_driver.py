"""Abstract driver interface shared by all backends, plus the unit helpers they rely on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree

HessianFormat = Literal["pyscf", "ase"]

ENERGY_TO_EV = {"eV": 1.0, "Eh": Hartree}
FORCES_TO_EV_ANG = {"eV/Ang": 1.0, "Eh/Bohr": Hartree / Bohr}
HESSIAN_AU_TO_ASE = Hartree / Bohr**2  # Eh/Bohr^2 -> eV/Ang^2


def to_ev(value: float, unit: str) -> float:
    """Convert an energy given in ``unit`` ('eV' or 'Eh') to eV."""
    try:
        return value * ENERGY_TO_EV[unit]
    except KeyError:
        raise ValueError(f"Invalid energy unit '{unit}'. Use 'eV' or 'Eh'.") from None


def to_ev_per_ang(forces: np.ndarray, unit: str) -> np.ndarray:
    """Convert forces given in ``unit`` ('eV/Ang' or 'Eh/Bohr') to eV/Ang."""
    try:
        return np.asarray(forces) * FORCES_TO_EV_ANG[unit]
    except KeyError:
        raise ValueError(f"Invalid forces unit '{unit}'. Use 'eV/Ang' or 'Eh/Bohr'.") from None


def convert_hessian(hessian: np.ndarray, natm: int, hess_format: HessianFormat) -> np.ndarray:
    """Convert between the two Hessian layouts used in this package.

    ``"ase"`` is ``(3N, 3N)`` in eV/Ang^2; ``"pyscf"`` is ``(N, N, 3, 3)`` in Eh/Bohr^2. The input layout
    is inferred from the shape, so passing a Hessian that is already in ``hess_format`` returns it as is.
    """
    if hess_format not in ("pyscf", "ase"):
        raise ValueError("Invalid hess_format. Use 'pyscf' or 'ase'.")
    hessian = np.asarray(hessian)
    is_pyscf = hessian.shape == (natm, natm, 3, 3)
    is_ase = hessian.shape == (3 * natm, 3 * natm)
    if not (is_pyscf or is_ase):
        raise ValueError(f"Invalid Hessian shape {hessian.shape} for {natm} atoms.")
    if hess_format == "pyscf":
        if is_pyscf:
            return hessian
        return hessian.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3) / HESSIAN_AU_TO_ASE
    if is_ase:
        return hessian
    return hessian.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm) * HESSIAN_AU_TO_ASE


def dummy_pyscf_mf(atoms: Atoms, e_tot_eh: float) -> SimpleNamespace:
    """A stand-in for a PySCF mean-field object exposing only ``mol`` and ``e_tot`` (for thermochemistry)."""
    from pyscf import gto

    mol = gto.M(
        atom=[(symbol, position) for symbol, position in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
        charge=atoms.info.get("charge", 0),
        spin=atoms.info.get("multiplicity", 1) - 1,
    )
    return SimpleNamespace(mol=mol, e_tot=e_tot_eh)


class BaseDriver(ABC):
    """A backend that evaluates energies, forces and Hessians for one molecular system.

    Subclasses implement ``_compute_*_impl`` returning values together with their unit, and
    ``to_ase_calc`` / ``to_pyscf_mf`` adapters. The public ``compute_*`` methods convert units,
    log, and cache Hessians by geometry.
    """

    def __init__(self, atoms: Optional[Atoms], config: dict, verbose: bool = True):
        self.atoms: Optional[Atoms] = atoms
        self.config = config
        self.verbose = verbose
        self._hessian_cache: dict[bytes, np.ndarray] = {}
        self._last_positions: Optional[np.ndarray] = None

    def log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------ structure bookkeeping
    def update_atoms(self, atoms: Optional[Atoms]) -> None:
        """Adopt ``atoms`` as the current structure; ``None`` keeps the current one.

        Species, charge and multiplicity must not change (create a new driver for that). Whether the
        *positions* changed is tracked separately by :meth:`_positions_changed`, because callers such as
        Sella mutate the very same ``Atoms`` object in place between calls.
        """
        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure available: pass an Atoms object.")
            return
        if self.atoms is not None and self.atoms is not atoms:
            if not np.array_equal(atoms.get_atomic_numbers(), self.atoms.get_atomic_numbers()):
                raise ValueError("Atomic numbers cannot be changed. Please create a new driver instance.")
            for key, default in (("charge", 0), ("multiplicity", 1)):
                if atoms.info.get(key, default) != self.atoms.info.get(key, default):
                    raise ValueError(f"{key.capitalize()} cannot be changed. Please create a new driver instance.")
        self.atoms = atoms

    def _positions_changed(self, atol: float = 1e-8) -> bool:
        """True unless the current positions equal the snapshot taken by :meth:`_mark_positions`."""
        if self.atoms is None:
            raise ValueError("No structure available.")
        if self._last_positions is None:
            return True
        positions = self.atoms.get_positions()
        return positions.shape != self._last_positions.shape or not np.allclose(
            positions, self._last_positions, rtol=0.0, atol=atol
        )

    def _mark_positions(self) -> None:
        """Snapshot the current positions as the geometry the backend state corresponds to."""
        self._last_positions = self.atoms.get_positions().copy()

    def clear_cache(self) -> None:
        self._hessian_cache.clear()
        self._last_positions = None

    # ------------------------------------------------------------------ adapters
    @abstractmethod
    def to_ase_calc(self) -> Calculator:
        """An ASE calculator backed by this driver."""

    @abstractmethod
    def to_pyscf_mf(self):
        """A PySCF mean-field object (or a :func:`dummy_pyscf_mf` stand-in) for thermochemistry."""

    # ------------------------------------------------------------------ backend implementations
    @abstractmethod
    def _compute_energy_impl(self, atoms: Optional[Atoms]) -> tuple[float, str]:
        """Return (energy, unit) with unit 'eV' or 'Eh'."""

    @abstractmethod
    def _compute_forces_impl(self, atoms: Optional[Atoms]) -> tuple[np.ndarray, str]:
        """Return (forces of shape (N, 3), unit) with unit 'eV/Ang' or 'Eh/Bohr'."""

    @abstractmethod
    def _compute_hessian_impl(self, atoms: Optional[Atoms]) -> np.ndarray:
        """Return the Hessian in either layout accepted by :func:`convert_hessian`."""

    # ------------------------------------------------------------------ public API
    def compute_energy(self, atoms: Optional[Atoms] = None) -> float:
        """Potential energy in eV."""
        energy_ev = to_ev(*self._compute_energy_impl(atoms))
        self.log(f"Total Energy        [eV]: {energy_ev:16.10f}")
        self.log(f"Total Energy        [Eh]: {energy_ev / Hartree:16.10f}")
        return energy_ev

    def compute_forces(self, atoms: Optional[Atoms] = None) -> np.ndarray:
        """Forces in eV/Ang, shape (N, 3)."""
        forces = to_ev_per_ang(*self._compute_forces_impl(atoms))
        self.log("Forces [eV/Angstrom]:")
        for i, f in enumerate(forces):
            self.log(f"{i:3d} {f[0]:12.6f} {f[1]:12.6f} {f[2]:12.6f}")
        return forces

    def compute_hessian(
        self,
        atoms: Optional[Atoms] = None,
        use_cache: bool = True,
        hess_format: HessianFormat = "ase",
    ) -> np.ndarray:
        """Hessian in ``hess_format``; cached per geometry (positions rounded to 1e-4 Ang).

        The returned array is always a private copy, so callers may modify it freely.
        """
        if hess_format not in ("pyscf", "ase"):
            raise ValueError("Invalid hess_format. Use 'pyscf' or 'ase'.")
        atoms = self.atoms if atoms is None else atoms
        natm = len(atoms)
        key = self._cache_key(atoms.get_positions())
        if use_cache and key in self._hessian_cache:
            return convert_hessian(self._hessian_cache[key], natm, hess_format).copy()
        hessian = convert_hessian(self._compute_hessian_impl(atoms), natm, hess_format)
        if use_cache:
            self._hessian_cache[key] = hessian.copy()
        return hessian

    @staticmethod
    def _cache_key(positions: np.ndarray, tolerance: float = 1e-4) -> bytes:
        return np.round(np.asarray(positions) / tolerance).astype(np.int64).tobytes()
