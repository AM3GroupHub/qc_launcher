"""Helpers shared by the task runners."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import ase.io
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree

from qc_launcher.drivers import BaseDriver

FMAX_DEFAULT = 4.5e-4 * Hartree / Bohr  # Gaussian's maximum-force criterion, in eV/Angstrom


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Print ``<label> completed in N seconds.`` when the block exits (also on error)."""
    start = time.time()
    try:
        yield
    finally:
        print(f"{label} completed in {time.time() - start:.2f} seconds.")


def sella_hessian_function(driver: BaseDriver) -> Callable[[Atoms], np.ndarray]:
    """Exact-Hessian callback for Sella: ASE layout in eV/Angstrom^2, cached per geometry."""

    def hessian(atoms: Atoms) -> np.ndarray:
        return driver.compute_hessian(atoms, use_cache=True, hess_format="ase")

    return hessian


def check_same_species(atoms_list: Sequence[Atoms]) -> None:
    first = atoms_list[0].get_chemical_symbols()
    if any(atoms.get_chemical_symbols() != first for atoms in atoms_list[1:]):
        raise ValueError("All images must have the same atoms (same symbols in the same order).")


def relax_endpoints(
    images: Sequence[Atoms],
    calc: Calculator,
    *,
    fmax: float,
    steps: int | None,
    fixed_reactant: bool,
    fixed_product: bool,
) -> None:
    """FIRE-relax the first and last image in place; ``steps`` limits the optimizer (None: unlimited)."""
    from ase.optimize import FIRE

    run_kwargs = {"fmax": fmax} if steps is None else {"fmax": fmax, "steps": int(steps)}
    for label, image, fixed in (("initial", images[0], fixed_reactant), ("final", images[-1], fixed_product)):
        if fixed:
            continue
        image.calc = calc
        print(f"Optimizing {label} image...")
        with FIRE(image) as opt:
            opt.run(**run_kwargs)


def write_path_results(images: Sequence[Atoms], filename: str) -> int:
    """Write ``{filename}_neb.xyz`` and ``{filename}_ts.xyz`` and print the profile; returns the TS index."""
    ase.io.write(f"{filename}_neb.xyz", images)
    energies = np.array([image.get_potential_energy() for image in images])
    for i, energy in enumerate(energies):
        print(f"Image {i:02d}: Energy = {energy:.6f} eV")
    ts_index = int(np.argmax(energies))
    print(f"Transition state is image {ts_index:02d} with energy {energies[ts_index]:.6f} eV")
    ase.io.write(f"{filename}_ts.xyz", images[ts_index])
    return ts_index
