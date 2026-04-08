import os
import shutil
from typing import Any, Optional, Iterable

import ase.io
import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from ase.units import Bohr, Hartree
from pysisyphus.Geometry import Geometry
from pysisyphus.calculators.Calculator import Calculator as PysisyphusCalculator
from pysisyphus.cos.GrowingString import GrowingString
from pysisyphus.optimizers.StringOptimizer import StringOptimizer

from qc_launcher.drivers import BaseDriver


class QCLauncherPysisyphusCalculator(PysisyphusCalculator):
    def __init__(self, driver: BaseDriver, **kwargs):
        super().__init__(**kwargs)
        self.driver = driver

    def _make_target_atoms(self, atoms: Iterable[str], coords: np.ndarray) -> Atoms:
        coords3d = np.asarray(coords, dtype=float).reshape(-1, 3) * Bohr
        target = self.driver.atoms.copy()
        symbols = target.get_chemical_symbols()
        symbols_lower = [s.lower() for s in symbols]
        atoms_lower = [s.lower() for s in atoms]
        if symbols_lower != atoms_lower:
            raise ValueError("Atom symbols from the geometry do not match those of the driver.")
        target.set_positions(coords3d)
        return target

    def get_energy(self, atoms, coords, **prepare_kwargs):
        ase_atoms = self._make_target_atoms(atoms, coords)
        return {"energy": self.driver.compute_energy(ase_atoms) / Hartree}

    def get_forces(self, atoms, coords, **prepare_kwargs):
        ase_atoms = self._make_target_atoms(atoms, coords)
        energy = self.driver.compute_energy(ase_atoms) / Hartree
        forces = self.driver.compute_forces(ase_atoms) / (Bohr / Hartree)
        return {"energy": energy, "forces": forces.flatten()}

    def get_hessian(self, atoms, coords, **prepare_kwargs):
        ase_atoms = self._make_target_atoms(atoms, coords)
        energy = float(ase_atoms.get_potential_energy()) / Hartree
        hessian = self.driver.compute_hessian(ase_atoms, use_cache=True, hess_format="pyscf")
        if hessian.ndim == 4:
            hessian = np.hstack(np.concatenate(hessian, axis=1))
        return {
            "energy": energy,
            "hessian": hessian,
        }

def geom_to_ase_atoms(
    geom: Geometry, charge: int = 0, multiplicity: int = 1, energy: Optional[float] = None
) -> Atoms:
    pos = geom.coords3d * Bohr
    symbols_lower = list(geom.atoms)
    symbols = [s.capitalize() for s in symbols_lower]
    atoms = Atoms(symbols=symbols, positions=pos)
    if energy is None:
        energy = geom.energy * Hartree
    atoms.info["energy"] = energy
    atoms.info["charge"] = charge
    atoms.info["multiplicity"] = multiplicity
    return atoms


def run_gsm(
    driver: BaseDriver,
    config: dict,
    atoms_list: list[Atoms],
    filename: str = "molecule",
) -> None:
    assert len(atoms_list) == 2, "Exactly two geometries (reactant and product) must be provided."
    reactant, product = atoms_list[0], atoms_list[1]
    symbols = reactant.get_chemical_symbols()
    assert symbols == product.get_chemical_symbols(), "Reactant and product must have the same atoms in the same order."

    preopt = bool(config.get("preopt", False))
    coordinate = str(config.get("coordinate", "DLC")).lower()
    max_nodes = int(config.get("max_nodes", 7))
    fmax = float(config.get("fmax", 2.5e-3 * Hartree / Bohr))
    frms = float(config.get("frms", 1.7e-3 * Hartree / Bohr))
    perp_thresh = float(config.get("perp_thresh", 0.05 * Hartree / Bohr))  # set in eV/Ang
    max_cycles = int(config.get("max_cycles", 64))
    dx = float(config.get("dx", 0.1 * Bohr))  # set in Angstrom
    reparam_every = int(config.get("reparam_every", 2))
    reparam_every_full = int(config.get("reparam_every_full", 3))
    climb = bool(config.get("climb", True))
    fix_ends = bool(config.get("fix_ends", True))
    stop_in_when_full = int(config.get("stop_in_when_full", -1))
    trajectory = config.get("trajectory", f"{filename}_gsm.xyz")
    ts_output = config.get("ts_output", f"{filename}_ts.xyz")

    # create a temporary directory for any intermediate files if needed
    sisyphus_calc = QCLauncherPysisyphusCalculator(
        driver=driver,
        base_name=f"{filename}_gsm",
        keep_kind="none",
        clean_after=True,
    )

    if preopt:
        calc = driver.to_ase_calc()
        reactant.calc = calc
        print("Optimizing initial image...")
        with FIRE(reactant) as opt:
            opt.run(fmax=fmax)
        product.calc = calc
        print("Optimizing final image...")
        with FIRE(product) as opt:
            opt.run(fmax=fmax)

    reactant_geom = Geometry(
        atoms=symbols,
        coord_type=coordinate,
        coords=reactant.get_positions() / Bohr,
    )
    product_geom = Geometry(
        atoms=symbols,
        coord_type=coordinate,
        coords=product.get_positions() / Bohr,
    )
    reactant_geom.set_calculator(sisyphus_calc)
    product_geom.set_calculator(sisyphus_calc)

    cos = GrowingString(
        [reactant_geom, product_geom],
        calc_getter=lambda: sisyphus_calc,
        max_nodes=max_nodes,
        perp_thresh=perp_thresh / (Hartree / Bohr),  # convert from eV/Ang to Hartree/Bohr
        reparam_every=reparam_every,
        reparam_every_full=reparam_every_full,
        climb=climb,
        fix_first=fix_ends,
        fix_last=fix_ends,
    )

    optimizer = StringOptimizer(
        cos,
        max_cycles=max_cycles,
        max_step=dx / Bohr,  # convert from Ang to Bohr
        rms_force=frms / (Hartree / Bohr),  # convert from eV/Ang to Hartree/Bohr
        stop_in_when_full=stop_in_when_full,
        dump=False,
        dump_restart=False,
        prefix=f"{filename}_gsm",
    )
    # assign a void file to the optimizer's final_fn
    optimizer.run()

    charge = reactant.info.get("charge", 0)
    multiplicity = reactant.info.get("multiplicity", 1)
    atoms_traj = [geom_to_ase_atoms(image, charge=charge, multiplicity=multiplicity) for image in cos.images]
    ase.io.write(trajectory, atoms_traj)

    hei_coords, hei_energy, _, _ = cos.get_splined_hei()
    atoms_ts = Atoms(symbols=symbols, positions=hei_coords.reshape(-1, 3) * Bohr)
    atoms_ts.info["charge"] = reactant.info["charge"]
    atoms_ts.info["multiplicity"] = reactant.info["multiplicity"]
    atoms_ts.info["energy"] = float(hei_energy * Hartree)
    ase.io.write(ts_output, atoms_ts)

    # Clean up any temporary files created by the calculator and optimizer
    shutil.rmtree(sisyphus_calc.out_dir)
    os.remove(optimizer.final_fn)
